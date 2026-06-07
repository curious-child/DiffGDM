import math
from operator import itemgetter
import torch
import numpy as np
import torch_geometric.utils as tg_utils
import networkx as nx
from torch_geometric.data import Batch,Data
from itertools import filterfalse
from torch_geometric.utils import k_hop_subgraph
from torch.distributions import Transform, constraints
from models.Diffusion_model.graph_matrix_utils import target_eigenvalue_guided, WR_energy_guided
import torch.nn.functional as F


#待修改 用2核去判断 全局判断
def fvs_decode_np(preds, graph_data):
  """Decode the labels to the FVS."""
  net_G=tg_utils.to_networkx(graph_data,to_undirected=True)

  solution = np.zeros_like(preds.astype(int))
  # 下几行将预测值与网络节点序号做映射，因此无需考虑网络节点序号问题
  predictions = list(zip(list(net_G.nodes), preds))
  sorted_prediction = sorted(predictions, key=itemgetter(1), reverse=True) #降序排列
  removal_order = list(map(itemgetter(0), sorted_prediction))
  removal_order = list(map(int, removal_order))

  #先判断 预删除节点是否在节点的环路里

  net_G = nx.k_core(net_G, k=2)
  core_nodes_set = set(net_G.nodes())
  core_nodes = len(core_nodes_set)

  while core_nodes:

    remove_counts = math.ceil(core_nodes * 0.01)
    remove_nodes, removal_order = extract_list_del(removal_order, end=remove_counts)
    real_remove_nodes=list(set(remove_nodes)&core_nodes_set)
    net_G.remove_nodes_from(real_remove_nodes)
    solution[real_remove_nodes]=1  # 注意，此处仅考虑图是从节点序号为0开始的！！后续可改善

    net_G = nx.k_core(net_G, k=2)
    core_nodes_set = set(net_G.nodes())
    core_nodes = len(core_nodes_set)

  return (solution == 1).astype(int)
def guided_cost(pred_nodes,edge_index,device="cuda",infer_direction="FVS",**kwargs):

    num_nodes = pred_nodes.shape[0]
    num_edges = edge_index.shape[1]
    ##
    # FVS的约束条件难以表示，因此简化其诱导方案
    # 方案一：利用生成模型 诱导其FVS相关能量函数
    if infer_direction == "MVC":
        # #方案二：考虑其余的诱导条件，使模型的解更倾向于 网络拆解与解环
        #   ##一（重点）：将模型倾向于诱导为最小顶点覆盖问题，（网络拆解问题的上界为最小顶点覆盖，下界为网络解环问题）
        pred_nodes = pred_nodes.reshape(-1, 1).to(device)
        Q = QUBO_MVC(num_nodes, num_edges, edge_index, P=kwargs["MVC_P"], device=device)
        cost_est = pred_nodes.t() @ Q @ pred_nodes / (num_nodes + kwargs["MVC_P"]* num_edges)
    elif  infer_direction in ["LCC","global_CC","average_CC","natural_connectivity","global_efficiency","degree_assortativity_coefficient","density"]:#用神经网络对网络鲁棒性指标进行预测

        guided_model=kwargs["guided_model"].to(device)

        guided_model.eval()
        data_graph = Data(x=pred_nodes, edge_index=edge_index).to(device)
        cost_est = guided_model(data_graph)

    else:
        raise Exception("infer_direction has not been defined", infer_direction)
    return cost_est

def cumulative_density_function(x, mu, sigma, height, offset):
  return (np.arctan((x - mu) / sigma) / np.pi + 0.5) * height + offset
def median(mu,sigma,height,offset):
    half_point=(height+offset)/2
    return np.tan(((half_point-offset)/height-0.5)*np.pi)*sigma+mu
class TanTransform(Transform):
    bijective = True
    domain = constraints.real
    codomain = constraints.real
    sign = 1

    def _call(self, x):
        return torch.tan(x)

    def _inverse(self, y):
        return torch.atan(y)

    def log_abs_det_jacobian(self, x, y):
        return 2 * torch.log(torch.abs(torch.cos(x)))


def get_codomain(function, tmin, tmax, *args):

    imax = function(tmax, *args)
    imin = function(tmin, *args)

    return imin, imax
def grad_convert(grad_xt,xt):
    """
    将梯度转换为[xt.shape[0],2]  的格式。
    """
    xt_grad_zero, xt_grad_one = torch.zeros(xt.shape, device=xt.device).unsqueeze(-1).repeat(1, 2), \
        torch.zeros(xt.shape, device=xt.device).unsqueeze(-1).repeat(1, 2)
    xt_grad_zero[..., 0] = (1 - xt) * grad_xt
    xt_grad_zero[..., 1] = -xt_grad_zero[..., 0]
    xt_grad_one[..., 1] = xt * grad_xt
    xt_grad_one[..., 0] = -xt_grad_one[..., 1]
    grad_xt = xt_grad_zero + xt_grad_one  # [b*N,2]
    return grad_xt
class model_inference:
    def __init__(self,model,edge_index,t,diffusion_type,device,batch_index=None,atbar=None):
        self.model=model.to(device)
        self.edge_index=edge_index
        self.t=t
        self.device=device
        self.batch_index=batch_index
        self.diffusion_type=diffusion_type
        self.atbar=atbar
    def __call__(self,xt):
        if self.diffusion_type=="categorical":
            xt_scale = xt * 2 - 1
            xt_scale = xt_scale * (1.0 + 0.05 * torch.rand_like(xt.float(), device=self.device))
            x0_pred=self.model(xt_scale.reshape(-1).float(), self.t.to(self.device), edge_index=self.edge_index.to(self.device), batch_index=self.batch_index)
            x0_pred_prob = x0_pred.softmax(dim=-1)
            x0_pred_expect = x0_pred_prob[..., 1].reshape(-1)
        elif self.diffusion_type=="gaussian":
            epsilon_pred =self.model(xt.reshape(-1).float(), self.t.to(self.device), edge_index=self.edge_index.to(self.device), batch_index=self.batch_index)
            epsilon_pred = epsilon_pred.reshape(-1)
            x0_pred_expect = torch.sqrt(1 / self.atbar) * (xt - torch.sqrt(1 - self.atbar) * epsilon_pred)

        else:
            raise Exception("diffusion_type has not been defined", self.diffusion_type)
        return x0_pred_expect

class model_target_update:
    def __init__(self,optimizer_params):
        self.target_update_lr=optimizer_params["target_update_lr"]
        self.weight_decay=optimizer_params["target_update_wd"]
        self.update_epoches=optimizer_params["target_update_epoches"]
        self.scheduler_gamma=optimizer_params["update_scheduler_gamma"]
        self.schduler_step_size=optimizer_params["update_scheduler_step"]

    def __call__(self,model,pred_nodes, edge_index,target_truth):

        optimizer = torch.optim.Adam(model.parameters(), lr=self.target_update_lr, weight_decay=self.weight_decay)
        weight_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.schduler_step_size,
                                                           gamma=self.scheduler_gamma)


        for epoch in range(self.update_epoches):
            optimizer.zero_grad()
            data_graph = Data(x=pred_nodes, edge_index=edge_index).to(pred_nodes.device)
            pred_target = model(data_graph).to(pred_nodes.device)
            loss = F.mse_loss(pred_target, target_truth)
            loss.backward(retain_graph=True)
            optimizer.step()
            weight_scheduler.step()
        return model

def list_del(orign_list,del_elements):
    for del_element in del_elements:
        orign_list.remove(del_element)
    return orign_list
def extract_list_del(l1,end,start=0,):
    l2=l1[start:end]
    del l1[start:end]
    return l2,l1
def QUBO_MVC(num_nodes,num_edges,edge_index, P, device):
    v = torch.ones(num_nodes).to(device)
    i = torch.cat((
      torch.arange(num_nodes).unsqueeze(0).to(device),
      torch.arange(num_nodes).unsqueeze(0).to(device)
    ), 0)
    Q = torch.sparse_coo_tensor(indices=i, values=v, size=(num_nodes, num_nodes))



    P_nodes_index = torch.cat(
      (edge_index[0].unsqueeze(0).to(device), edge_index[0].unsqueeze(0).to(device)), 0
    )
    P_nodes = P * torch.ones(num_edges).to(device)
    Q = Q - torch.sparse_coo_tensor(indices=P_nodes_index, values=P_nodes,
                                    size=(num_nodes, num_nodes))

    P_edges = P / 2 * torch.ones(num_edges).to(device)
    Q = Q + torch.sparse_coo_tensor(indices=edge_index, values=P_edges,
                                    size=(num_nodes, num_nodes))
    Q.coalesce()  # pytorch中关于两个稀疏tensor的求和运算只是简单地对各自的indices和values进行叠加  所以，再执行完加法运算后，使用.coalesce()进行压缩：

    return Q


def select_subgraphs(graph, num_neighbors=5, omega=3):

    """
    使用 `ov` 向量分解图为子图集合，同时保留原图节点序号对齐。

    Args:
        graph: 输入图，使用 torch_geometric.data.Data 表示。
        k_neighbors: k近邻规则中 k 的值。
        omega: 停止条件，`min(ov)` 的阈值。

    Returns:
        子图列表，其中每个子图为 torch_geometric.data.Data 对象，保留原图节点索引。
    """

    num_nodes = graph.num_nodes
    ov = torch.zeros(num_nodes, dtype=torch.int)  # 初始化每个节点的出现次数为 0
    subgraphs = []

    while ov.min().item() < omega:
        # Step 1: 根据 `argmin(ov)` 选择当前聚类中心
        cluster_center = torch.argmin(ov).item()

        # Step 2: 使用 k近邻规则，选择 cluster_center 的 k 个最近邻节点
        selected_nodes = torch.tensor([cluster_center])
        selected_edge_index = torch.tensor([])
        k_hop = 1
        while selected_nodes.size(0) < num_neighbors:
            subset, subgraph_edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx=cluster_center, num_hops=k_hop, edge_index=graph.edge_index, relabel_nodes=False)
            if subset.size(0) > 2 * num_neighbors and k_hop >= 2:
                break
            else:
                selected_nodes = subset
                selected_edge_index = subgraph_edge_index
            k_hop += 1
        selected_nodes = selected_nodes.unique()  # 去重
        selected_nodes,_ = torch.sort(selected_nodes)  # 排序 默认升序
        # Step 3: 构造子图，并保留原图中的节点编号
        subgraph_data = Data(
            x=graph.x[selected_nodes],
            edge_index=selected_edge_index,
            node_idx=selected_nodes  # 保留原图中的节点索引
        )
        subgraphs.append(subgraph_data)

        # Step 4: 更新 `ov` 向量
        ov[selected_nodes] += 1

    return subgraphs, ov