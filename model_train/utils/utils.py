import copy
import json
import math
import os
import random

from scipy import stats
import itertools as it
import igraph
import numpy as np
import networkx as nx
import torch_geometric
import torch_geometric.utils as tg_utils
from torch_geometric.data import Data, Batch, HeteroData
import torch
import numpy as np
from igraph import *
import matplotlib.pyplot as plt
from glob import glob
from sklearn.manifold import TSNE
import yaml
from torch_geometric.utils import k_hop_subgraph

from models.models import graph_models, diffusion_models


def grid_parameters_supervised_learning(train_params,net_params,loss_params,optimizer_params,**kwargs):
    parameters_list=[]
    for params_values in it.product(*loss_params.values()):
        loss_param = dict(zip(loss_params.keys(), params_values))
        for params_values in it.product(*train_params.values()):
            train_param = dict(zip(train_params.keys(), params_values))
            #  print(train_param)

            net_params_copy = net_params.copy()
            net_params_copy.pop("prelayers_gnn_param")
            net_params_copy.pop("enclayers_gnn_param")
            #net_params_copy.pop("mask_model")
            for params_values in it.product(*net_params_copy.values()):
                net_param = dict(zip(net_params_copy.keys(), params_values))

                # print(net_param["prelayers_gnn"])
                # print(net_param["enclayers_gnn"])
                for prelayers_values in it.product(*net_params["prelayers_gnn_param"][net_param["prelayers_gnn"]].values()):
                    net_param["prelayers_gnn_param"] = dict(
                        zip(net_params["prelayers_gnn_param"][net_param["prelayers_gnn"]].keys(), prelayers_values))
                    # print(net_param["prelayers_gnn_param"])
                    for enclayers_values in it.product(
                            *net_params["enclayers_gnn_param"][net_param["enclayers_gnn"]].values()):
                        net_param["enclayers_gnn_param"] = dict(
                            zip(net_params["enclayers_gnn_param"][net_param["enclayers_gnn"]].keys(), enclayers_values))

                        for params_values in it.product(*optimizer_params.values()):
                            optimizer_param = dict(zip(optimizer_params.keys(), params_values))
                            parameters_list.append((train_param,net_param,loss_param,optimizer_param))

    return parameters_list

def record_scores_analysis(configs_record_scores,metrics,train_validation,score_thershold):
    best_scores = float('-inf')
    best_config = None
    score_thershold_config = dict()
    for config_name,record_scores in configs_record_scores.items():
        cmp_best_score = max(record_scores[metrics][train_validation])
        if cmp_best_score > best_scores:
            best_scores = cmp_best_score
            best_config = config_name
        if cmp_best_score >= score_thershold:
            score_thershold_config[config_name] = cmp_best_score
    print("Best config:", best_config)
    print("Best scores:", best_scores)
    for n,config_name in enumerate(score_thershold_config.keys()):

        print("Score thershold config {}:\n".format(n), config_name)
    return best_config, best_scores,score_thershold_config
def grid_parameters_generative_learning(train_params, net_params, loss_params, optimizer_params, **kwargs):
    parameters_list = []

    for params_values in it.product(*train_params.values()):
        train_param = dict(zip(train_params.keys(), params_values))
        #  print(train_param)

        net_params_copy = net_params.copy()
        net_params_copy.pop("GNNEncoder_param")
        for params_values in it.product(*net_params_copy.values()):
            net_param = dict(zip(net_params_copy.keys(), params_values))

            if net_param["GNNEncoder_set"]=="T2TGNNEncoder":
                GNNEncoder_params__copy=net_params["GNNEncoder_param"][net_param["GNNEncoder_set"]].copy()
                GNNEncoder_params__copy.pop("gconv_param")
                for GNNEncoder_values in it.product(*GNNEncoder_params__copy.values()):
                    net_param["GNNEncoder_param"] = dict(
                        zip(GNNEncoder_params__copy.keys(), GNNEncoder_values))

                    for gconv_values in it.product(*net_params["GNNEncoder_param"]["T2TGNNEncoder"]["gconv_param"][net_param["GNNEncoder_param"]["gconv"]].values()):
                        net_param["GNNEncoder_param"]["gconv_param"]= dict(
                        zip(net_params["GNNEncoder_param"]["T2TGNNEncoder"]["gconv_param"][net_param["GNNEncoder_param"]["gconv"]].keys(), gconv_values) )
                        for params_values in it.product(*loss_params.values()):
                            loss_param = dict(zip(loss_params.keys(), params_values))
                            #  print("policy gradient:", loss_param["sample"])
                            for params_values in it.product(*optimizer_params.values()):
                                optimizer_param = dict(zip(optimizer_params.keys(), params_values))
                                parameters_list.append((copy.deepcopy(train_param), copy.deepcopy(net_param), copy.deepcopy(loss_param), copy.deepcopy(optimizer_param)))

            elif net_param["GNNEncoder_set"]=="NDGNNEncoder":
                GNNEncoder_params__copy = net_params["GNNEncoder_param"][net_param["GNNEncoder_set"]].copy()
                GNNEncoder_params__copy.pop("prelayers_gnn_param")
                GNNEncoder_params__copy.pop("enclayers_gnn_param")
                for params_values in it.product(*GNNEncoder_params__copy.values()):

                    net_param["GNNEncoder_param"] = dict(zip(GNNEncoder_params__copy.keys(), params_values))

                    for prelayers_values in it.product(*net_params["GNNEncoder_param"]["NDGNNEncoder"]["prelayers_gnn_param"][net_param["GNNEncoder_param"]["prelayers_gnn"]].values()):
                        net_param["GNNEncoder_param"]["prelayers_gnn_param"] = dict(
                            zip(net_params["GNNEncoder_param"]["NDGNNEncoder"]["prelayers_gnn_param"][net_param["GNNEncoder_param"]["prelayers_gnn"]].keys(), prelayers_values))
                        for enclayers_values in it.product(
                                *net_params["GNNEncoder_param"]["NDGNNEncoder"]["enclayers_gnn_param"][net_param["GNNEncoder_param"]["enclayers_gnn"]].values()):
                            net_param["GNNEncoder_param"]["enclayers_gnn_param"] = dict(
                                zip(net_params["GNNEncoder_param"]["NDGNNEncoder"]["enclayers_gnn_param"][net_param["GNNEncoder_param"]["enclayers_gnn"]].keys(), enclayers_values))


                            for params_values in it.product(*loss_params.values()):
                                loss_param = dict(zip(loss_params.keys(), params_values))
                                #  print("policy gradient:", loss_param["sample"])
                                for params_values in it.product(*optimizer_params.values()):
                                    optimizer_param = dict(zip(optimizer_params.keys(), params_values))
                                    parameters_list.append((copy.deepcopy(train_param), copy.deepcopy(net_param), copy.deepcopy(loss_param), copy.deepcopy(optimizer_param)))


    return parameters_list
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def gen_graph(g_type, num_min=20, num_max=40):
    max_n = num_max
    min_n = num_min
    cur_n = np.random.randint(max_n - min_n + 1) + min_n

    if g_type == 'erdos_renyi':
        while True:
            ER_p=random.uniform(0.1,0.9)
            g = Graph.Erdos_Renyi(n=cur_n,p=ER_p,directed=False)
            if g.is_connected():
                break
    elif g_type == 'small-world':
        while True:
            nei_max=round(0.35*cur_n)
            nei_min=round(0.15*cur_n)
            WS_nei=random.randint(nei_min,nei_max)
            WS_p = random.uniform(0, 0.15)
            g = Graph.Watts_Strogatz(dim=1,size=cur_n,nei=WS_nei,p=WS_p)#可包含最近邻规则图（p=0）
            if g.is_connected():
                break
    elif g_type == 'barabasi_albert':
        while True:
            m_max = round(0.25 * cur_n)
            m_min = round(0.1 * cur_n)
            BA_m = random.randint(m_min, m_max)
            g = Graph.Barabasi(n=cur_n, m=BA_m,directed=False)
            if g.is_connected():

                break
    elif g_type=="static_power_law":
        while True:
            exp=random.uniform(2,3)
            pl_max = round(0.25*cur_n* cur_n)
            pl_min = round(0.05*cur_n* cur_n)
            pl_edges=random.randint(pl_min,pl_max)
            g = Graph.Static_Power_Law(n=cur_n,m=pl_edges,exponent_out=exp)
            if g.is_connected():
                break
    elif g_type == "K_Regular":
        while True:
            k_regular=random.randint(round(0.2*cur_n),cur_n-2)  #可包含完全图（cur-1）
            if k_regular*cur_n %2==0 and cur_n>=(k_regular+1) : #k*n必须为偶数 且k+1 ≤n
                g = Graph.K_Regular(n=cur_n, k=k_regular)
                if g.is_connected():
                    break

    return g

def to_gnngraph(g,features,targets=None,SL=True):

    if "None" in features:
        x = np.ones((g.vcount(), 1))
    elif "one-hot" in features:
        x=np.identity(g.vcount())
    else:
        x = np.column_stack(
            tuple(
                g.vs[feature] for feature in features
            )
        )
    x = torch.from_numpy(x).to(torch.float)
    if SL:
        y = g.vs[targets]
        y = torch.tensor(y).to(torch.float)
    else:
        # G=igraph.Graph.to_networkx(g)
        # y=[pre_graph_op(G)]
        y=None


    source_nodes1 = [edge.source for edge in g.es]
    target_nodes1 = [edge.target for edge in g.es]
    source_nodes = source_nodes1 + target_nodes1
    target_nodes = target_nodes1 + source_nodes1

    return Data(x=x,y=y,edge_index=torch.tensor([source_nodes,target_nodes],dtype=torch.long))

def graph_properties(gnn_g,targets):
    node_v = 1 - gnn_g.x
    node_mask = torch.bernoulli(node_v).to(torch.bool)
    nodes_to_keep = torch.nonzero(node_mask).squeeze(1)
    subgraph = gnn_g.subgraph(nodes_to_keep)
    subgraph =tg_utils.to_networkx(subgraph, to_undirected=True)
    net_G = tg_utils.to_networkx(gnn_g, to_undirected=True)
    if targets == "LCC":
        return len(max(nx.connected_components(subgraph), key=len))
    elif targets == "global_CC":
        return nx.transitivity(subgraph)
    elif targets == "average_CC":
        return nx.average_clustering(subgraph)
    elif targets == "natural_connectivity":
        estrada_index = nx.estrada_index(subgraph)
        # 自然连通性
        n = nx.number_of_nodes(subgraph)
        return np.log(estrada_index / n)
    elif targets=="global_efficiency":
        return nx.global_efficiency(subgraph)
    elif targets=="density":
        return nx.density(subgraph)
    else:
        raise ValueError("Error fitness_func_type:{}".format(targets))
def to_gnngraph_nx(g,features,targets=None,SL=True):

    if "None" in features:
        gnn_g=torch_geometric.utils.from_networkx(g,group_node_attrs=None)
        gnn_g.x = torch.ones(gnn_g.num_nodes, 1).float()

    elif "random" in features:
        gnn_g = torch_geometric.utils.from_networkx(g, group_node_attrs=None)
        x0 = torch.rand(gnn_g.num_nodes)
        gnn_g.x = x0.reshape(-1,).float()
    else:
        gnn_g = torch_geometric.utils.from_networkx(g, group_node_attrs=features)
    if SL:

        if targets in g.nodes[0].keys():
            y_dict = nx.get_node_attributes(g,targets)
            gnn_g.y = torch.tensor(list(y_dict.values())).to(torch.long)
        else:
            gnn_g.y = torch.tensor(graph_properties(gnn_g,targets)).to(torch.float)
            gnn_g.y_name=targets
    else:
        #G=igraph.Graph.to_networkx(g)
        gnn_g.y=None
    return gnn_g
def pre_graph_op(G):
    G_ball_S = {}
    p = {}
    radius = 2
    for i in G.nodes():
        G_ball_S[i] = (get_neigbors(G, node=i, depth=radius)[radius])
        for j in G_ball_S[i]:
            p[i, j] = nx.shortest_paths.shortest_path(G, i, j)

    degree_dict = dict(G.degree())
    degree_tensor = torch.tensor(list(degree_dict.values())).reshape(-1, 1)
    d_1 = degree_tensor - torch.ones(size=degree_tensor.size())
    pdata = torch.tensor(list(p.values()))
    degree_k = d_1[pdata[:, -1].detach()].reshape(-1, 1)
    index = pdata[:, 0].reshape(-1, 1)
    y_dict={"d_1":d_1,
            "pdata":pdata,
            "degree_k":degree_k,
            "index":index}
    return y_dict
def get_neigbors(g, node, depth=1):

    output = {}
    layers = dict(nx.bfs_successors(g, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1,depth+1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x,[]))
        nodes = output[i]
    return output

def pre_DataSet( file_path=None,targets=None,filter="*",**params):
    from  tqdm import tqdm
    g_types_list=['erdos_renyi','small-world','barabasi_albert',"K_Regular","static_power_law"]
    dataSet=list()
    if file_path is not None:
        for file in tqdm(sorted(glob(file_path+'/'+filter+".graphml"))):
            try:
                nx_g=nx.read_graphml(file)
                nx_g=nx.convert_node_labels_to_integers(nx_g)
            except Exception:

                raise Exception("There are errors in {}".format(file))
           #g=igraph.Graph.Read_GraphML(file)
            # if g.vcount()<25:
            #     print("error node number:{}".format(g.vcount()))
            #     continue
            # if "random" in params["features"]:
            #    # for _ in range(10):
            #     g=to_gnngraph_nx(nx_g,features=params["features"],targets=targets,SL=params["SL"])
            #     dataSet.append(g)
            # else:
            g = to_gnngraph_nx(nx_g, features=params["features"], targets=targets, SL=params["SL"])
            dataSet.append(g)
    else:
        if  set(params["g_types"]).issubset(set(g_types_list)):

            for data_count in tqdm(range(params['data_num'])):
                for g_type in params["g_types"]:
                    g=gen_graph(g_type, num_min=20, num_max=40)
                    g = to_gnngraph(g, features=params["features"], targets=targets, SL=params["SL"])
                    dataSet.append(g)
        else:
            raise ValueError("Error:{} is not subset of g_type_list:{}".format(params["g_types"],g_types_list))
    return dataSet


def bron_kerbosch_with_limit(R, P, X, neighbors, max_clique_size, cliques):
    """
        改进的 Bron-Kerbosch 算法，支持最大团簇规模限制，使用集合加速操作。

        参数：
        - R: 当前团簇（集合，正在构建的子图）。
        - P: 候选节点集合（集合）。
        - X: 已处理的节点集合（集合）。
        - neighbors: 邻接表。
        - max_clique_size: 最大团簇规模限制。
        - cliques: 存储找到的团簇的集合。
        """
    if len(R) > max_clique_size:
        # 如果当前团簇超过规模限制，将其拆分为符合条件的小团簇
        cliques.update(split_clique(R, max_clique_size))  # 使用集合的更新操作
        return

    if not P and not X:
        # 如果没有候选节点和已处理节点，当前团簇是极大团簇
        cliques.add(frozenset(R))  # 使用 frozenset 确保集合可哈希
        return

    # 遍历候选节点 P 中的每个节点
    while P:
        v = P.pop()  # 从 P 中选择一个节点
        new_R = R | {v}  # 将 v 加入当前团簇
        new_P = P & neighbors[v]  # 计算与 v 相连的候选节点
        new_X = X & neighbors[v]  # 计算与 v 相连的已处理节点
        bron_kerbosch_with_limit(new_R, new_P, new_X, neighbors, max_clique_size, cliques)
        X.add(v)  # 将 v 移动到已处理节点集合


def split_clique(clique, max_clique_size):
    """
    将超出规模限制的团簇拆分为多个符合最大规模限制的小团簇。

    参数：
    - clique: 超出规模限制的团簇节点集合。
    - max_clique_size: 最大团簇规模限制。

    返回：
    - 小团簇列表，每个团簇不超过 max_clique_size。
    """
    from itertools import combinations
    # 枚举节点组合，确保每个组合是完全子图
    return {frozenset(sub_clique) for sub_clique in combinations(clique, max_clique_size)}


def find_cliques_with_limit(data, max_clique_size):
    """
    使用改进的 Bron-Kerbosch 算法寻找满足规模限制的团簇。

    参数：
    - data: torch_geometric.data.Data 图数据对象，包含 edge_index 和 num_nodes。
    - max_clique_size: 最大团簇规模限制。

    返回：
    - cliques: 找到的所有团簇列表。
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    # 构建邻接表
    neighbors = {i: set() for i in range(num_nodes)}
    for u, v in edge_index.t().tolist():
        neighbors[u].add(v)
        neighbors[v].add(u)  # 无向图需要双向添加
    # 过滤孤立节点
    neighbors = {node: adj for node, adj in neighbors.items() if adj}
    # 初始化 P（候选节点集合）为非孤立节点集合
    P = set(neighbors.keys())

    cliques = set()  # 使用集合存储团簇
    bron_kerbosch_with_limit(set(), P, set(), neighbors, max_clique_size, cliques)
    return cliques

def convert_simple_graph_to_factor_graph(orig_data,factor_dim,edge_dim,higher_order="inf",edge_attr=None):
    """
    Create a factor graph dataset using HeteroData from a simple graph with undirected edges and node features.

    Parameters:
    - orig_data: Data object containing the simple graph with node features and edge list
    - factor_dim: Dimension of the factor node features
    - edge_dim: Dimension of the edge features
    Returns:
    - data: HeteroData object containing all information of the factor graph
    """
    # Create HeteroData object
    if not tg_utils.is_undirected(orig_data.edge_index):

        raise ValueError("Simple graph must be undirected")
    node_features = orig_data.x
    data = HeteroData()

    # Add variable node features
    data['variable'].x = node_features
    data['variable'].y = orig_data.y
    data['variable', 'to', 'factor'].src4v_edge_index = orig_data.edge_index
    data['variable', 'to', 'factor'].tgt4v_edge_index = torch.flip(orig_data.edge_index, dims=[0])




    # Add connection relationships
    # Add edges from variable nodes to factor nodes
    factor_edge = []


    if higher_order =="inf":
        nx_graph = tg_utils.to_networkx(orig_data, to_undirected=True)
        cliques_G = nx.find_cliques(nx_graph)
        cliques = {frozenset(c) for c in cliques_G if len(c) > 1}
    else:
        cliques = find_cliques_with_limit(orig_data, max_clique_size=(higher_order + 1))
    factor_node_features = torch.zeros(len(cliques), factor_dim)  # features for factor nodes
    data['factor'].x = factor_node_features
    for factor_node, clique in enumerate(cliques):
        for var_node in clique:
            factor_edge.append([var_node, factor_node])

    factor_edge_index=torch.tensor(factor_edge,dtype=torch.long).t()

    data['variable', 'to', 'factor'].edge_index = factor_edge_index

    # Add edge attributes
    if edge_attr is None:
        data['variable', 'to', 'factor'].edge_attr = torch.ones(data['variable', 'to', 'factor'].edge_index.size(1), edge_dim)
    else:
        data['variable', 'to', 'factor'].edge_attr = edge_attr

    return data
def pre_Dataset_factor(file_path=None,targets=None,filter="*",**params):
    from tqdm import tqdm
    g_types_list = ['erdos_renyi', 'small-world', 'barabasi_albert', "K_Regular", "static_power_law"]
    dataSet = list()
    if file_path is not None:
        for file in tqdm(sorted(glob(file_path + '/' + filter + ".graphml"))):
            try:
                nx_g = nx.read_graphml(file)
            except Exception:

                raise Exception("There are errors in {}".format(file))
            # g=igraph.Graph.Read_GraphML(file)
            # if g.vcount()<25:
            #     print("error node number:{}".format(g.vcount()))
            #     continue
            g = to_gnngraph_nx(nx_g, features=params["features"], targets=targets, SL=params["SL"])
            factor_data = convert_simple_graph_to_factor_graph(g, factor_dim=params["factor_dim"],
                                                               edge_dim=params["edge_dim"],
                                                               higher_order=params["higher_order"])
            dataSet.append(factor_data)
    else:
        if set(params["g_types"]).issubset(set(g_types_list)):

            for data_count in tqdm(range(params['data_num'])):
                for g_type in params["g_types"]:
                    g = gen_graph(g_type, num_min=20, num_max=40)
                    g = to_gnngraph(g, features=params["features"], targets=targets, SL=params["SL"])
                    factor_data = convert_simple_graph_to_factor_graph(g, factor_dim=params["factor_dim"],
                                                                       edge_dim=params["edge_dim"],
                                                                       higher_order=params["higher_order"])
                    dataSet.append(factor_data)
        else:
            raise ValueError("Error:{} is not subset of g_type_list:{}".format(params["g_types"], g_types_list))
    return dataSet
def draw_3d(G,ax,pos,color_value):

    x, y, z = zip(*pos.values())

    for n, m in G.edges():
        zline = (z[n], z[m])
        xline = (x[n], x[m])
        yline = (y[n], y[m])
        ax.plot3D(xline, yline, zline,'k',linewidth=1)
    return ax.scatter3D(x, y, z, c=color_value, marker='o',alpha=1, s=100)

def visualization_evalution(pred,graph,target):


    data = graph  #可随机选取
    out =sigmoid( pred)
    G=torch_geometric.utils.to_networkx(data,to_undirected=True)

    ###########################2D可视化展示（可选）###########################################################
    fig=plt.figure("2d visualization of predict")
    pos = nx.kamada_kawai_layout(G)

    ax=fig.add_subplot(211)
    ax.set(title="Prediction  using GAT")
    nodes = nx.draw_networkx_nodes(G,pos=pos, node_color=out)
    nx.draw_networkx_edges(G, pos=pos,width=1)
    fig.colorbar(nodes)

    ax = fig.add_subplot(212)
    ax.set(title="Labels of network key nodes ")
    nodes = nx.draw_networkx_nodes(G, pos=pos, node_color=target)
    nx.draw_networkx_edges(G, pos=pos, width=1)
    fig.colorbar(nodes)
    #######################################################################################################
    ###########################3D可视化展示（可选）###########################################################
    # fig = plt.figure("3d visualization of predict")
    # pos = nx.kamada_kawai_layout(G,dim=3)
    #
    # ax = fig.add_subplot(121,projection="3d")
    # ax.set(title="Prediction  using GAT")
    # ax0=draw_3d(G=G,pos=pos,ax=ax,color_value=out)
    # fig.colorbar(ax0,ax=ax)
    #
    # ax = fig.add_subplot(122,projection="3d")
    # ax.set(title="Labels of network key nodes ")
    # ax0 = draw_3d(G=G, pos=pos, ax=ax, color_value=target)
    # fig.colorbar(ax0, ax=ax)
    #######################################################################################################




    return 0
def pred_accuracy(pred, y, num_graph,data):  # torch tensor变量 可batch
    sum=0
    last_split=0
    for i in range(num_graph):
        batch_node = data[i].num_nodes
        c_num=math.ceil(batch_node*0.6)
        out_node = pred[last_split: last_split + batch_node]
        label_node = y[last_split: last_split + batch_node]
        out_rank = np.argsort(out_node, axis=0).reshape(-1)
        label_rank = np.argsort(label_node, axis=0).reshape(-1)
        correct = (out_rank[:c_num] == label_rank[:c_num])
        accuracy = int(correct.sum()) / correct.numel()
        sum += accuracy
        last_split = last_split + batch_node

    return sum/num_graph
def kendall_rank_coffecient(out, label, num_graph, data):
    sum = 0
    last_split = 0
    for i in range(num_graph):
        batch_node = data[i].num_nodes
        out_node = out[last_split: last_split + batch_node]
        label_node = label[last_split: last_split + batch_node]
      #  out_rank = np.argsort(out_node, axis=0).reshape(-1)
      #  label_rank = np.argsort(label_node, axis=0).reshape(-1)
        tau, p_value = stats.kendalltau(out_node, label_node)
        sum += tau
        last_split = last_split + batch_node

    return sum/num_graph
def set_correlation_coffecient(out,label):
    pred_set = set(np.where(out == 1)[0].tolist())
    label_set=set(np.where(label == 1)[0].tolist())
    intersection_set_len = len(pred_set & label_set)
    union_set_len = len(pred_set | label_set)
    # difference_set_len=len(set_1-set_2)

    correlation_coefficient = (intersection_set_len) / union_set_len
    return correlation_coefficient
def visualize_node_emb(fig_label,node_emb,node_mean,node_var,target):
    z = TSNE(n_components=1).fit_transform(node_emb)
    # mean=sigmoid(node_mean)
    # var=sigmoid(node_var+node_mean)-mean
    mean=node_mean
    var=node_var
    print("mean:{}".format(node_mean))
    print("var:{}".format(var))
    print("target:{}".format(target))

    fig=plt.figure("3D visualization of  "+fig_label)
   # ax=fig.add_subplot(111,projection="3d")
    # ax.scatter(z[:, 0], z[:, 1], mean.T,   c="k",marker='o', alpha=1, s=10)
    # ax.scatter(z[:, 0], z[:, 1], target.T, c="r", marker='o', alpha=1, s=10)
    # ax.scatter(z[:, 0], z[:, 1], mean.T,   c=mean,    marker='o', alpha=0.7, s=10*var)
    ax = fig.add_subplot(111)
    ax.scatter(z[:, 0], mean.T, c="k", marker='o', alpha=1, s=10)
    ax.scatter(z[:, 0], target.T, c="r", marker='o', alpha=1, s=10)
    ax.scatter(z[:, 0],  mean.T, c='b', marker='o', alpha=0.5, s=10+10 * var)

def save_checkpoint(path: str,model_name: str, model,net_param):
    """
    Saves a model checkpoint.
    """
    # Convert args to namespace for backwards compatibility

    model_state = {
        'net_param': net_param,
        'state_dict': model.state_dict(),
    }
    model_path=os.path.join(path,model_name)
    torch.save(model_state,model_path)


def load_diffusion_model(path: str, device, infer_para=None) :
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :return: The loaded model,loaded network parameters.
    """

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)

    loaded_net_param = state["net_param"]
    if loaded_net_param.get("diffusion_training_type"):
        if loaded_net_param["diffusion_training_type"] == "FastT2T":
            loaded_net_param["diffusion_training_type"]="Fast"
        elif loaded_net_param["diffusion_training_type"] == "T2T":
            loaded_net_param["diffusion_training_type"]="Vanilla"
    loaded_state_dict = state['state_dict']
    if infer_para is not None:
        loaded_net_param.update(infer_para)

    loaded_net_param["GNNEncoder_param"]["device"] = device

    model = diffusion_models(task_model=loaded_net_param["task_model"], net_param=loaded_net_param).to(
        device)
    model.load_state_dict(loaded_state_dict,strict=False)
    model = model.to(device)

    return model,loaded_net_param
def load_graph_model(path: str, device, infer_para=None) :
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :return: The loaded model,loaded network parameters.
    """

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)

    loaded_net_param = state["net_param"]
    loaded_state_dict = state['state_dict']
    if infer_para is not None:
        loaded_net_param.update(infer_para)


    model = graph_models(gnn_name=loaded_net_param["gnn_name"], net_param=loaded_net_param).to(
        device)
    model.load_state_dict(loaded_state_dict,strict=False)
    model = model.to(device)

    return model,loaded_net_param


def save_config(path:str,configs_name="configs.yaml",dataset_param=None,net_param=None,train_param=None,optimizer_param=None,loss_param=None):
    train_state = {
        'dataset':dataset_param,
        'train': train_param,
        'net': net_param,
        'optimizer': optimizer_param,
        'loss': loss_param,
    }
    if train_state["dataset"].get("factor_dim"):
        train_state["net"]["GNNEncoder_param"]["gconv_param"]["factor_dim"] = train_state["dataset"]["factor_dim"]
        train_state["net"]["GNNEncoder_param"]["gconv_param"]["edge_dim"] = train_state["dataset"]["edge_dim"]
    file_path=os.path.join(path,configs_name)


    if os.path.exists(file_path):

        with open(file_path, 'r') as f:
            saved_train_parameters=yaml.safe_load(f)

        if json.dumps(saved_train_parameters,sort_keys=True) ==json.dumps(train_state,sort_keys=True):
            trained_model_path = os.path.join(path, "hold_out/trained_model")
            #print(trained_model_path)
            if os.path.exists(trained_model_path):
                configs_name=configs_name.replace("config_", "")
                print("{} model has existed".format(configs_name))
                return False
            else:
                return True
        else:
            with open(file_path, "w") as f:
                yaml.dump(train_state, f)
            return True
    else:
        with open(file_path, "w") as f:
            yaml.dump(train_state, f)
        return True

if __name__=="__main__":
    from torch_geometric.loader import DataLoader
    dataset_param={

                   "features": "none",
                   "filter":"*",
                   "SL":False,
                    "data_num":100,
                    "g_types":['erdos_renyi','small-world','barabasi_albert',"static_power_law","K_Regular"]
                   }

    trainset=pre_DataSet(**dataset_param)
    train_loader = DataLoader(trainset, batch_size=8, shuffle=True, drop_last=False)
    for n, data in enumerate(train_loader):
        torch_graphs=data.to_data_list()
        for torch_graph in torch_graphs:
            predata = torch_graph.y[0]
            print(type(predata))
