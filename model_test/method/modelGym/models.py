import copy
import json
import random
from collections import deque
from operator import itemgetter
import numpy as np
import torch
import yaml
import itertools as it
from utils.utils import seed_torch

import os
import time
import networkx as nx
import math
from .model_utils.utils import load_model, to_gnngraph, extract_list_del
from ..NodeRein.noderein import NodeRein
from ..NodeRein.TreeBreak import TreeBreak
from typing import Callable, List, Dict, Tuple
import warnings
import copy
def graph_models(gnn_name,net_param):
    if gnn_name=="ND_GNN_vanilla":
        from .GNN_ND.vanilla_models import ND_GNN_vanilla
        return ND_GNN_vanilla(net_param=net_param)
    elif gnn_name=="robustenss_metircs_model":
        from .metrics_model.vanilla_models import robustenss_metircs_model
        return robustenss_metircs_model(net_param=net_param)
    elif gnn_name=="diffusion_CEP_model":
        from .metrics_model.vanilla_models import diffusion_CEP_model
        return diffusion_CEP_model(net_param=net_param)
    else:
        raise ValueError("the definition  don't exit\n"
                         "\tyou can define it before using it")

def diffusion_models(task_model,net_param):
    from .Diffusion_model.diffusion_gnn_encoder import T2TGNNEncoder, NDGNNEncoder
    from .Diffusion_model.diffusion_model import FVSModel
    if net_param["GNNEncoder_set"] == "T2TGNNEncoder":
        GNNEnconder = T2TGNNEncoder
    elif net_param["GNNEncoder_set"] == "NDGNNEncoder":
        GNNEnconder = NDGNNEncoder
    else:
        raise ValueError("the definition  don't exit\n"
                         "\tyou can define it before using it")

    if task_model=="FVSModel":
        return FVSModel(net_param=net_param,GNNEncoder=GNNEnconder)
    else:
        raise ValueError("the definition  don't exit\n"
                         "\tyou can define it before using it")

def load_diffusion_model(path: str,device,infer_para=None) :
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

    loaded_net_param["device"] = device
   # loaded_net_param["parallel_sampling"]=4
    model = diffusion_models(task_model=loaded_net_param["task_model"], net_param=loaded_net_param).to(
        device)
    model.load_state_dict(loaded_state_dict,strict=False)
    model = model.to(device)

    return model,loaded_net_param
def modelGym_evaluation(G,target,method_name,models_REIN,graph_name=None,infer_param=None,iszero=False,**kwargs):
    method_path=os.path.join("method/modelGym/trained_models",method_name)

    method_config_path=method_path+"/{}.yaml".format(method_name)
    with open(method_config_path,'r') as f:
        method_config_param=yaml.safe_load(f)

    method_model_path = method_path + "/{}".format(method_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   # device="cpu"
    dismatle_data_path = "result/DiffGDM/dismantle_dataset_synth_size_new"
    if os.path.exists(dismatle_data_path):
        decycle_dimantle_flag = True
        with open(dismatle_data_path+"/graphs_dismantle_nodelist_target_{}_{}.json".format(1,False), "r") as f:
            dismatle_nodelist = json.load(f)
        with open(dismatle_data_path+"/graphs_dismantle_time_target_{}_{}.json".format(1,False), "r") as f:
            dismantle_time=json.load(f)




    g = copy.deepcopy(G)
    num_nodes = nx.number_of_nodes(g)
    features=method_config_param["dataset"]["features"]
    input_g = copy.deepcopy(g)
    if infer_param is not None:
        if infer_param.get("Missing_link"):
            input_g, _ = inject_deletion_noise(input_g,
                                               deletion_rate=infer_param["Missing_link"],
                                               )
        elif infer_param.get("Spurious_link"):
            input_g, _ = inject_addition_noise(input_g,
                                               addition_rate=infer_param["Spurious_link"],
                                               )
    tg_g=to_gnngraph(input_g,graph_name,features).to(device)
    print("modelGym_evaluation")
    #tg_g = to_gnngraph(g, features)
    seed_torch(123)
    dismantle_method_name = method_name + "_D_{}_{}".format(infer_param["gradient_estimator_type"],infer_param["infer_direction"])
    print("modelGym_evaluation:", )
    if dismantle_method_name in dismatle_nodelist[graph_name].keys():
        remove_list=dismatle_nodelist[graph_name][dismantle_method_name]
        time_int=dismantle_time[graph_name][dismantle_method_name]
    else:
        raise ValueError("error: no method {}".format(dismantle_method_name))
    num_nodes = nx.number_of_nodes(g)
    stop_condition=math.ceil(num_nodes * target * 0.01)
    remove_list=NodeRein(G, remove_list, target=stop_condition)
    # if "DiffGDM" in method_name:
    #
    #
    #
    #     model, _ = load_diffusion_model(path=method_model_path, device=device, infer_para=infer_param)
    #     model.eval()
    #     time_start = time.time()
    #
    #     keynodes_probs_list, _ = model.heatmap_pred_step(batch=tg_g, device=device)
    #     all_sample = len(keynodes_probs_list)
    #     keynodes_predictions = np.zeros(num_nodes)
    #     for keynodes_probs in keynodes_probs_list:
    #         keynodes_predictions = keynodes_predictions + keynodes_probs
    #
    #     pred = keynodes_predictions / all_sample
    #
    # else:
    #     model, _ = load_model(path=method_model_path, model=graph_models, device=device)
    #     model.eval()
    #     time_start = time.time()
    #
    #     pred = model(tg_g).squeeze().cpu().detach().numpy()
    # time_int = time.time() - time_start
    # torch.cuda.empty_cache()
    # #下几行将预测值与网络节点序号做映射，因此无需考虑网络节点序号问题
    # predictions = list(zip(list(g.nodes), pred))
    # sorted_prediction = sorted(predictions, key=itemgetter(1), reverse=True)
    # removal_order = list(map(itemgetter(0), sorted_prediction))
    # removal_order=list(map(int, removal_order))
    #
    # num_nodes = nx.number_of_nodes(g)
    # lcc_size = len(max(nx.connected_components(g), key=len))
    # stop_condition=math.ceil(num_nodes * target * 0.01)
    # remove_list = []
    #
    # if stop_condition!=0:
    #
    #     while lcc_size >= stop_condition:
    #         remove_counts = math.ceil(lcc_size * 0.01)
    #        # remove_counts=1
    #         remove_nodes, removal_order = extract_list_del(removal_order, end=remove_counts)
    #         g.remove_nodes_from(remove_nodes)
    #         remove_list += remove_nodes
    #         if g.number_of_edges()==0:
    #             lcc_size=0
    #         else:
    #             lcc_size = len(max(nx.connected_components(g), key=len))
    #     if models_REIN:
    #         remove_list=NodeRein(G,remove_list,target=stop_condition)
    # else:
    #     print("stop condition is equal to 0")
    #     print("record all nodes sequence")
    #     remove_list=removal_order


    return remove_list, time_int
def CoreDiffGDM_Emsemble(G,decode_target,method_name,graph_name=None,models_REIN=True,iszero=False,infer_param=None,**kwargs):
    method_name = method_name.replace("_Emsemble", "")

    g = G.copy()

    num_nodes = nx.number_of_nodes(g)
    decycle_dimantle_flag = False
    # dismatle_data_path = "result/DiffGDM/dismantle_dataset_real_all"
    # if os.path.exists(dismatle_data_path):
    #     decycle_dimantle_flag = True
    #     with open(dismatle_data_path+"/graphs_dismantle_nodelist_target_{}_{}.json".format(decode_target,models_REIN), "r") as f:
    #         dismatle_nodelist = json.load(f)
    #     with open(dismatle_data_path+"/graphs_dismantle_time_target_{}_{}.json".format(decode_target,models_REIN), "r") as f:
    #         dismantle_time=json.load(f)

    infer_directions=["MVC","LCC","GCC","ACC","ge","Density","NC"]

    final_remove_list=[]
    final_time_all=0
    best_remove_scores=num_nodes
    #此处可做并行化处理，加快推理速度
    for infer_direction in infer_directions:
        infer_param.update({"infer_direction":infer_direction})

        seed_torch(123)
        # if infer_param.get("rewrite"):
        #     decycle_method_name = method_name + "_{}".format(
        #                                                         infer_param["infer_direction"])
        # else:
        #
        #     decycle_method_name = method_name+"_base"
        # print("decycle_method_name:",decycle_method_name)
        # if decycle_dimantle_flag and decycle_method_name in dismatle_nodelist[graph_name].keys():
        #     remove_list=dismatle_nodelist[graph_name][decycle_method_name]
        #     time_all=dismantle_time[graph_name][decycle_method_name]
        #     if iszero:
        #         for i in range(len(remove_list)):
        #             if remove_list[i] == 0:
        #                 remove_list[i] = num_nodes
        # else:
        method_name=method_name.replace("Core","")
        method_path = os.path.join("method/modelGym/trained_models", method_name)
        method_config_path = method_path + "/{}.yaml".format(method_name)
        with open(method_config_path, 'r') as f:
            method_config_param = yaml.safe_load(f)
        method_model_path = method_path + "/{}".format(method_name)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device = "cpu"
        model,_=load_diffusion_model(path=method_model_path,device=device,infer_para=infer_param)
        model.eval()
        features=method_config_param["dataset"]["features"]
        #features ="none"
        tg_g = to_gnngraph(g,graph_name,features).to(device)
        time_start = time.time()
        g_stacked_predict_labels = model.decode_step(batch=tg_g, device=device)  #
        time_all = time.time() - time_start
        solved_solutions = [fvs_decode_np_fast(predict_labels, tg_g) for predict_labels in
                            g_stacked_predict_labels]  #

        solved_costs = [solved_solution.sum() for solved_solution in solved_solutions]
        # best_solved_cost = np.min(solved_costs)
        best_solved_id = np.argmin(solved_costs)
        solved_solutions = solved_solutions[best_solved_id]

        torch.cuda.empty_cache()
        solutions_map_dict = dict(zip(list(g.nodes), solved_solutions))  # 将节点实际编码序号与删除节点顺序相对应
        remove_dict = dict(filter(lambda x: x[1] == 1, solutions_map_dict.items()))
        remove_list = list(remove_dict.keys())

        num_nodes = nx.number_of_nodes(G)
        stop_condition = math.ceil(num_nodes * decode_target * 0.01)
        remove_list,_=break_rein(g,remove_list,stop_condition,models_REIN)


        if len(remove_list)<best_remove_scores:
            best_remove_scores=len(remove_list)
            final_remove_list=remove_list

        final_time_all+=time_all
    return final_remove_list, final_time_all
def modelGym_decycle2dismantle_evaluation(G,decode_target,method_name,graph_name=None,models_REIN=True,iszero=False,infer_param=None):

    g = G.copy()
    num_nodes = nx.number_of_nodes(g)
    # decycling_data_path = "result/DiffGDM/decycle_dataset_synth_size"
    # if os.path.exists(decycling_data_path):
    #     with open(decycling_data_path+"/graphs_decycle_nodelist.json", "r") as f:
    #         decycle_nodelist = json.load(f)
    #     with open(decycling_data_path+"/graphs_decycle_time.json", "r") as f:
    #         decycle_time=json.load(f)
    # else:
    #     decycle_nodelist={}
    #     decycle_time={}
    decycle_dimantle_flag=False
    if method_name.startswith("Core"):
        method_name=method_name.replace("Core","")
        decycle_dimantle_flag=True
    # if infer_param.get("rewrite"):
    #     decycle_method_name=method_name+"_{}_{}".format(infer_param["gradient_estimator_type"],infer_param["infer_direction"])
    # else:
    #     decycle_method_name=method_name

    seed_torch(123)
    # if decycle_dimantle_flag and method_name in decycle_nodelist[graph_name].keys()  :
    #     print("decycle_method_name:", decycle_method_name)
    #     remove_list=decycle_nodelist[graph_name][method_name]
    #     decode_time=decycle_time[graph_name][method_name]
    #     if iszero:
    #         for i in range(len(remove_list)):
    #             if remove_list[i] == 0:
    #                 remove_list[i] = num_nodes
    # else:

    method_path = os.path.join("method/modelGym/trained_models", method_name)
    method_config_path = method_path + "/{}.yaml".format(method_name)
    with open(method_config_path, 'r') as f:
        method_config_param = yaml.safe_load(f)
    method_model_path = method_path + "/{}".format(method_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = "cpu"
    model,_=load_diffusion_model(path=method_model_path,device=device,infer_para=infer_param)
    model.eval()
    features=method_config_param["dataset"]["features"]
    #features ="none"
    input_g=copy.deepcopy(g)
    if infer_param.get("Missing_link"):
        input_g,_=inject_deletion_noise(input_g,
                      deletion_rate=infer_param["Missing_link"],
                      )
    elif infer_param.get("Spurious_link"):
        input_g, _ = inject_addition_noise(input_g,
                                           addition_rate=infer_param["Spurious_link"],
                                           )
    #print(input_g.nodes)
    tg_g = to_gnngraph(input_g,graph_name,features).to(device)
    time_start = time.time()
    g_stacked_predict_labels=model.decode_step(batch=tg_g,device=device)#
    decode_time = time.time() - time_start
    solved_solutions = [fvs_decode_np_fast(predict_labels, tg_g) for predict_labels in
                        g_stacked_predict_labels]  #

    solved_costs = [solved_solution.sum() for solved_solution in solved_solutions]

    # best_solved_cost = np.min(solved_costs)
    best_solved_id = np.argmin(solved_costs)
    solved_solutions = solved_solutions[best_solved_id]

    torch.cuda.empty_cache()
    solutions_map_dict = dict(zip(list(g.nodes), solved_solutions))  # 将节点实际编码序号与删除节点顺序相对应
    remove_dict = dict(filter(lambda x: x[1] == 1, solutions_map_dict.items()))
    remove_list = list(remove_dict.keys())

    seed_torch(123)
    if decode_target=="decycle":
        remove_list=remove_list
    else:

        num_nodes = nx.number_of_nodes(G)
        stop_condition = math.ceil(num_nodes * decode_target * 0.01)
        remove_list,_=break_rein(g,remove_list,stop_condition,models_REIN)
    torch.cuda.empty_cache()

    return remove_list, decode_time

def fvs_decode_np_fast(preds, graph_data):
    """
    高效版自顶向下 FVS 解码：保持原算法逻辑，速度提升数十倍。
    """
    # 1. 获取基础数据 (适配 PyG 格式)
    edge_index = graph_data.edge_index.cpu().numpy()
    num_nodes = graph_data.num_nodes

    # 建立邻接表 (使用 list of lists，比 NetworkX 快得多)
    adj = [[] for _ in range(num_nodes)]
    degrees = np.zeros(num_nodes, dtype=int)
    for u, v in edge_index.T:
        adj[u].append(v)
        degrees[u] += 1

    # 2. 初始 2-Core 提取 (线性剪枝)
    # in_core 标记节点是否在当前的 2-Core 中
    in_core = np.ones(num_nodes, dtype=bool)

    def prune_graph(core_mask, current_degrees):
        """将所有度数 < 2 的节点从 core 中剔除"""
        queue = deque([i for i in range(num_nodes) if core_mask[i] and current_degrees[i] < 2])
        while queue:
            node = queue.popleft()
            if not core_mask[node]: continue
            core_mask[node] = False
            for neighbor in adj[node]:
                if core_mask[neighbor]:
                    current_degrees[neighbor] -= 1
                    if current_degrees[neighbor] == 1:
                        queue.append(neighbor)

    # 初始修剪
    prune_graph(in_core, degrees)

    # 3. 排序
    # 这里的逻辑与原代码一致：分数越高越优先被删除
    removal_order = np.argsort(preds)[::-1]
    order_ptr = 0  # 用于遍历排序后的节点

    solution = np.zeros(num_nodes, dtype=int)
    core_nodes_indices = np.where(in_core)[0]
    core_count = len(core_nodes_indices)

    # 4. 迭代删除逻辑 (1% 步长)
    while core_count > 0:
        # 计算这一轮需要从 core 中删除多少个预测最高分的节点
        num_to_remove = math.ceil(core_count * 0.01)
        removed_this_round = 0

        while removed_this_round < num_to_remove and order_ptr < num_nodes:
            node = removal_order[order_ptr]
            order_ptr += 1

            # 只有当该节点还在 2-Core 中时，才执行删除
            if in_core[node]:
                solution[node] = 1
                in_core[node] = False
                # 节点被删除，更新邻居度数
                for neighbor in adj[node]:
                    if in_core[neighbor]:
                        degrees[neighbor] -= 1
                removed_this_round += 1

        # 批量删除后，立即进行剪枝，缩小 2-Core 范围
        # 只需对度数可能受影响的节点进行修剪
        queue = deque([i for i in range(num_nodes) if in_core[i] and degrees[i] < 2])
        while queue:
            node = queue.popleft()
            if not in_core[node]: continue
            in_core[node] = False
            for neighbor in adj[node]:
                if in_core[neighbor]:
                    degrees[neighbor] -= 1
                    if degrees[neighbor] == 1:
                        queue.append(neighbor)

        core_count = np.sum(in_core)

    return solution.astype(int)
def break_rein(G,decycle_list,target,rein):
    '''

    :param G: nx.graph before remove nodes in decycle_list
    :param decycle_list:
    :param target: int,max lcc size
    :return: final remove list,sol_treebraek
    '''
    sol_treebraek = TreeBreak(G, decycle_list, target)
    if rein:
        final_remove_list=NodeRein(G,decycle_list+sol_treebraek,target)
        return final_remove_list,sol_treebraek
    else:
        return decycle_list+sol_treebraek,sol_treebraek
def modelGym_keynodes_predict(G,method_name,infer_param=None,graph_name=None):
    method_path = os.path.join("method/modelGym/trained_models", method_name)
    target_radio=100
    method_config_path = method_path + "/{}.yaml".format(method_name)
    with open(method_config_path, 'r') as f:
        method_config_param = yaml.safe_load(f)

    method_model_path = method_path + "/{}".format(method_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = "cpu"
    g = copy.deepcopy(G)
    num_nodes = nx.number_of_nodes(g)
    features = method_config_param["dataset"]["features"]
    tg_g = to_gnngraph(g,graph_name,features).to(device)
    seed_torch(123)
    if "DiffGDM" in method_name:

        model, _ = load_diffusion_model(path=method_model_path, device=device, infer_para=infer_param)
        model.eval()
        time_start = time.time()
        decyle_keynodes_probs_list,target_keynodes_probs_list, _ = model.heatmap_pred_step(batch=tg_g, device=device)

        all_sample=len(decyle_keynodes_probs_list)
        keynodes_predictions = np.zeros(num_nodes)
        for keynodes_probs in decyle_keynodes_probs_list:
            keynodes_predictions = keynodes_predictions + keynodes_probs
        if target_keynodes_probs_list is not None:
            target_keynodes_predictions = np.zeros(num_nodes)
            for keynodes_probs in target_keynodes_probs_list:
                target_keynodes_predictions = target_keynodes_predictions + keynodes_probs
            keynodes_predictions=(keynodes_predictions+target_radio*target_keynodes_predictions)/(1+target_radio)

        average_predicts = keynodes_predictions / all_sample
        predictions =[ dict(zip(list(g.nodes), average_predicts))]
        # predictions=[]
        # for keynodes_probs in keynodes_probs_list:
        #     predictions.append(dict(zip(list(g.nodes), keynodes_probs.tolist())))
        time_int = time.time() - time_start
        return predictions, time_int
    else:
        model, _ = load_model(path=method_model_path, model=graph_models, device=device)
        model.eval()
        time_start = time.time()

        x, edge_index = tg_g.x, tg_g.edge_index

        pred = model(tg_g).squeeze().cpu().detach().numpy()
        pred=pred.astype(float)
        # 下几行将预测值与网络节点序号做映射，因此无需考虑网络节点序号问题
        predictions =[ dict(zip(list(g.nodes), pred))]
        time_int = time.time() - time_start
        return predictions, time_int


def Random_dismantle(G,target,method_name):



    g = G.copy()
    num_nodes = nx.number_of_nodes(g)

    time_start = time.time()
    pred=np.random.rand(num_nodes)
    predictions = list(zip(list(g.nodes), pred))
    sorted_prediction = sorted(predictions, key=itemgetter(1), reverse=True)
    removal_order = list(map(itemgetter(0), sorted_prediction))
    removal_order=list(map(int, removal_order))

    num_nodes = nx.number_of_nodes(g)
    lcc_size = len(max(nx.connected_components(g), key=len))
    stop_condition=math.ceil(num_nodes * target * 0.01)
    remove_list = []
    while lcc_size >= stop_condition:
        remove_counts = math.ceil(lcc_size * 0.01)
       # remove_counts=1
        remove_nodes, removal_order = extract_list_del(removal_order, end=remove_counts)
        g.remove_nodes_from(remove_nodes)
        remove_list += remove_nodes
        connected_component = list(nx.connected_components(g))
        if connected_component:
            lcc_size = len(max(connected_component, key=len))
        else:
            lcc_size=0
    if method_name=="random_rein":
        remove_list=NodeRein(G,remove_list,target=stop_condition)
    time_int = time.time() - time_start
    return remove_list, time_int
def model_rank(G,method_name):
    method_path=os.path.join("method/modelGym/trained_models",method_name)

    method_config_path=method_path+"/{}.yaml".format(method_name)
    with open(method_config_path,'r') as f:
        method_config_param=yaml.safe_load(f)

    method_model_path = method_path + "/{}".format(method_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device="cpu"
    model,_=load_model(path=method_model_path,model=graph_models,device=device)

    model.eval()
    g = G.copy()
    features=method_config_param["dataset"]["dataset_features"]
    tg_g = to_gnngraph(g,features).to(device)
    #tg_g = to_gnngraph(g, features)


    pred = model(tg_g).squeeze().tolist()

    # predictions = list(zip(list(g.nodes), pred))
    # sorted_prediction = sorted(predictions, key=itemgetter(1), reverse=True)
    # removal_order = list(map(itemgetter(0), sorted_prediction))
    # removal_order=list(map(int, removal_order))

    return pred


def inject_deletion_noise(G: nx.Graph,
                          deletion_rate: float,
                          seed: int = 42,
                          ensure_connected: bool = False) -> Tuple[nx.Graph, float]:
    """
    注入第一类结构噪声：随机缺失真实边（Edge Deletion）

    参数:
    ----------
    G : nx.Graph
        输入的无向图
    deletion_rate : float
        边删除比例，范围[0, 1]。表示将被删除的边数占总边数的比例。
    seed : int
        随机种子，确保结果可重复
    ensure_connected : bool
        如果为True，在删除边后检查并保持图的连通性（可能删除少于预期的边）

    返回:
    ----------
    G_noisy : nx.Graph
        注入删除噪声后的图
    actual_deletion_rate : float
        实际删除的边比例
    """
    if deletion_rate < 0 or deletion_rate > 1:
        raise ValueError("删除比例必须在[0,1]范围内")

    np.random.seed(seed)
    G_noisy = G.copy()

    if G_noisy.number_of_edges() == 0:
        warnings.warn("输入图没有边，无法进行边删除")
        return G_noisy, 0.0

    # 获取所有边
    edges = list(G_noisy.edges())
    num_edges = len(edges)
    num_to_delete = int(num_edges * deletion_rate)

    if num_to_delete <= 0:
        return G_noisy, 0.0

    # 随机选择要删除的边
    indices_to_delete = np.random.choice(num_edges, size=num_to_delete, replace=False)

    # 记录原始连通性（如果需要保持）
    was_connected = nx.is_connected(G_noisy) if ensure_connected else False

    deleted_count = 0
    for idx in indices_to_delete:
        u, v = edges[idx]
        if G_noisy.has_edge(u, v):  # 确保边还存在
            G_noisy.remove_edge(u, v)
            deleted_count += 1

            # 如果要求保持连通性且图已不连通，则停止删除
            if ensure_connected and was_connected and not nx.is_connected(G_noisy):
                G_noisy.add_edge(u, v)  # 恢复这条边
                deleted_count -= 1
                break

    actual_deletion_rate = deleted_count / num_edges
    return G_noisy, actual_deletion_rate


# 可选的优化：使用更高效的方法采样不存在的边
def inject_addition_noise(G: nx.Graph,
                                    addition_rate: float,
                                    seed: int = 42,
                                    allow_self_loops: bool = False) -> Tuple[nx.Graph, float]:
    """
    高效版本：特别适用于大型稀疏图
    """
    if addition_rate < 0 or addition_rate > 1:
        raise ValueError("添加比例必须在[0,1]范围内")

    np.random.seed(seed)
    random.seed(seed)
    G_noisy = G.copy()
    nodes = list(G_noisy.nodes())
    n = len(nodes)

    if n < 2:
        warnings.warn("节点数太少，无法添加边")
        return G_noisy, 0.0

    # 计算边数
    max_possible_edges = n * (n - 1) // 2
    current_edges = G_noisy.number_of_edges()
    num_possible = max_possible_edges - current_edges

    if num_possible == 0:
        warnings.warn("图已是完全图，无法添加更多边")
        return G_noisy, 0.0

    # 计算需要添加的边数
    num_by_current = int(current_edges * addition_rate)
    num_by_possible = int(num_possible * addition_rate)
    num_to_add = min(num_by_current, num_by_possible)

    if num_to_add <= 0:
        return G_noisy, 0.0

    # 方法1：如果图比较稠密，使用基于索引的方法
    density = 2 * current_edges / (n * (n - 1)) if n > 1 else 0
    if density > 0.5:  # 图比较稠密
        # 为稀疏的缺失边生成索引
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        # 使用邻接矩阵加速查找（稀疏存储）
        adj_set = set()
        for u, v in G_noisy.edges():
            i, j = node_to_idx[u], node_to_idx[v]
            if i < j:
                adj_set.add(i * n + j)
            else:
                adj_set.add(j * n + i)

        # 采样不存在的边
        added_count = 0
        max_attempts = num_to_add * 10

        while added_count < num_to_add and max_attempts > 0:
            i, j = random.sample(range(n), 2)
            if i > j:
                i, j = j, i

            edge_hash = i * n + j
            if edge_hash not in adj_set:
                G_noisy.add_edge(nodes[i], nodes[j])
                adj_set.add(edge_hash)
                added_count += 1

            max_attempts -= 1
    else:
        # 方法2：如果图比较稀疏，使用简单的随机采样
        added_edges = set()
        added_count = 0
        attempts = 0
        max_attempts = num_to_add * 100

        while added_count < num_to_add and attempts < max_attempts:
            u = random.choice(nodes)
            v = random.choice(nodes)

            if u == v and not allow_self_loops:
                attempts += 1
                continue

            if u > v:
                u, v = v, u

            edge_key = (u, v)

            if not G_noisy.has_edge(u, v) and edge_key not in added_edges:
                G_noisy.add_edge(u, v)
                added_edges.add(edge_key)
                added_count += 1

            attempts += 1

    actual_addition_rate = added_count / num_possible
    return G_noisy, actual_addition_rate


if __name__=="__main__":
   pass