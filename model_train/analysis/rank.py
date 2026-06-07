"""
输入为图和阈值，返回删除的节点及顺序
"""

import os
import time
from operator import itemgetter
import torch
import math
import networkx as nx
import yaml
import numpy as np
from torch_geometric.data import Data, HeteroData
import torch_geometric.utils as tg_utils
from analysis.NodeRein.noderein import NodeRein
from analysis.TreeBreak import TreeBreak
from utils.utils import load_diffusion_model




def nodereinsert(G, remove_list):
    remove_list=remove_list.copy()
    H=G.copy()
    H.remove_nodes_from(remove_list)
    time_start = time.time()
    nodeseq=[]
    while len(remove_list):
        node_score = {}
        for i in remove_list:
            g_temp=H.copy()
            g_temp.add_node(i)
            g_temp.add_edges_from([(i,n) for n in (set(nx.neighbors(G,i))-set(remove_list))])
            node_score[i] = len( nx.node_connected_component(g_temp,i))
        nodebest = min(node_score, key=lambda k: node_score[k])
        H.add_node(nodebest)
        H.add_edges_from([(nodebest,n) for n in (set(nx.neighbors(G,nodebest))-set(remove_list))])
        remove_list.remove(nodebest)
        nodeseq.append(nodebest)
    nodeseq.reverse()
    time_finl = time.time()
    time_int = time_finl - time_start
    return nodeseq, time_int

#基于最优解环策略的
def modelGym_diffusion_evaluation(graph_name,G,target,method_name,models_REIN=True,
                                  model_path="results/ND_pre_trainl/grid_search",infer_param=None):


    method_path=model_path+"/"+method_name

    method_config_path=method_path + "/{}.yaml".format(method_name)
    with open(method_config_path,'r') as f:
        method_config_param=yaml.safe_load(f)

    method_model_path = method_path+"/hold_out/trained_model/NDlearner_diffusion_trained"
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if method_config_param["net"].get("diffusion_training_type") is None:
        infer_param["diffusion_training_type"]="FastT2T"



    device="cuda"

    g = G.copy()
    num_nodes = g.number_of_nodes()
    G_edges_num = g.number_of_edges()
    print("G edges num:{}".format(G_edges_num))
    features=method_config_param["dataset"]["features"]

    tg_g = to_gnngraph(g, graph_name,features).to(device)
    model, _ = load_diffusion_model(path=method_model_path, device=device, infer_para=infer_param)
    model.eval()
    print("Start decoding")
    time_start = time.time()



    if target =="decycling":
        solved_solutions, _, _, _, _ = model.decode_step(batch=tg_g, device=device)
        solutions_map_dict = dict(zip(list(g.nodes), solved_solutions))  # 将节点实际编码序号与删除节点顺序相对应
        remove_dict = dict(filter(lambda x: x[1] == 1, solutions_map_dict.items()))
        remove_list = list(remove_dict.keys())
    else:

        keynodes_probs_list, _ = model.heatmap_pred_step(batch=tg_g, device=device)
        all_sample = len(keynodes_probs_list)
        keynodes_predictions = np.zeros(num_nodes)
        for keynodes_probs in keynodes_probs_list:
            keynodes_predictions = keynodes_predictions + keynodes_probs
        pred = keynodes_predictions / all_sample
        predictions = list(zip(list(g.nodes), pred))
        sorted_prediction = sorted(predictions, key=itemgetter(1), reverse=True)
        removal_order = list(map(itemgetter(0), sorted_prediction))
        removal_order = list(map(int, removal_order))

        lcc_size = len(max(nx.connected_components(g), key=len))
        stop_condition = math.ceil(num_nodes * target * 0.01)
        remove_list = []


        while lcc_size >= stop_condition:
            remove_counts = math.ceil(lcc_size * 0.01)
            # remove_counts=1
            remove_nodes, removal_order = extract_list_del(removal_order, end=remove_counts)
            g.remove_nodes_from(remove_nodes)
            remove_list += remove_nodes
            lcc_size = len(max(nx.connected_components(g), key=len))
        if models_REIN:
            remove_list = NodeRein(G, remove_list, target=stop_condition)
    torch.cuda.empty_cache()
    time_int = time.time() - time_start
    print("Time used for decoding: ", time_int)

    return remove_list, time_int

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
def extract_list_del(l1,end,start=0,):
    l2=l1[start:end]
    del l1[start:end]
    return l2,l1


def del_tensor_ele(arr, index):
    """
    arr: 输入tensor
    index: 需要删除位置的索引
    n: 从index开始，需要删除的行数
    """
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)



def to_gnngraph(g,graph_name,features):

    g = g.to_directed() if not nx.is_directed(g) else g
    if isinstance(g, (nx.MultiGraph, nx.MultiDiGraph)):
        edges = list(g.edges(keys=False))
    else:
        edges = list(g.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().view(2, -1)

    if "None" in features:

        x = np.ones((g.number_of_nodes(), 1))
    elif "one-hot" in features:
        x=np.identity(g.number_of_nodes())
    else:
        x = np.column_stack(
            tuple(
                list(nx.get_node_attributes(g,feature).values()) for feature in features
            )
        )

    x = torch.from_numpy(x).to(torch.float)

    return Data(x=x,edge_index=edge_index,graph_name=graph_name)
def bron_kerbosch(R, P, X, neighbors, cliques):
    if not P and not X:
        cliques.append(R)
    while P:
        v = P.pop()
        new_R = R | {v}
        new_P = P & neighbors[v]
        new_X = X & neighbors[v]
        bron_kerbosch(new_R, new_P, new_X, neighbors, cliques)
        X.add(v)


def find_maximal_cliques(data):
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    neighbors = {i: set() for i in range(num_nodes)}
    for u, v in edge_index.t().tolist():
        neighbors[u].add(v)
        neighbors[v].add(u)  # 无向图需要双向添加
    # 过滤孤立节点
    neighbors = {node: adj for node, adj in neighbors.items() if adj}
    # 初始化 P（候选节点集合）为非孤立节点集合
    P = set(neighbors.keys())
    cliques = []
    bron_kerbosch(set(), P , set(), neighbors, cliques)
    return cliques

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

def convert_simple_graph_to_factor_graph(orig_data,factor_dim,edge_dim,higher_order=None,edge_attr=None):
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
    data['variable'].y = None
    data['variable', 'to', 'factor'].src4v_edge_index = orig_data.edge_index
    data['variable', 'to', 'factor'].tgt4v_edge_index = torch.flip(orig_data.edge_index, dims=[0])




    # Add connection relationships
    # Add edges from variable nodes to factor nodes
    factor_edge = []
    nx_graph = tg_utils.to_networkx(orig_data, to_undirected=True)
    if higher_order is None:
        cliques_G = nx.find_cliques(nx_graph)
        cliques = {frozenset(c) for c in cliques_G if len(c) > 1}
    else:
        cliques = find_cliques_with_limit(orig_data, max_clique_size=(higher_order + 1))
    print("cliques num:", len(cliques))
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


