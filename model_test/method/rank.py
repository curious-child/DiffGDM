"""
输入为图和阈值，返回删除的节点及顺序
"""
import copy
import os
import random
import time
from ctypes import *
from os import remove

import torch
import math
import networkx as nx
import scipy.sparse.linalg as lin
import scipy.sparse as sp
import win32api
from matplotlib import pyplot as plt
import numpy as np
from sympy import rem

from method.NodeRein.TreeBreak import TreeBreak
from method.NodeRein.noderein import NodeRein
from utils.utils import extract_list_del
import pandas as pd
from method.COREHD.CHD import corehd
from method.COREHD.chouffe import destroy
from method.MinSum.minsum import minsum
from method.DQN.model import single_eva
from method.modelGym.models import modelGym_evaluation,modelGym_decycle2dismantle_evaluation
from typing import Callable, List, Dict, Tuple
import warnings
import copy
class Network_env():
    def __init__(self,graph,Target,method=0):
        self.inputmethod=method    #主要是方便画图时添加标签
        self.graph=graph
        self.method=0
        self.remove_list=[]
        self.terminal=False
        self.num_nodes=nx.number_of_nodes(self.graph)
        self.max_comp=math.ceil(0.01*self.num_nodes*Target)
        if method=='HDA':
            self.method=HDA
        if method=='HBA':
            self.method=betweenness
        if method=='CLA':
            self.method=closeness
        if method=='EIGA':
            self.method=EIG

    def step(self,action):
        assert action in self.graph.nodes
        self.graph.remove_node(action)
        self.remove_list.append(action)
        if len(self.graph.nodes) == 0:
            self.terminal = True
        else:
            largest_components = max(nx.connected_components(self.graph), key=len)
            self.comp = len(largest_components)
            if self.comp <= self.max_comp:
                self.terminal = True
            else:
                self.terminal = False

    def pred(self):
        if self.terminal == False:

            pred=self.method(self.graph)
            action = max(pred, key=lambda x: pred[x])

            return action
        else:
            return 0

    def perco(self,node_list):
        key=0
        nx.set_node_attributes(self.graph, 0, 'state')
        nx.set_node_attributes(self.graph, dict(zip(node_list, np.ones(len(node_list)))), 'state')
        while 1:
            for i in self.graph.nodes:
                value = sum((self.graph.nodes.data('state')[nb] for nb in self.graph[i]))
                if value > self.graph.degree(i) - 2:
                    self.graph.nodes[i]['state'] = 1
            x = [b for a, b in self.graph.nodes.data('state')]
            if key==sum(x):
                break
            key = sum(x)
           # print(key)
        #print(key)
        for i in node_list:
            x[i]=2          #标记初始感染点
        #print(x)
        plt.title(self.inputmethod+"   Ratio of percolated nodes == "+str(key/self.num_nodes))
       # nx.draw(self.graph, with_labels=True, node_color=color(x))
        plt.show()
    def perco_pred(self,n):
        pred = self.method(self.graph)
    # return pred[0:n]          #如果方法返回的是列表
        d_order = sorted(pred.items(), key=lambda x: x[1], reverse=True)
        return [x for x, y in d_order[0:n]]


def iter_method(method, target,G):
    print("method start")
    net = Network_env(G.copy(), method=method,Target=target)
    time_start = time.time()
    while not net.terminal:
        net.step(net.pred())
    time_finl = time.time()
    time_int = time_finl - time_start
    print("已达到阈值，当前网络最大连通子图大小：", net.comp )
    return net.remove_list,time_int
  # 对图进行h_index排序
def onepass_method(method, target,G,REIN=False):
    g = G.copy()
    num_nodes = nx.number_of_nodes(g)
    lcc_size = len(max(nx.connected_components(g), key=len))
    stop_condition = math.ceil(num_nodes * target * 0.01)
    remove_list = []
    time_start = time.time()
    if method=="HD" or method=="HD_R" :
        pred=nx.degree_centrality(g)
    elif method=="HB" or method=="HB_R":
        pred=nx.betweenness_centrality(g)
    elif method=="HSubgraph" :
        pred=nx.subgraph_centrality(g)
    elif method=="HCluster" :
        pred=nx.clustering(g)
    elif method=="HClose" :
        pred=nx.closeness_centrality(g)
    elif method=="HEIG" :
        pred=nx.eigenvector_centrality(g)
    elif method=="HKCore" :
        pred=nx.core_number(g)

    removal_order = list(dict(sorted(pred.items(), key=lambda x:x[1], reverse=True)).keys())
    if stop_condition!=0:
        while lcc_size >= stop_condition:
            remove_counts = math.ceil(lcc_size * 0.01)
            # remove_counts=1
            remove_nodes, removal_order = extract_list_del(removal_order, end=remove_counts)
            g.remove_nodes_from(remove_nodes)
            remove_list += remove_nodes
            lcc_size = len(max(nx.connected_components(g), key=len))

        if REIN:
            remove_list = NodeRein(G, remove_list, target=stop_condition)
    else:
        remove_list = removal_order
    time_int = time.time() - time_start
    return remove_list,time_int

def hindex(g, iter=3):
    h = dict(g.degree)
    k={}
    for e in range(iter):
        for q in g.nodes:
            n = [h[a] for a in nx.neighbors(g, q)]
            n.sort()  # 排序算法 最耗时的部分
            score = 1 if n[-1] > 0 else 0
            small = n[-1]
            for i in n[-2::-1]:
                if i == small and i > score:
                    score += 1
                elif i > score:
                    score += 1
                    small = i
                else:
                    break
            k[q] = score
        h = k
    return k

def cirefine(A, G, t,rein):
    a = []
    orign_G=copy.deepcopy(G)
    num_nodes = nx.number_of_nodes(G)
    for i in A:
        assert i in G.nodes()
        G.remove_node(i)
        a.append(i)
        largest_components = max(nx.connected_components(G), key=len)
        if len(largest_components) <= math.ceil(t * 0.01 * num_nodes):
            break
   # print("remove size",len(a))
    if rein:
        remove_list = NodeRein(orign_G, a, target=math.ceil(t * 0.01 * num_nodes))
   # print("NodeRein",len(remove_list))
    return remove_list
def CI(G, target, rein=False,biggestComponentEndThreshold=0.01, ballRadius=2, CImethod=0, outputNumBatch=1, updateBatch=1,
       isPrintMinPointCausingMinComponent=True,infer_param=None):
    """
    :param G: nx.graph ,undirected
    :param target: break the network into target size
    :param biggestComponentEndThreshold: complex network collapses to the certain giant component ratio where the reinsert algorithm starts for cpp ci
    :param ballRadius:is the Radius parameter defined in the Collective Influence Algorithm
    :param CImethod:method of several algorithms in collective influence. The user can check the ReadMe
    :param outputNumBatch: output num of point each line in the final result file, set to be 1
    :param updateBatch:batch size of deleted points per updating Collective Influence value
    :param isPrintMinPointCausingMinComponent:whether output limited point leading to 0(new cpp ci)/0.01(traditional ci) of giant component ratio or all points
    :return:
    """
    input_g = copy.deepcopy(G)
    if infer_param.get("Missing_link"):
        input_g, _ = inject_deletion_noise(input_g,
                                           deletion_rate=infer_param["Missing_link"],
                                           )
    elif infer_param.get("Spurious_link"):
        input_g, _ = inject_addition_noise(input_g,
                                           addition_rate=infer_param["Spurious_link"],
                                           )
    nx.write_edgelist(input_g, 'method/CI/CITEMP.csv', delimiter=',', data=False)
    path = 'method/CI/CITEMP.csv'
    dll = CDLL("method\CI\ComplexCi.dll")

    # 0 means the new c++ implementation of Collective Influence (CI) algorithm without re-insert process at the end
    time_start = time.time()
    dll.submain(create_string_buffer(path.encode(encoding="utf-8")), c_int(ballRadius), c_int(CImethod),
                c_int(outputNumBatch), c_int(updateBatch), c_double(biggestComponentEndThreshold),
                c_bool(isPrintMinPointCausingMinComponent))
    time_finl = time.time()
    time_int = time_finl - time_start
    source_data = pd.read_csv(path + '_out', header=None)
    A = list(source_data[1])
    win32api.FreeLibrary(dll._handle)
    return cirefine(A, G, target,rein=rein), time_int


def BPD(G,target):
    #add_path = os.path.split(os.path.abspath(__file__))[0] + '\\'
    #os.add_dll_directory(add_path)
    dll = CDLL("method/BPD/BPD.dll")
    nx.write_edgelist(G, 'method/BPD/bpdtemp.txt', delimiter=' ', data=False)
    path='method/BPD/bpdtemp.txt'
    num_nodes = nx.number_of_nodes(G)
    num_edges = nx.number_of_edges(G)
    time_start = time.time()
    target=math.ceil(0.01*target*num_nodes)

    #传入的target都为整数，网络拆解后的最大连通子图大小不大于target
    dll.main(create_string_buffer(path.encode(encoding="utf-8")), c_int(num_nodes),c_int(num_edges),c_int(target))
    time_finl = time.time()
    time_int = time_finl - time_start
    # source_data = pd.read_csv(path + 'TA', header=None)
    #
    # remove_list = list(source_data[0])
    win32api.FreeLibrary(dll._handle)
#新重插方法
    source_data = pd.read_csv('method/BPD/bpdtemp.txtFVS', header=None)
    remove_list = list(source_data[1:][0])

    stop_condition = target
    # G_undirected=G.copy()
    # G_undirected.remove_nodes_from(remove_list)
    # if nx.is_tree(G_undirected):
    #     print("无环（树）")
    # else:
    #     print("有环")
    remove_list, _ = break_rein(G.copy(), remove_list, stop_condition, rein=True)


    return remove_list,time_int
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
def GND(G, target, remove_strategy=3, plot_size=1, sort_strategy=2,REIN=False):
    """

    :param G:
    :param target:
    :param remove_strategy: 1: weighted method: powerIterationB(); vertex_cover_2() -- remove the node with smaller degree first
							    3: unweighted method with one-degree in vertex cover： powerIteration; vertex_cover() -- remove the node with larger degree first
	:param sort_strategy int Sort_Strategy; // removing order
	                          // 0 origrn order
							 // 1: ascending order - better strategy for weighted case
							 // 2: descending order - better strategy for unweighted case
    :param plot_size: the removal size of each line in FILE_PLOT. E.g. PLOT_SIZE=2 means each line of FILE_PLOT is the result that remove two nodes from the network
    :return:
    """

    dll = CDLL("method/GND/GND.dll")
    nx.write_edgelist(G, 'method/GND/GNDtemp.txt', delimiter=' ', data=False)
    path = "method/GND/GNDtemp.txt"
    num_nodes = nx.number_of_nodes(G)
    num_edges=nx.number_of_edges(G)
    target_size = math.ceil(0.01 * target * num_nodes)

    if target_size < 3:
        target=math.ceil(3/(0.01*num_nodes))

    time_start = time.time()
    dll.main(create_string_buffer(path.encode(encoding="utf-8")), c_int(num_nodes), c_int(num_edges),c_int(remove_strategy),
             c_int(target), c_int(plot_size))
    if REIN:
        dll = CDLL("method/GND/Rein.dll")
        dll.main(create_string_buffer(path.encode(encoding="utf-8")), c_int(sort_strategy), c_int(target),c_int(num_nodes))
        source_data = pd.read_csv(path + '_rein', header=None)
    else:
        source_data = pd.read_csv(path + '_out', header=None)
    time_finl = time.time()
    time_int = time_finl - time_start
    win32api.FreeLibrary(dll._handle)
    return list(source_data[0]), time_int


def pero_threold(G):

    degree = nx.degree_histogram(G)  # 返回list描述度分布的频数，数组的索引为度的大小，数组的值为度的频数
    degree_freq = torch.tensor([z / float(sum(degree)) for z in degree]).view(1, -1)
    degree_num = torch.arange(0, len(degree)).view(-1, 1).float()
    k_aver = torch.mm(degree_freq, degree_num)
    k2_aver = torch.mm(degree_freq, degree_num * degree_num)
    return (k2_aver/k_aver).item()

def greedy(G,target):
    """

    :param G1: G: networkx Graph, undirected
    :param target: int 网络拆解的目标大小，target%
    :return: 删除的节点及顺序
    """
    G=G.copy()
    nodeseq=[]
    num_nodes = nx.number_of_nodes(G)
    time_start = time.time()
    while(1):
        node_score={}
        for i in G.nodes:
            gtemp=G.copy()
            gtemp.remove_node(i)
            node_score[i]=pero_threold(gtemp);
        nodebest=min(node_score,key =lambda k : node_score[k])
        G.remove_node(nodebest)
        nodeseq.append(nodebest)
        largest_components = max(nx.connected_components(G), key=len)
        if len(largest_components) <= math.ceil(0.01*target * num_nodes) :
            print("当前最大联通子图节点数 = ", len(largest_components), "\n", "已到达阈值", "阈值=", int(0.01 * num_nodes) + 1)
            break
    time_finl = time.time()
    time_int = time_finl - time_start
    return nodeseq,time_int

def hashi_eig_cen(G):
    S = nx.to_scipy_sparse_matrix(G, dtype='d')

    I = sp.identity(S.shape[0])
    I_minus_D = sp.lil_matrix(S.shape)

    for node, deg in enumerate(G.degree()):
        I_minus_D[node, node] = 1.0 - deg[1]
    crazy = sp.bmat([[None, I], [I_minus_D, S]])
    eigenvalue, eigenvector = lin.eigs(crazy, k=1, which="LR")

    largest = abs(eigenvector[:len(G.nodes()), 0].flatten().real)
    norm = np.sqrt(sum(x * x for x in largest))

    centrality = dict(zip(G, map(float, largest / norm)))
    return centrality;

def HDA(G):  #度排序，度越高越大
    return nx.degree_centrality(G)

def EIG(G):
    return nx.eigenvector_centrality_numpy(G)

def closeness(G):
    return nx.closeness_centrality(G)

def betweenness(G):
    return nx.betweenness_centrality(G)

def random_walk(G):
    return nx.current_flow_betweenness_centrality(G)

def coreness(G):
    return nx.core_number(G)

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
#
# def nodereinsert(G, remove_list):
#     remove_list=remove_list.copy()
#     H=G.copy()
#     H.remove_nodes_from(remove_list)
#     time_start = time.time()
#     nodeseq=[]
#     while len(remove_list):
#         node_score = {}
#         for i in remove_list:
#             g_temp=H.copy()
#             g_temp.add_node(i)
#             g_temp.add_edges_from([(i,n) for n in (set(nx.neighbors(G,i))-set(remove_list))])
#             node_score[i] = len( nx.node_connected_component(g_temp,i))
#         nodebest = min(node_score, key=lambda k: node_score[k])
#         H.add_node(nodebest)
#         H.add_edges_from([(nodebest,n) for n in (set(nx.neighbors(G,nodebest))-set(remove_list))])
#         remove_list.remove(nodebest)
#         nodeseq.append(nodebest)
#     nodeseq.reverse()
#     time_finl = time.time()
#     time_int = time_finl - time_start
#     return nodeseq, time_int
#
#
#
# def NodeBreakList(G,method,target,radius=2,treebreakersize=None,k=2,scm='HD',**kwargs):
#     """
#     :param G: networkx图
#     :param method: 使用的方法
#     :param target: 目标阈值，target%
#     :param radius: CI的半径
#     :param treebreakersize: MinSum在破坏树时的目标
#     :return: 移除节点顺序，所用时间(s)
#
#     """
#     modelgym_methods = ["elu_stable", "elu_stable_f", "elu_stable_skip", "leakyrelu_stable", "leakyrelu_stable_f",
#                         "leakyrelu_stable_skip", "GMD", "leakyrelu_GMD", "GMD_nf"]
#
#     G=copy.deepcopy(G)
#     num_nodes = nx.number_of_nodes(G)
#     if 0 in G.nodes:
#         nx.relabel_nodes(G, {0: num_nodes}, copy=False)
#         iszero = True
#     else:
#         iszero = False
#
#     treebreakersize=target/2.0 if treebreakersize==None else treebreakersize
#     if method =="CI" :
#         remove_list, time =CI(G=G,target=target,radius=radius)
#     elif method=='BPD':
#         remove_list,time=BPD(G=G,target=target)
#     elif method=='GND':
#         remove_list,time=GND(G=G,target=target)
#     elif method=="HDA" or method=='HBA' or method=='CLA' or method=='EIGA':
#         remove_list,time=iter_method(method=method,G=G,target=target)
#     elif method=="MinSum":
#         remove_list,time=minsum(G=G,target=target,treebreakersize=treebreakersize)
#     elif method=='Greedy':
#         remove_list, time = greedy(G=G, target=target)
#     elif method=='CoreHD':
#         remove_list, time = corehd(G=G,target=target,treebreakersize=treebreakersize,k=k,scm=scm)
#     elif method=='DQN':
#         remove_list,time=single_eva(G,target)
#     elif method in modelgym_methods:
#         remove_list, time = modelGym_evaluation(G, target,method)
#     else:
#          raise Exception("method has not been defined",method ,"method list=[CI,GND,BPD,HBA,HDA,CLA,EIGA,MinSum,Greedy,CoreHD]")
#
#     if iszero:
#         for i in range(len(remove_list)):
#             if remove_list[i]== num_nodes:
#                 remove_list[i]=0
#     return remove_list,time














