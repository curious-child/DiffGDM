import copy
import json
import random
from ctypes import CDLL, create_string_buffer, c_int
import igraph
import networkx as nx
import numpy as np
import os
import pandas as pd

from operator import itemgetter
import random
import torch
import networkx.algorithms.community as nx_comm
import powerlaw  # pip install powerlaw
from collections import Counter
def seed_torch(seed=1029):
    random.seed(seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)   # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True   # 选择确定性算法


def check_small_world_sigma(G):
    """
    检测网络是否具有小世界特性
    """
    # 基本拓扑参数

    # 注意：对于不连通图，需要只考虑最大连通分量
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(largest_cc).copy()

        print("⚠️ 图不连通，仅使用最大连通分量计算")
    else:
        G_lcc=G.copy()

    sigma=nx.algorithms.sigma(G_lcc)

    return sigma
def check_small_world_omega(G):
    """
    检测网络是否具有小世界特性
    """
    # 基本拓扑参数

    # 注意：对于不连通图，需要只考虑最大连通分量
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(largest_cc).copy()

        print("⚠️ 图不连通，仅使用最大连通分量计算")
    else:
        G_lcc=G.copy()

    sigma=nx.algorithms.omega(G_lcc)

    return sigma

def check_scale_free(G):
    """
    检测网络是否具有无标度特性
    """
    # 获取所有节点的度
    igraph_G=igraph.Graph.from_networkx(G)
    # 2. 获取所有节点的度序列
    degrees = igraph_G.degree()

    # 3. 使用内置函数拟合幂律分布
    fit_result = igraph.statistics.power_law_fit(degrees)

    return fit_result.alpha
def graph_properties_calculate(nx_g,targets):

    if targets == "LCC":
        return len(max(nx.connected_components(nx_g), key=len))/nx_g.number_of_nodes()
    elif targets == "global_CC" or targets=="GCC":
        return nx.transitivity(nx_g)
    elif targets == "average_CC" or targets=="ACC":
        return nx.average_clustering(nx_g)
    elif targets == "natural_connectivity" or targets=="NC":
        estrada_index = nx.estrada_index(nx_g)
        # 自然连通性
        n =nx_g.number_of_nodes()
        return np.log(estrada_index / n)
    elif targets=="global_efficiency" or targets=="GE":

        return nx.global_efficiency(nx_g)
    elif targets=="density" or targets=="Density":
        return nx.density(nx_g)
    elif targets=="Assortativity":
        return nx.degree_assortativity_coefficient(nx_g)
    elif targets=="modularity":
        return nx_comm.modularity(nx_g,nx_comm.label_propagation_communities(nx_g))
    elif targets=="SW_sigma":
        return check_small_world_sigma(nx_g)
    elif targets=="SW_omega":
        return check_small_world_omega(nx_g)
    elif targets=="SF_alpha":
        return check_scale_free(nx_g)
    else:
        raise ValueError("Error fitness_func_type:{}".format(targets))

def my_all_shortest_paths(G, source, target):
    pred = nx.predecessor(G, source)
    if target not in pred:
        raise nx.NetworkXNoPath(
            f"Target {target} cannot be reached" f"from given sources"
        )
    sources = {source}
    seen = {target}
    stack = [[target, 0]]
    top = 0
    while top >= 0:
        node, i = stack[top]
        if node in sources:
            yield [p for p, n in reversed(stack[: top + 1])]
        if len(pred[node]) > i:
            stack[top][1] = i + 1
            next = pred[node][i]
            if next in seen:
                continue
            else:
                seen.add(next)
            top += 1
            if top == len(stack):
                stack.append([next, 0])
            else:
                stack[top][:] = [next, 0]
        else:
            seen.discard(node)
            top -= 1


def getSmallestCycles(Mygraph):
    SmallestCycles = set()
    NodeGirth = dict()
    SmallestCyclesOfNodes = {}  #
    CycleRatio = {}
    DEF_IMPOSSLEN = Mygraph.number_of_nodes() + 1
    Coreness = nx.core_number(Mygraph)

    removeNodes = set()
    NodeList = list(Mygraph.nodes())
    for i in Mygraph.nodes():  #
        SmallestCyclesOfNodes[i] = set()
        CycleRatio[i] = 0
        if Mygraph.degree(i) <= 1 or Coreness[i] <= 1:
            NodeGirth[i] = 0
            removeNodes.add(i)
        else:
            NodeGirth[i] = DEF_IMPOSSLEN
    NodeList.sort()
    #setp 1
    curCyc = list()
    for ix in NodeList[:-2]:  #v1
        if NodeGirth[ix] == 0:
            continue
        curCyc.append(ix)
        for jx in NodeList[NodeList.index(ix) + 1 : -1]:  #v2
            if NodeGirth[jx] == 0:
                continue
            curCyc.append(jx)
            if Mygraph.has_edge(ix,jx):
                for kx in NodeList[NodeList.index(jx) + 1:]:      #v3
                    if NodeGirth[kx] == 0:
                        continue
                    if Mygraph.has_edge(kx,ix):
                        curCyc.append(kx)
                        if Mygraph.has_edge(kx,jx):
                            SmallestCycles.add(tuple(curCyc))
                            for i in curCyc:
                                NodeGirth[i] = 3
                        curCyc.pop()
            curCyc.pop()
        curCyc.pop()
    # setp 2
    ResiNodeList = []  # Residual Node List
    for nod in NodeList:
        if NodeGirth[nod] == DEF_IMPOSSLEN:
            ResiNodeList.append(nod)
    if len(ResiNodeList) == 0:
        return SmallestCycles
    else:
        visitedNodes = dict.fromkeys(ResiNodeList,set())
        for nod in ResiNodeList:
            if Coreness[nod] == 2 and NodeGirth[nod] < DEF_IMPOSSLEN:
                continue
            for nei in list(Mygraph.neighbors(nod)):
                if Coreness[nei] == 2 and NodeGirth[nei] < DEF_IMPOSSLEN:
                    continue
                if not nei in visitedNodes.keys() or not nod in visitedNodes[nei]:
                    visitedNodes[nod].add(nei)
                    if nei not in visitedNodes.keys():
                        visitedNodes[nei] = set([nod])
                    else:
                        visitedNodes[nei].add(nod)
                    if Coreness[nei] == 2 and NodeGirth[nei] < DEF_IMPOSSLEN:
                        continue
                    Mygraph.remove_edge(nod, nei)
                    if nx.has_path(Mygraph, nod, nei):
                        for path in my_all_shortest_paths(Mygraph, nod, nei):
                            lenPath = len(path)
                            path.sort()
                            SmallestCycles.add(tuple(path))
                            for i in path:
                                if NodeGirth[i] > lenPath:
                                    NodeGirth[i] = lenPath
                    Mygraph.add_edge(nod, nei)
    return SmallestCycles

def dismantle_statistics_data_cover(target,Rein,method_list,file1_path=None,file2_path=None,save_path=None):

    with open(file1_path + '/graphs_dismantle_size_target_{}_{}.json'.format(target, Rein), "r") as f:
        file1_dismantle_size = json.load(f)
    with open(file1_path + '/graphs_dismantle_time_target_{}_{}.json'.format(target, Rein), "r") as f:
        file1_dismantle_time = json.load(f)
    with open(file1_path + '/graphs_dismantle_nodelist_target_{}_{}.json'.format(target, Rein), "r") as f:
        file1_dismantle_nodelist = json.load(f)
    file1_graphs_name = list(file1_dismantle_size.keys())
    if os.path.exists(file2_path + '/graphs_dismantle_size_target_{}_{}.json'.format(target, Rein)):
        with open(file2_path + '/graphs_dismantle_size_target_{}_{}.json'.format(target, Rein), "r") as f:
            file2_dismantle_size = json.load(f)
        with open(file2_path + '/graphs_dismantle_nodelist_target_{}_{}.json'.format(target, Rein), "r") as f:
            file2_dismantle_nodelist = json.load(f)
        with open(file2_path + '/graphs_dismantle_time_target_{}_{}.json'.format(target, Rein), "r") as f:
            file2_dismantle_time = json.load(f)
    else:

        if not os.path.exists(file2_path):
            print("no file2!!!")
            os.mkdir(file2_path)
        file2_dismantle_size={}
        file2_dismantle_nodelist={}
        file2_dismantle_time={}
        for graph in file1_graphs_name:
            file2_dismantle_size[graph]={}
            file2_dismantle_nodelist[graph]={}
            file2_dismantle_time[graph]={}
    for method in method_list:
        for graph in file1_graphs_name:
            file2_dismantle_size[graph][method] = file1_dismantle_size[graph][method]
            file2_dismantle_time[graph][method] = file1_dismantle_time[graph][method]
            file2_dismantle_nodelist[graph][method] = file1_dismantle_nodelist[graph][method]

    if os.path.exists(save_path):
        print("records_path:{}文件夹目录已存在".format(save_path))

    else:
        os.mkdir(save_path)
    with open(save_path + '/graphs_dismantle_size_target_{}_{}.json'.format(target, Rein), 'w') as f:
        json.dump(file2_dismantle_size, f, indent=4, separators=(',', ':'))
    with open(save_path + '/graphs_dismantle_time_target_{}_{}.json'.format(target, Rein), 'w') as f:
        json.dump(file2_dismantle_time, f, indent=4, separators=(',', ':'))
    with open(save_path + '/graphs_dismantle_nodelist_target_{}_{}.json'.format(target, Rein), 'w') as f:
        json.dump(file2_dismantle_nodelist, f, indent=4, separators=(',', ':'))

#暂略
def DiffGDM_method_infer_param_update(method,DiffDGM_infer_param):
    updated_infer_param = {}
    if method=="DiffGDM_cf":
        updated_infer_param={}
    elif method=="DiffGDM_cv":
        updated_infer_param={}
    elif method=="DiffGDM_gf":
        updated_infer_param={}
    elif method=="DiffGDM_gv":
        updated_infer_param={}

    DiffDGM_infer_param.update(updated_infer_param)
    return DiffDGM_infer_param

def dismantle_statistics_data_merge(target,Rein,method_list,file1_path=None,file2_path=None,save_path=None):
    if os.path.exists(save_path):
        print("records_path:{}文件夹目录已存在".format(save_path))

    else:
        os.mkdir(save_path)
    with open(file1_path + '/graphs_dismantle_size_target_{}_{}.json'.format(target, Rein), "r") as f:
        file1_dismantle_size = json.load(f)
    with open(file1_path + '/graphs_dismantle_time_target_{}_{}.json'.format(target, Rein), "r") as f:
        file1_dismantle_time = json.load(f)
    with open(file1_path + '/graphs_dismantle_nodelist_target_{}_{}.json'.format(target, Rein), "r") as f:
        file1_dismantle_nodelist = json.load(f)
    if os.path.exists(file2_path + '/graphs_dismantle_size_target_{}_{}.json'.format(target, Rein)):
        with open(file2_path + '/graphs_dismantle_size_target_{}_{}.json'.format(target, Rein), "r") as f:
            file2_dismantle_size = json.load(f)
        with open(file2_path + '/graphs_dismantle_nodelist_target_{}_{}.json'.format(target, Rein), "r") as f:
            file2_dismantle_nodelist = json.load(f)
        with open(file2_path + '/graphs_dismantle_time_target_{}_{}.json'.format(target, Rein), "r") as f:
            file2_dismantle_time = json.load(f)
    else:
        file2_dismantle_size={}
        file2_dismantle_nodelist={}
        file2_dismantle_time={}

    file2_dismantle_size = copy.deepcopy(file2_dismantle_size)
    deep_merge(file2_dismantle_size, file1_dismantle_size)
    file2_dismantle_nodelist = copy.deepcopy(file2_dismantle_nodelist)
    deep_merge(file2_dismantle_nodelist, file1_dismantle_nodelist)
    file2_dismantle_time = copy.deepcopy(file2_dismantle_time)
    deep_merge(file2_dismantle_time, file1_dismantle_time)

    with open(save_path + '/graphs_dismantle_size_target_{}_{}.json'.format(target,Rein), 'w') as f:
        json.dump(file2_dismantle_size, f, indent=4, separators=(',', ':'))
    with open(save_path + '/graphs_dismantle_time_target_{}_{}.json'.format(target,Rein), 'w') as f:
        json.dump(file2_dismantle_time, f, indent=4, separators=(',', ':'))
    with open(save_path + '/graphs_dismantle_nodelist_target_{}_{}.json'.format(target,Rein), 'w') as f:
        json.dump(file2_dismantle_nodelist, f, indent=4, separators=(',', ':'))
def graph_properties_dismantle_data_merge(graph_property,method_list,file1_path=None,file2_path=None,save_path=None):
    if os.path.exists(save_path):
        print("records_path:{}文件夹目录已存在".format(save_path))

    else:
        os.mkdir(save_path)
    with open(file1_path + '/graphs_properties_{}_nodelist.json'.format(graph_property), 'r') as f:
        file1_graphs_properties_nodelist = json.load(f)
    with open(file1_path + '/graphs_properties_{}_AUC.json'.format(graph_property), 'r') as f:
        file1_graphs_properties_AUC = json.load(f)
    if os.path.exists(file2_path + '/graphs_properties_{}_nodelist.json'.format(graph_property)):
        with open(file2_path + '/graphs_properties_{}_nodelist.json'.format(graph_property), 'r') as f:
            file2_graphs_properties_nodelist = json.load(f)
        with open(file2_path + '/graphs_properties_{}_AUC.json'.format(graph_property), 'r') as f:
            file2_graphs_properties_AUC = json.load(f)
    else:
        file2_graphs_properties_AUC={}
        file2_graphs_properties_nodelist={}

    file2_graphs_properties_AUC = copy.deepcopy(file2_graphs_properties_AUC)
    deep_merge(file2_graphs_properties_AUC, file1_graphs_properties_AUC)
    file2_graphs_properties_nodelist = copy.deepcopy(file2_graphs_properties_nodelist)
    deep_merge(file2_graphs_properties_nodelist, file1_graphs_properties_nodelist)


    with open(save_path + '/graphs_properties_{}_AUC.json'.format(graph_property), 'w') as f:
        json.dump(file2_graphs_properties_AUC, f, indent=4, separators=(',', ':'))

    with open(save_path + '/graphs_properties_{}_nodelist.json'.format(graph_property), 'w') as f:
        json.dump(file2_graphs_properties_nodelist, f, indent=4, separators=(',', ':'))
def deep_merge(d1, d2):
    """
    递归合并两个字典，d2会合并到d1中
    """
    for key in d2:
        if key in d1 and isinstance(d1[key], dict) and isinstance(d2[key], dict):
            deep_merge(d1[key], d2[key])
        else:
            d1[key] = d2[key]
    return d1
def decycle_statistics_data_merge(file1_path=None,file2_path=None,save_path=None):
    if os.path.exists(save_path):
        print("records_path:{}文件夹目录已存在".format(save_path))

    else:
        os.mkdir(save_path)
    with open(file1_path + '/graphs_decycle_size.json', "r") as f:
        file1_dismantle_size = json.load(f)
    with open(file1_path + '/graphs_decycle_time.json', "r") as f:
        file1_dismantle_time = json.load(f)
    with open(file1_path + '/graphs_decycle_nodelist.json', "r") as f:
        file1_dismantle_nodelist = json.load(f)
    if os.path.exists(file2_path + '/graphs_decycle_size.json'):
        with open(file2_path + '/graphs_decycle_size.json', "r") as f:
            file2_dismantle_size = json.load(f)
        with open(file2_path + '/graphs_decycle_time.json', "r") as f:
            file2_dismantle_time = json.load(f)
        with open(file2_path + '/graphs_decycle_nodelist.json', "r") as f:
            file2_dismantle_nodelist = json.load(f)
    else:
        file2_dismantle_size={}
        file2_dismantle_nodelist={}
        file2_dismantle_time={}

    file2_dismantle_size = copy.deepcopy(file2_dismantle_size)
    deep_merge(file2_dismantle_size, file1_dismantle_size)
    file2_dismantle_nodelist = copy.deepcopy(file2_dismantle_nodelist)
    deep_merge(file2_dismantle_nodelist, file1_dismantle_nodelist)
    file2_dismantle_time = copy.deepcopy(file2_dismantle_time)
    deep_merge(file2_dismantle_time, file1_dismantle_time)
    with open(save_path + '/graphs_decycle_size.json', 'w') as f:
        json.dump(file2_dismantle_size, f, indent=4, separators=(',', ':'))
    with open(save_path + '/graphs_decycle_time.json', 'w') as f:
        json.dump(file2_dismantle_time, f, indent=4, separators=(',', ':'))
    with open(save_path + '/graphs_decycle_nodelist.json', 'w') as f:
        json.dump(file2_dismantle_nodelist, f, indent=4, separators=(',', ':'))

def dismantle_threshold_data_merge(Rein,method_list,file1_path,file2_path="result/dataset_real",save_path="result/dataset_real_all"):
    if os.path.exists(save_path):
        print("records_path:{}文件夹目录已存在".format(save_path))

    else:
        os.mkdir(save_path)
    with open(file1_path + '/_graphs_dismantle_size_{}.json'.format( Rein), "r") as f:
        file1_dismantle_size = json.load(f)
    with open(file1_path + '/_graphs_dismantle_nodelist_{}.json'.format(Rein), "r") as f:
        file1_dismantle_nodelist = json.load(f)
    with open(file2_path + '/graphs_dismantle_size_{}.json'.format( Rein), "r") as f:
        file2_dismantle_size = json.load(f)
    with open(file2_path + '/graphs_dismantle_nodelist_{}.json'.format( Rein), "r") as f:
        file2_dismantle_nodelist = json.load(f)




    file1_graphs_name=list(file1_dismantle_size.keys())
    file1_methods_name = list(list(file1_dismantle_size.values())[0].keys())
    file2_graphs_name=list(file2_dismantle_size.keys())
    file2_methods_name = list(list(file2_dismantle_size.values())[0].keys())
    add_graphs_name = set(file1_graphs_name) - set(file2_graphs_name)
    add_method_list=set(file1_methods_name)-set(file2_methods_name)

    # for method in method_list:
    #     print(method)
    #     if method == "Stable-AGDM":
    #         method2 = "stable_AGDM"
    #     elif method == "Stable-GDM":
    #         method2 = "stable_GDM"
    #     elif method == "Stable-AGDM_gat":
    #         method2 = "stable_AGDM_gat"
    #     elif method == "Stable-AGDM_ng":
    #         method2 = "stable_AGDM_ng"
    #     else:
    #         method2 = method
    #     file2_methods_name = list(list(file2_dismantle_size.values())[0].keys())
    #     print(file2_methods_name)
    #     for graph in file2_graphs_name:
    #         file2_dismantle_size[graph][method] = file2_dismantle_size[graph][method2]
    #         file2_dismantle_nodelist[graph][method] = file2_dismantle_nodelist[graph][method2]

    if add_graphs_name:
        for graph in add_graphs_name:
            file2_dismantle_size[graph]=file1_dismantle_size[graph]
            file2_dismantle_nodelist[graph]=file1_dismantle_nodelist[graph]
    else:
        if add_method_list:
            for method in add_method_list:
                if method == "stable_AGDM4":
                    method2 = "stable_AGDM"
                else:
                    method2=method
                for graph in file1_graphs_name:
                    file2_dismantle_size[graph][method2] = file1_dismantle_size[graph][method]
                    file2_dismantle_nodelist[graph][method2] = file1_dismantle_nodelist[graph][method]

        else:
           # print("无需更改，请检查输入地址")
            pass

    with open(save_path + '/graphs_dismantle_size_{}.json'.format(Rein), 'w') as f:
        json.dump(file2_dismantle_size, f, indent=4, separators=(',', ':'))
    with open(save_path + '/graphs_dismantle_nodelist_{}.json'.format(Rein), 'w') as f:
        json.dump(file2_dismantle_nodelist, f, indent=4, separators=(',', ':'))

def dismantle_explain_data_merge(target,Rein,file1_path,explain_dismantle_method="AGDMX_MVC",file2_path="result/dataset_real",save_path="result/dataset_real_all"):
    if os.path.exists(save_path):
        print("records_path:{}文件夹目录已存在".format(save_path))

    else:
        os.mkdir(save_path)
    with open(file1_path + '/graphs_dismantle_size_{}.json'.format(explain_dismantle_method), "r") as f:
        file1_dismantle_size = json.load(f)
    with open(file2_path + '/graphs_dismantle_size_target_{}_{}.json'.format( target,Rein), "r") as f:
        file2_dismantle_size = json.load(f)
    with open(file1_path + '/graphs_dismantle_nodelist_{}.json'.format(explain_dismantle_method),  "r") as f:
        file1_dismantle_nodelist = json.load(f)
    with open(file2_path + '/graphs_dismantle_nodelist_target_{}_{}.json'.format(target, Rein), "r") as f:
        file2_dismantle_nodelist = json.load(f)





    file1_graphs_name=list(file1_dismantle_size.keys())
    file2_graphs_name=list(file2_dismantle_size.keys())
    add_graphs_name=set(file1_graphs_name)-set(file2_graphs_name)
    file1_methods_name=list(list(file1_dismantle_size.values())[0].keys())
    file2_methods_name = list(list(file2_dismantle_size.values())[0].keys())
    add_method_list=set(file1_methods_name)-set(file2_methods_name)
    if add_graphs_name:
        for graph in add_graphs_name:
            file2_dismantle_size[graph]=file1_dismantle_size[graph]
            file2_dismantle_nodelist[graph]=file1_dismantle_nodelist[graph]
    else:
        if add_method_list:
            for method in add_method_list:
                if method == "stable_AGDM4":
                    method2 = "stable_AGDM"
                else:
                    method2=method
                for graph in file1_graphs_name:
                    file2_dismantle_size[graph][method2] = file1_dismantle_size[graph][method]
                    file2_dismantle_nodelist[graph][method2] = file1_dismantle_nodelist[graph][method]

        else:
           # print("无需更改，请检查输入地址")
            pass

    with open(save_path + '/graphs_dismantle_size_{}.json'.format(Rein), 'w') as f:
        json.dump(file2_dismantle_size, f, indent=4, separators=(',', ':'))
    with open(save_path + '/graphs_dismantle_nodelist_{}.json'.format(Rein), 'w') as f:
        json.dump(file2_dismantle_nodelist, f, indent=4, separators=(',', ':'))

def feature_rank(graph,feature):
    predictions = list(zip(list(graph.nodes), feature))
    sorted_prediction = sorted(predictions, key=itemgetter(1), reverse=True)
    removal_order = list(map(itemgetter(0), sorted_prediction))
    removal_order = list(map(int, removal_order))

    return removal_order
def extract_list_del(l1,end,start=0,):
    l2=l1[start:end]
    del l1[start:end]
    return l2,l1
def get_median(data):
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half]) / 2

def chi(o, e):
    if e == 0:
        return 0
    return (o - e) ** 2 / e
def feature_for_graph(graph,feature):
    values={}

    g=igraph.Graph.from_networkx(graph)
    vertices=g.vs
    degree = g.degree()
    max_degree = g.maxdegree()
    degree = np.divide(degree, max_degree)
    if feature=="Degree"or feature=="degree":
        out_features = degree
    elif feature=="Chi degree" or feature=="chi_degree":
        average_degree = np.mean(degree)
        out_features = [chi(v.degree() / max_degree, average_degree) for v in vertices]
    elif feature=="CC" or feature=="clustering_coefficient":
        out_features=g.transitivity_local_undirected(mode="zero")
    elif feature=="Kcore" or feature=="kcore":
        kcore = g.coreness()
        kcore_array = np.divide(kcore, np.max(kcore))
        out_features= kcore_array
    elif feature=="BC":
        bc = g.betweenness()
        bc_array = np.divide(bc, np.max(bc))
        out_features = bc_array
    elif feature=="PageRank":
        out_features=g.pagerank()
    elif feature=="Eigenvectors":
        out_features=g.eigenvector_centrality()
    else:
        raise ValueError("the definition  don't exit\n"
                         "\tyou can define it before using it")

    for node,out_feature_value in zip(graph.nodes,out_features):
        values[node]=out_feature_value
    return values
def features_for_graph(graph,features_list):
    features=["Degree","CC","Kcore","Chi degree","BC","PageRank","Eigenvectors",
              "degree","clustering_coefficient","kcore","chi_degree"]
    if not set(features_list).issubset(set(features)):
        raise ValueError("the definition  don't exit\n"
                         "\tyou can define it before using it")
    for feature in features_list:

        values=feature_for_graph(graph,feature)
        nx.set_node_attributes(graph,values=values,name=feature)
def GND_rein(path,num_nodes,target):
    dll = CDLL("method/GND/rein.dll")
    dll.reinmain(create_string_buffer(path.encode(encoding="utf-8")), c_int(num_nodes), c_int(target))
    return 0

def get_allfile(path):  # 获取所有文件
    all_file = []
    for f in os.listdir(path):  #listdir返回文件中所有目录
        f_name = os.path.join(path, f)
        all_file.append(f_name)
    return all_file
def gen_graph(g_type, num_min, num_max):#生成图
    max_n = num_max
    min_n = num_min
    cur_n = np.random.randint(max_n - min_n + 1) + min_n
    if g_type == 'erdos_renyi':
        g = nx.erdos_renyi_graph(n=cur_n, p=0.1)

    elif g_type == 'powerlaw':
        g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)

    elif g_type == 'small-world':
        g = nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)

    elif g_type == 'barabasi_albert':
        g = nx.barabasi_albert_graph(n=cur_n, m=4)

    nx.set_node_attributes(g, 0, 'state')
    return g

def read_files(path):
   _,type=os.path.splitext(path)
   if type=='.txt':
       G = nx.read_edgelist(path=path,nodetype=int)
   elif type=='.csv':
       source_data = pd.read_csv(path, header=None)
       G = nx.Graph(source_data.values.tolist())
   a=nx.number_of_nodes(G)
   print("nember of nodes:", a, "nember of edges: ", nx.number_of_edges(G))
   return G


def compute_auc_dismantling():
    pass



