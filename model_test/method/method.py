import copy
import time

import pandas as pd

from method.NodeRein.noderein import NodeRein
from method.modelGym.models import Random_dismantle, modelGym_keynodes_predict, CoreDiffGDM_Emsemble
from method.rank import *
import networkx as nx
import re
from networkx import generate_edgelist
from method.DQN.model import single_eva
from method.NIRM.evaluation import NIRM_evaluation, NIRM_predict
from utils.utils import seed_torch
from .Domirank import domirank as dr
from operator import itemgetter
def nodereinsert(G, remove_list):
    remove_list=remove_list.copy()
    H=G.copy()
    H.remove_nodes_from(remove_list)
    nodeseq = []
    node_score = {}
    node_component={}
    for i in remove_list:
        g_temp = H.copy()
        g_temp.add_node(i)
        g_temp.add_edges_from([(i, n) for n in (set(nx.neighbors(G, i)) - set(remove_list))])
        node_component[i]=set(nx.node_connected_component(g_temp, i))
        node_score[i] = len(node_component[i])
    nodebest = min(node_score, key=lambda k: node_score[k])
    H.add_node(nodebest)
    H.add_edges_from([(nodebest, n) for n in (set(nx.neighbors(G, nodebest)) - set(remove_list))])
    node_score.pop(nodebest)
    nodeseq.append(nodebest)

    while len(node_score.keys()):
        remove_neighbors_best = set()
       # nodebest_componet=nx.node_connected_component(H,nodebest)
      #  for n in node_score.keys():
       #     if nodebest_componet&node_component[n]:
      #          remove_neighbors_best.add(n)
        remove_neighbors_best=remove_neighbors_best|(set(nx.neighbors(G,nodebest))&set(node_score.keys()) )
        for i in remove_neighbors_best:
            g_temp = H.copy()
            g_temp.add_node(i)
            g_temp.add_edges_from([(i, n) for n in (set(nx.neighbors(G, i)) - set(node_score.keys()))])
            node_component[i] = set(nx.node_connected_component(g_temp, i))
            node_score[i] = len(node_component[i])
        nodebest = min(node_score, key=lambda k: node_score[k])
        H.add_node(nodebest)
        H.add_edges_from([(nodebest,n) for n in (set(nx.neighbors(G,nodebest))-set(node_score.keys()))])

        node_score.pop(nodebest)
        nodeseq.append(nodebest)
    nodeseq.reverse()

    return nodeseq




def DecycleList(G, method,graph_name=None, **kwargs):
    modelgym_methods = [ "DiffGDM","CoreDiffGDM"
                        ]

    if "DiffGDM" in method:
        if kwargs.get("DiffDGM_infer_param") is not None:
            DiffDGM_infer_param = kwargs.get("DiffDGM_infer_param")

            DiffDGM_infer_param.update(DiffGDM_param_update(method))
            if DiffDGM_infer_param["rewrite"]:

                infer_direction = DiffDGM_infer_param["infer_direction"]

                if "_{}".format(infer_direction) in method:
                    method = method.replace("_{}".format(infer_direction), "")
            else:
                method = method.replace("_base", "")

    start_time = time.time()
    if method=="MinSum":
        nodelist, _ = NodeBreakList(G, method='MinSum', target=20)
        # path='method/MinSum/seeds.txt'
        fo = open("method/MinSum/seeds.txt", 'r')
        seeds_list = []
        num_nodes = nx.number_of_nodes(G)
        for line in fo:
            if 'S' == line[0]:
                number = re.findall("\d+", line)
                number = int(number.pop())
                seeds_list.append(number)
        if 0 in G.nodes:
            for i in range(len(seeds_list)):
                if seeds_list[i] == num_nodes:
                    seeds_list[i] = 0
    elif method=="BPD":
        num_nodes = nx.number_of_nodes(G)
        nodelist, _ = NodeBreakList(G, method='BPD', target=20)
        # path='method/MinSum/seeds.txt'
        source_data = pd.read_csv('method/BPD/bpdtemp.txtFVS', header=None)
        seeds_list = list(source_data[1:][0])
        if 0 in G.nodes:
            for i in range(len(seeds_list)):
                if seeds_list[i] == num_nodes:
                    seeds_list[i] = 0
    elif method=="CoreHD":
        seeds_list, _ = destroy(2,G,scm='HD')
    elif method in modelgym_methods:
        seeds_list, _ = modelGym_decycle2dismantle_evaluation(G, decode_target="decycle",graph_name=graph_name,
                                                              method_name=method, models_REIN=True,
                                                              infer_param=kwargs["DiffDGM_infer_param"] if kwargs.get("DiffDGM_infer_param") else None)

    else:
        raise Exception("method has not been defined", method   )
    infer_time = time.time() - start_time
    return seeds_list,infer_time
def Domirank_break(G,target,Rein=True,infer_param= None):
    time_start= time.time()
    nx_g=G.copy()
    num_nodes = nx.number_of_nodes(nx_g)


    lcc_size = len(max(nx.connected_components(nx_g), key=len))
    stop_condition = math.ceil(num_nodes * target * 0.01)
    input_g = copy.deepcopy(nx_g)
    if infer_param.get("Missing_link"):
        input_g, _ = inject_deletion_noise(input_g,
                                           deletion_rate=infer_param["Missing_link"],
                                           )
    elif infer_param.get("Spurious_link"):
        input_g, _ = inject_addition_noise(input_g,
                                           addition_rate=infer_param["Spurious_link"],
                                           )
    GAdj = nx.to_scipy_sparse_array(input_g)
    lambN = dr.find_eigenvalue(GAdj, maxIter=1000, dt=0.1, checkStep=100)
    sigma, sigmaArray = dr.optimal_sigma(GAdj, lambN,iterationNo = 10)
    _, ourDomiRankDistribution = dr.domirank(GAdj, sigma)
    node_centrality = ourDomiRankDistribution.tolist()
    predictions = list(zip(list(nx_g.nodes), node_centrality))
    sorted_prediction = sorted(predictions, key=itemgetter(1), reverse=True)
    removal_order = list(map(itemgetter(0), sorted_prediction))
    removal_order = list(map(int, removal_order))
    print("remove size",len(removal_order))
    remove_list=[]
    if stop_condition!=0:
        while lcc_size >= stop_condition:
            remove_counts = math.ceil(lcc_size * 0.1)
            # remove_counts=1
            remove_nodes, removal_order = extract_list_del(removal_order, end=remove_counts)
            nx_g.remove_nodes_from(remove_nodes)
            remove_list += remove_nodes
            lcc_size = len(max(nx.connected_components(nx_g), key=len))
           # print("lcc size",lcc_size)


    else:
        remove_list = removal_order
    if Rein:
        remove_list = NodeRein(G, remove_list, target=stop_condition)
    time_int = time.time() - time_start
    return remove_list,time_int
def DiffGDM_param_update(DiffGDM_name):
    if DiffGDM_name=="CoreDiffGDM" or DiffGDM_name=="DiffGDM":
        dict = {"rewrite": True,
                "infer_direction": "MVC"}
    elif DiffGDM_name=="DiffGDM_base" or DiffGDM_name=="CoreDiffGDM_base":
        dict = {"rewrite":False}
    elif "ens" not in DiffGDM_name:
        dict = {"rewrite":True,
                "infer_direction":DiffGDM_name.split("_")[-1]}
    elif DiffGDM_name=="CoreDiffGM_Emsemble":
        dict = {"rewrite":True,}
    return dict
def NodeBreakList(G, target,graph_name=None, **kwargs):
    """

    :param G: networkx图
    :param method: 使用的方法
    :param target: 目标阈值，target%
    :param ballRadius: CI的半径
    :param treebreakersize: MinSum在破坏树时的目标
    :return: 移除节点顺序，所用时间(s)
    """
    method = kwargs['method']
    treebreakersize = kwargs['treebreakersize'] if 'treebreakersize' in kwargs else target
    biggestComponentEndThreshold = kwargs['biggestComponentEndThreshold'] if 'biggestComponentEndThreshold' in kwargs else 0.01
    ballRadius = kwargs['ballRadius'] if 'ballRadius' in kwargs else 2
    updateBatch = kwargs['updateBatch'] if 'updateBatch' in kwargs else 1
    CoreHD_k = kwargs['CoreHD_k'] if 'CoreHD_k' in kwargs else 2
    CImethod = kwargs['CImethod'] if 'CImethod' in kwargs else 0
    scm = kwargs['CoreHD_scm'] if 'CoreHD_scm' in kwargs else 'HD'
    remove_strategy = kwargs['remove_strategy'] if 'remove_strategy' in kwargs else 3
    sort_strategy = kwargs['sort_strategy'] if 'sort_strategy' in kwargs else 2
    GND_REIN = kwargs['GND_REIN'] if 'GND_REIN' in kwargs else False
    CoreHD_REIN=kwargs['CoreHD_REIN'] if 'CoreHD_REIN' in kwargs else True
    MinSum_REIN = kwargs['MinSum_REIN'] if 'MinSum_REIN' in kwargs else True
    models_REIN = kwargs['models_REIN'] if 'models_REIN' in kwargs else True
    modelgym_methods = ["Stable-AGDM", "GDM",  "DiffGDM"
                        ]
    if "DiffGDM" in method:
        if kwargs.get("DiffDGM_infer_param") is not None:
            DiffDGM_infer_param = kwargs.get("DiffDGM_infer_param")

            DiffDGM_infer_param.update(DiffGDM_param_update(method))
            if DiffDGM_infer_param["rewrite"]:

                infer_direction = DiffDGM_infer_param["infer_direction"]

                if "_{}".format( infer_direction) in method:
                    method = method.replace("_{}".format( infer_direction), "")
            else:
                method = method.replace("_base", "")

    G = G.copy()
    num_nodes = nx.number_of_nodes(G)
    # if 0 in G.nodes:
    #     nx.relabel_nodes(G, {0: num_nodes}, copy=False)
    #     iszero = True
    # else:
    #     iszero = False
    iszero = False
    if method == "CI":
        ##def CI(G,target,biggestComponentEndThreshold=0.01,ballRadius=2,method=0,outputNumBatch=1,updateBatch=1,isPrintMinPointCausingMinComponent=True):
        remove_list, time = CI(G=G, target=target, biggestComponentEndThreshold=biggestComponentEndThreshold,
                               ballRadius=ballRadius, CImethod=CImethod,rein=models_REIN,
                               updateBatch=updateBatch,infer_param=kwargs["DiffDGM_infer_param"] if kwargs.get("DiffDGM_infer_param") else None)
    elif method == 'BPD':
        if 0 in G.nodes:
            nx.relabel_nodes(G, {0: num_nodes}, copy=False)
            iszero = True
        else:
            iszero = False
        remove_list, time = BPD(G=G, target=target)
    elif method == 'DomiRank':

        remove_list, time = Domirank_break(G=G, target=target,Rein=models_REIN,
                                           infer_param=kwargs["DiffDGM_infer_param"] if kwargs.get("DiffDGM_infer_param") else None)
    elif method == 'GND':
        #def GND(G, target, remove_strategy=3, plot_size=1, sort_strategy=2,REIN=False):
        remove_list, time = GND(G=G, target=target,remove_strategy=remove_strategy,sort_strategy=sort_strategy,REIN=GND_REIN)
    elif method == "HDA" or method == 'HBA' or method == 'CLA' or method == 'EIGA' :
        remove_list, time = iter_method(method=method, G=G, target=target)
    elif  method in ["HD","HB","HSubgraph","HCluster","HClose","HEIG","HKCore"]:
        remove_list,time = onepass_method(method=method, G=G, target=target,REIN=False)
    elif  method in ["HD_R","HB_R"]:
        remove_list,time = onepass_method(method=method, G=G, target=target,REIN=True)
    elif method == "MinSum":
        remove_list, time = minsum(G=G, target=target, treebreakersize=treebreakersize,minsum_rein=MinSum_REIN)
    elif method == 'Greedy':
        remove_list, time = greedy(G=G, target=target)
    elif method == 'CoreHD':
        remove_list, time = corehd(G=G, target=target, treebreakersize=treebreakersize, k=CoreHD_k, scm=scm,corehd_rein=CoreHD_REIN)
    elif method == 'DQN':
        remove_list, time = single_eva(G, target)
    elif method =="NIRM":
        remove_list, time = NIRM_evaluation(G, target, models_REIN,
                                            infer_param=kwargs["DiffDGM_infer_param"] if kwargs.get("DiffDGM_infer_param") else None)
    elif  method=="CoreDiffGDM_Emsemble":
        remove_list, time =CoreDiffGDM_Emsemble(G, target,
                                           graph_name=graph_name,method_name=method,
                                           models_REIN=models_REIN,iszero=iszero,infer_param=kwargs["DiffDGM_infer_param"])
    elif method in modelgym_methods:

        remove_list, time = modelGym_evaluation(G, target,method_name=method,
                                                models_REIN=models_REIN,
                                                graph_name=graph_name,iszero=iszero,
                                                infer_param=kwargs["DiffDGM_infer_param"] if kwargs.get("DiffDGM_infer_param") else None)
    elif method=="random"or method=="random_rein":
        remove_list,time=Random_dismantle(G,target,method_name=method)
    elif "CoreDiffGDM" in method:

        remove_list,time=modelGym_decycle2dismantle_evaluation(G, target,
                                                               graph_name=graph_name,method_name=method,
                                                               models_REIN=True,iszero=iszero,
                                                               infer_param=kwargs["DiffDGM_infer_param"] if kwargs.get("DiffDGM_infer_param") else None)
    else:
        raise Exception("method has not been defined", method,
                        )

    if iszero:

        for i in range(len(remove_list)):
            if remove_list[i] == num_nodes:
                remove_list[i] = 0
    return remove_list, time
# def keyNodes_predict(G,method,DiffDGM_infer_param,graph_name=None):
#     modelgym_methods = ["Stable-AGDM", "GDM", "DiffGDM_D"]
#     if method in modelgym_methods:
#         keynodes_probs,time=modelGym_keynodes_predict(G,method,infer_param=DiffDGM_infer_param,graph_name=graph_name)
#     elif method=="NIRM":
#         keynodes_probs,time=NIRM_predict(G)
#     else:
#         raise Exception("method has not been defined", method,)
#     return keynodes_probs,time

if __name__=='__main__':
    G=nx.barabasi_albert_graph(50,3)
    remove_list=[1,2,3,4,5,6,7,8,9,0,11,12,13,14,15]
