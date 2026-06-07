import bisect
import json
import math
import os
import shutil
import sys
import time
from glob import glob

import igraph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from analysis_visi.LCC_size_removed_nodes import draw_robustness_curves, draw_dismantle_statistics_real, \
    draw_dismantle_methods_graphs_real, threshold_dismantlesize_curve, get_dismantl_curves_data, \
    draw_dismantle_statistics_synth, draw_dismantle_methods_graphs_synth, draw_dismantle_methods_synth_curve, \
    draw_runtime_methods_synth_curve
from analysis_visi.correlation_analysis import kendall_correlation_analysis,set_correlation_analysis_average,set_correlation_analysis
from analysis_visi.network_dismantling_topology import draw_dimantle_nodes
from method.modelGym.models import model_rank
from method.method import NodeBreakList, nodereinsert, DecycleList
from utils.utils import features_for_graph, get_median, graph_properties_calculate
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
def draw_LCCcurve_methods(methods,target,graph_path,nep=False,**kwargs):
    g = igraph.Graph.Read_GraphML(graph_path)
    G = g.to_networkx()
    G.remove_edges_from(nx.selfloop_edges((G)))  # 去自环
   # pool = multiprocessing.Pool()
    dismantle_size = {}
    dismantle_time = {}
    remove_nodes_dic = {}
    for method in methods:
        print("---------methods_name-----------------")
        print(method)
        remove_list, cost_time = NodeBreakList(G, method=method, target=target, **kwargs)  #
        if nep == True:
            remove_list = nodereinsert(G=G, remove_list=remove_list)
        remove_nodes_dic[method] = remove_list
        dismantle_time[method] = cost_time
        dismantle_size[method] = len(remove_list) / nx.number_of_nodes(G)

    for method in methods:
        print("{}\n dismantle size:{}  dismantle time:{}".format(method,dismantle_size[method],dismantle_time[method]))
    fig = plt.figure(num="robustness_curves 测试")
    # ax = fig.add_subplot(122)
    ax = fig.gca()
    draw_robustness_curves(ax=ax, remove_nodes_dic=remove_nodes_dic, network=G, title=True, data_name="test graph")
    plt.show()
    return dismantle_size,dismantle_time,remove_nodes_dic
def draw_LCCcurve_graphs(method_list,graphs_path,methods_nodelist_path,target=20,Rein=True,subplot_numCols=2,graphs_name=None):

    re_graphs_dismantle_nodelist={}
    with open(methods_nodelist_path+'/graphs_dismantle_nodelist_target_{}_{}.json'.format(target,Rein),"r") as f:
        graphs_dismantle_nodelist=json.load(f)
    if graphs_name is None:
        graphs_name = list(graphs_dismantle_nodelist.keys())
    for i, graph_name in enumerate(graphs_name):
        re_graphs_dismantle_nodelist[graph_name] = {}
        for method in method_list:
            re_graphs_dismantle_nodelist[graph_name][method] = graphs_dismantle_nodelist[graph_name][method]
    fig = plt.figure(figsize=(3,6))
    subplot_nums = len(graphs_name)
    subplot_numRows = int(np.ceil(subplot_nums / subplot_numCols))
    subplot_iter = 1
    for graph_name in graphs_name:
        file = graphs_path + "/{}.graphml".format(graph_name)
        g = igraph.Graph.Read_GraphML(file)
        G = g.to_networkx()
       # G.remove_edges_from(nx.selfloop_edges((G)))  # 去自环
        print("graph name:{}".format(graph_name))
        ax = fig.add_subplot(subplot_numRows, subplot_numCols, subplot_iter)
        remove_nodes_dic=re_graphs_dismantle_nodelist[graph_name]
        draw_robustness_curves(ax=ax, remove_nodes_dic=remove_nodes_dic, network=G, title=True, data_name=graph_name)
        subplot_iter += 1
    plt.show()

def dismantle_methods(graph_name,G,methods,target,
                      graphs_dismantle_time,graphs_dismantle_size,graphs_dismantle_nodelist,nep=False,**kwargs):
    dismantle_time = {}
    dismantle_size = {}
    dismantle_list={}
    methods_record=["FINDER","CoreDQN"]

    for method in methods:
        print("---------methods_name-----------------")
        print(method)
        if graphs_dismantle_nodelist.get(graph_name) is not None:
            if graphs_dismantle_nodelist[graph_name].get(method):  # None or []
                print("the result {} of {} already exists".format(method, graph_name))
                dismantle_list[method] = graphs_dismantle_nodelist[graph_name][method]
                dismantle_time[method] = graphs_dismantle_time[graph_name][method]
                dismantle_size[method] = graphs_dismantle_size[graph_name][method]
                continue
        if method in methods_record:
            method_file_path = "method/{}".format(method)
            with open(method_file_path + '/synth_findernd_step0.01_t{}_graph_dismantle.json'.format(target), "r") as f:
                method_graph_dismantle_list = json.load(f)

            remove_list=method_graph_dismantle_list["{}.graphml".format(graph_name)]["nodelist"]
            cost_time=method_graph_dismantle_list["{}.graphml".format(graph_name)]["time"]

        else:
            remove_list, cost_time = NodeBreakList(G, method=method,graph_name=graph_name, target=target,**kwargs)  #
        if nep == True:
            remove_list= nodereinsert(G=G, remove_list=remove_list)
        dismantle_list[method] = remove_list
        dismantle_time[method] = cost_time
        dismantle_size[method] = len(remove_list) / nx.number_of_nodes(G)
        print("dismantle size",dismantle_size[method])

    graphs_dismantle_time.update({graph_name:dismantle_time})
    graphs_dismantle_size.update({graph_name:dismantle_size})
    graphs_dismantle_nodelist.update({graph_name:dismantle_list})

def dismantle_graph_properties(graph_property,graph_name,G,methods,target,
                               graphs_properties_nodelist,graphs_properties_AUC,
                      graphs_dismantle_time,graphs_dismantle_size,graphs_dismantle_nodelist,nep=False,**kwargs):
    dismantle_time = {}
    dismantle_size = {}
    dismantle_list={}
    dismantle_graph_property_list={}
    dismantle_graph_property_AUC={}
    methods_record=["FINDER","CoreDQN"]

    for method in methods:
        print("---------methods_name-----------------")
        print(method)

        if graphs_properties_nodelist.get(graph_name) is not None:
            if graphs_properties_nodelist[graph_name].get(method):  # None or []
                dismantle_graph_property_list[method] = graphs_properties_nodelist[graph_name][method]
                dismantle_graph_property_AUC[method] = graphs_properties_AUC[graph_name][method]
                dismantle_list[method] = graphs_dismantle_nodelist[graph_name][method]
                dismantle_time[method] = graphs_dismantle_time[graph_name][method]
                dismantle_size[method] = graphs_dismantle_size[graph_name][method]
                continue

        if graphs_dismantle_nodelist.get(graph_name) is not None and graphs_dismantle_nodelist[graph_name].get(method):

            print("the result {} of {} already exists".format(method, graph_name))
            remove_list = graphs_dismantle_nodelist[graph_name][method]
            cost_time = graphs_dismantle_time[graph_name][method]

        elif method in methods_record:
            method_file_path = "method/{}".format(method)
            with open(method_file_path + '/findernd_step0.01_t{}_graph_dismantle.json'.format(target), "r") as f:
                method_graph_dismantle_list = json.load(f)

            remove_list=method_graph_dismantle_list["{}.graphml".format(graph_name)]["nodelist"]
            cost_time=method_graph_dismantle_list["{}.graphml".format(graph_name)]["time"]

        else:
            remove_list, cost_time = NodeBreakList(G, method=method,graph_name=graph_name, target=target,**kwargs)  #
        if nep == True:
            remove_list= nodereinsert(G=G, remove_list=remove_list)
        dismantle_graph_property_list[method],dismantle_graph_property_AUC[method] = cp_graph_properties_nodelist(G, graph_property, remove_list)
        dismantle_list[method] = remove_list
        dismantle_time[method] = cost_time
        dismantle_size[method] = len(remove_list) / nx.number_of_nodes(G)


    graphs_dismantle_time[graph_name] = dismantle_time
    graphs_dismantle_size[graph_name] = dismantle_size
    graphs_dismantle_nodelist[graph_name]=dismantle_list
    graphs_properties_nodelist[graph_name]=dismantle_graph_property_list
    graphs_properties_AUC[graph_name]=dismantle_graph_property_AUC
def cp_graph_properties_nodelist(G,graph_property,node_list):
    print("computing graph property {} for node_list".format(graph_property))
    G=G.copy()
    graph_num_nodes = G.number_of_nodes()
    dismantle_graph_property_list=[]
    AUC=0
    for node in tqdm(node_list):
        G.remove_node(node)
        if len(G.nodes())==0:
            dismantle_graph_property_list.append(0)
            AUC+=0
            break
        else:
            dismantle_graph_property=graph_properties_calculate(G, graph_property)/graph_num_nodes
            dismantle_graph_property_list.append(dismantle_graph_property)
            AUC+=dismantle_graph_property

    if len(node_list)==graph_num_nodes:
        AUC=AUC/graph_num_nodes
        return dismantle_graph_property_list,AUC
    else:
        remove_lcc_size = len(max(nx.connected_components(G), key=len))
      #  print("remove_lcc_size:{}".format(remove_lcc_size))
        assert remove_lcc_size <= 1, "remove_lcc_size is greater than 1"
        last_dismantle_graph_property=dismantle_graph_property_list[-1]
        remain_nodes_num=graph_num_nodes-len(node_list)
        for _ in range(remain_nodes_num-1):
            dismantle_graph_property_list.append(last_dismantle_graph_property)
            AUC+=last_dismantle_graph_property
        AUC=AUC/graph_num_nodes
        return dismantle_graph_property_list,AUC
def dismantle_statistics_methods(target,methods,dataset_path,Rein,
                                 graphs_dismantle_time,graphs_dismantle_size,graphs_dismantle_nodelist,
                                 save_path="result",filter="arenas*",nep=False,**kwargs):
    if os.path.exists(save_path):
        print("records_path:{}文件夹目录已存在".format(save_path))

    else:
        os.mkdir(save_path)
    features = ["Degree", "CC", "Kcore", "Chi degree"]
    graphs_name = []

    graph_max_nodes = 2000000  # 系统最大整数值 可视为Inf


   # graph_max_nodes = 1000 #
    #  pool = multiprocessing.Pool()

    for file in sorted(glob(dataset_path + '/' + filter + ".graphml")):

        g = igraph.Graph.Read_GraphML(file)
        G = g.to_networkx()
        G.remove_edges_from(nx.selfloop_edges((G)))  # 去自环
        G.remove_nodes_from(list(nx.isolates(G)))  # 去孤立节点
        G = nx.convert_node_labels_to_integers(G)  # 将节点编号从0开始,重新编号
        #features_for_graph(G, features_list=features)
        G_nodes=G.number_of_nodes()
        if G_nodes<=graph_max_nodes:
            graph_name = file[len(dataset_path) + 1:].replace(".graphml", "")

            graphs_name.append(graph_name)
            print("-----------networks------------")
            print("{}_size:{}".format(graph_name, G_nodes))
            CI_updateBatch=math.ceil(0.005*G_nodes)
            dismantle_methods(graph_name,G,methods,target,
                              graphs_dismantle_time,graphs_dismantle_size,graphs_dismantle_nodelist,
                              nep=nep,updateBatch=CI_updateBatch,
                              CoreHD_REIN=Rein,MinSum_REIN=Rein,GND_REIN=Rein,models_REIN=Rein,**kwargs)
        with open(save_path + '/graphs_dismantle_size_target_{}_{}.json'.format(target,Rein), 'w') as f:
            json.dump(graphs_dismantle_size, f, indent=4, separators=(',', ':'))
        with open(save_path + '/graphs_dismantle_time_target_{}_{}.json'.format(target,Rein), 'w') as f:
            json.dump(graphs_dismantle_time, f, indent=4, separators=(',', ':'))
        with open(save_path + '/graphs_dismantle_nodelist_target_{}_{}.json'.format(target,Rein), 'w') as f:
            json.dump(graphs_dismantle_nodelist, f, indent=4, separators=(',', ':'))
def dismantle_graph_properties_ROC_methods(graph_property,target,methods,dataset_path,Rein,
                                      graphs_properties_nodelist,graphs_properties_AUC,
                                 graphs_dismantle_time,graphs_dismantle_size,graphs_dismantle_nodelist,
                                 save_path="result",filter="*",nep=False,**kwargs):
    if os.path.exists(save_path):
        print("records_path:{}文件夹目录已存在".format(save_path))

    else:
        os.mkdir(save_path)

    graphs_name = []
    #graphs_name_filter = ["subelj_jdk_jdk"]
    graphs_name_filter = ["arenas-meta","corruption","moreno_crime_projected","econ-wm1","inf-USAir97","maayan-foodweb"]
    graph_max_nodes = 1000  # 系统最大整数值 可视为Inf


   # graph_max_nodes = 1000 #
    #  pool = multiprocessing.Pool()

    for file in sorted(glob(dataset_path + '/' + filter + ".graphml")):
        graph_name = file[len(dataset_path) + 1:].replace(".graphml", "")
        if graph_name not in graphs_name_filter:
            continue

        g = igraph.Graph.Read_GraphML(file)
        G = g.to_networkx()
        G.remove_edges_from(nx.selfloop_edges((G)))  # 去自环
        G.remove_nodes_from(list(nx.isolates(G)))  # 去孤立节点
        G = nx.convert_node_labels_to_integers(G)  # 将节点编号从0开始,重新编号
        #features_for_graph(G, features_list=features)
        G_nodes=G.number_of_nodes()
        if G_nodes<=graph_max_nodes:

            graphs_name.append(graph_name)
            print("-----------networks------------")
            print("{}_size:{}".format(graph_name, G_nodes))
            CI_updateBatch=math.ceil(0.005*G_nodes)
            dismantle_graph_properties(graph_property,graph_name,G,methods,target,
                                       graphs_properties_nodelist,graphs_properties_AUC,
                              graphs_dismantle_time,graphs_dismantle_size,graphs_dismantle_nodelist,
                              nep=nep,updateBatch=CI_updateBatch,
                              CoreHD_REIN=Rein,MinSum_REIN=Rein,GND_REIN=Rein,models_REIN=Rein,**kwargs)

        with open(save_path + '/graphs_dismantle_size_target_{}_{}.json'.format(target,Rein), 'w') as f:
            json.dump(graphs_dismantle_size, f, indent=4, separators=(',', ':'))
        with open(save_path + '/graphs_dismantle_time_target_{}_{}.json'.format(target,Rein), 'w') as f:
            json.dump(graphs_dismantle_time, f, indent=4, separators=(',', ':'))
        with open(save_path + '/graphs_dismantle_nodelist_target_{}_{}.json'.format(target,Rein), 'w') as f:
            json.dump(graphs_dismantle_nodelist, f, indent=4, separators=(',', ':'))
        with open(save_path + '/graphs_properties_{}_nodelist.json'.format(graph_property), 'w') as f:
            json.dump(graphs_properties_nodelist, f, indent=4, separators=(',', ':'))
        with open(save_path + '/graphs_properties_{}_AUC.json'.format(graph_property), 'w') as f:
            json.dump(graphs_properties_AUC, f, indent=4, separators=(',', ':'))

def decycle_methods(graph_name,G,methods,
                      graphs_dismantle_time,graphs_dismantle_size,graphs_dismantle_nodelist,nep=False,**kwargs):
    dismantle_time = {}
    dismantle_size = {}
    dismantle_list={}
    methods_record=["CoreDQN"]

    for method in methods:
        print("---------methods_name-----------------")
        print(method)

        if graphs_dismantle_nodelist.get(graph_name) is not None:
            if graphs_dismantle_nodelist[graph_name].get(method) :#None or []

                print("the result {} of {} already exists".format(method, graph_name))
                dismantle_list[method] = graphs_dismantle_nodelist[graph_name][method]
                dismantle_time[method] = graphs_dismantle_time[graph_name][method]
                dismantle_size[method] = graphs_dismantle_size[graph_name][method]
                continue
        if method in methods_record:
            method_file_path = "method/{}".format(method)
            with open(method_file_path + '/synth_findernd_step0.01_t1_graph_dismantle.json', "r") as f:
                method_graph_decycle_list = json.load(f)

            remove_list = method_graph_decycle_list["{}.graphml".format(graph_name)]["decycle_list"]
            cost_time = method_graph_decycle_list["{}.graphml".format(graph_name)]["time"]
        else:
            remove_list, cost_time = DecycleList(G, method=method,graph_name=graph_name,**kwargs)  #
        if nep == True:
            remove_list= nodereinsert(G=G, remove_list=remove_list)
        dismantle_list[method]=remove_list
        dismantle_time[method] = cost_time
        dismantle_size[method] = len(remove_list) / nx.number_of_nodes(G)

    graphs_dismantle_time[graph_name] = dismantle_time
    graphs_dismantle_size[graph_name] = dismantle_size
    graphs_dismantle_nodelist[graph_name]=dismantle_list

def decycle_statistics_methods(methods,dataset_path,
                               graphs_dismantle_time,graphs_dismantle_size,graphs_dismantle_nodelist,
                               save_path="result",filter="arenas*",**kwargs):
    if os.path.exists(save_path):
        print("records_path:{}文件夹目录已存在".format(save_path))

    else:
        os.mkdir(save_path)

    graphs_name = []

    graph_max_nodes=200000# 系统最大整数值 可视为Inf
   # graph_max_nodes = 1000 #
    #  pool = multiprocessing.Pool()

    for file in sorted(glob(dataset_path + '/' + filter + ".graphml")):

        g = igraph.Graph.Read_GraphML(file)
        G = g.to_networkx()
        G.remove_edges_from(nx.selfloop_edges((G)))  # 去自环
        G.remove_nodes_from(list(nx.isolates(G)))  # 去孤立节点
        G = nx.convert_node_labels_to_integers(G)  # 将节点编号从0开始,重新编号
        #features_for_graph(G, features_list=features)
        G_nodes=G.number_of_nodes()
        if G_nodes<=graph_max_nodes:
            graph_name = file[len(dataset_path) + 1:].replace(".graphml", "")
            graphs_name.append(graph_name)
            print("-----------networks------------")
            print("{}_size:{}".format(graph_name, G_nodes))
            decycle_methods(graph_name,G,methods,
                              graphs_dismantle_time,graphs_dismantle_size,graphs_dismantle_nodelist,**kwargs
                             )
        with open(save_path + '/graphs_decycle_size.json', 'w') as f:
            json.dump(graphs_dismantle_size, f, indent=4, separators=(',', ':'))
        with open(save_path + '/graphs_decycle_time.json', 'w') as f:
            json.dump(graphs_dismantle_time, f, indent=4, separators=(',', ':'))
        with open(save_path + '/graphs_decycle_nodelist.json', 'w') as f:
            json.dump(graphs_dismantle_nodelist, f, indent=4, separators=(',', ':'))
def draw_statistics_methods(method_list,graphs_dismantle_size,
                            vision_mode="statistics",dataset_type="real",
                            task_type="dismantling",
                            graph_filter=None):
    #
  #  filter=["advogato","moreno_crime_projected","opsahl-ucsocial","ego-twitter",]#,"arenas-meta","corruption","subelj_jung-j_jung-j","power-eris1176","econ-wm1"
   # filter=["subelj_jdk_jdk",]

    # with open(file_path + '/graphs_dismantle_size_MVC_True.json', 'r') as f:
    #     graphs_dismantle_size=json.load(f)
    fig = plt.figure(num="{} complex systems -set_size 测试".format(task_type),figsize=(10,6))
    # ax = fig.add_subplot(122)
    ax = fig.gca()
    re_graphs_dismantle_size={}

    graphs_name = list(graphs_dismantle_size.keys())
    #
   # graph_filter="200000"
   #  if graph_filter is not None:
   #
   #      graphs_name = [name for name in graphs_name if graph_filter not in name]
   #  if graph_filter is not None:
   #     for graph_name in graph_filter:
   #         index = graphs_name.index(graph_name)
   #         graphs_name.pop(index)
   # print("graphs_name:{}".format(graphs_name))
   #  if graph_filter is None:
   #      graph_filter=list(graphs_dismantle_size.keys())
    for i, graph_name in enumerate(graphs_name):
     #   print("graph_name:{}".format(graph_name))
        re_graphs_dismantle_size[graph_name]={}
        for method in method_list:
            re_graphs_dismantle_size[graph_name][method]=graphs_dismantle_size[graph_name][method]


    if vision_mode=="statistics":
        if "real" in dataset_type:
            draw_dismantle_statistics_real(ax,re_graphs_dismantle_size)
        elif "synth" in dataset_type:
            draw_dismantle_statistics_synth(ax,re_graphs_dismantle_size)
    elif vision_mode=="table":
        if  "real" in dataset_type:
            draw_dismantle_methods_graphs_real(ax,re_graphs_dismantle_size)
        elif "synth" in dataset_type:
            draw_dismantle_methods_graphs_synth(ax, re_graphs_dismantle_size)
    elif vision_mode=="curve":
        if "synth" in dataset_type:
            draw_dismantle_methods_synth_curve(ax, re_graphs_dismantle_size)

    plt.show()
    plt.tight_layout()




def create_performance_table(data_dict, output_file='performance_table.xlsx'):
    """
    创建方法性能统计表，每行一个方法，每列一个尺寸，显示均值和方差

    参数:
    data_dict: 包含实验数据的字典
    output_file: 输出的Excel文件名
    """
    # 提取所有尺寸和方法
    sizes = set()
    methods = set()

    for exp_name, exp_values in data_dict.items():
        size = exp_name.split('_')[2].split('.')[0]  # 提取尺寸
        sizes.add(size)
        methods.update(exp_values.keys())

    sizes = sorted(sizes, key=lambda x: int(x.split('-')[0]))  # 按数字排序
    methods = sorted(methods)

    # 准备数据
    mean_data = []
    var_data = []

    for method in methods:
        mean_row = {'Method': method}
        var_row = {'Method': method}

        for size in sizes:
            # 收集该方法和尺寸下的所有值
            values = []
            for exp_name, exp_values in data_dict.items():
                if exp_name.split('_')[2].split('.')[0] == size:
                    if method in exp_values:
                        values.append(exp_values[method])

            # 计算统计值
            if values:
                mean_val = np.mean(values)
                var_val = np.var(values)
            else:
                mean_val = np.nan
                var_val = np.nan

            mean_row[size] = mean_val*100
            var_row[size] = var_val*100

        mean_data.append(mean_row)
        var_data.append(var_row)

    # 创建DataFrame
    mean_df = pd.DataFrame(mean_data).set_index('Method')
    var_df = pd.DataFrame(var_data).set_index('Method')

    # 合并均值和方差为单个表格（格式：均值±标准差）
    combined_df = pd.DataFrame(index=methods)
    for size in sizes:
        combined_df[f'{size}_Mean±Std'] = [
            f"{mean_df.loc[method, size]:.2f}±{np.sqrt(var_df.loc[method, size]):.2f}"
            if not np.isnan(mean_df.loc[method, size]) else "N/A"
            for method in methods
        ]

    # 保存为Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 合并表格（均值±标准差）
        combined_df.to_excel(writer, sheet_name='Mean±Std')

        # 单独的均值表
        mean_df.to_excel(writer, sheet_name='Mean_Only')

        # 单独的方差表
        var_df.to_excel(writer, sheet_name='Variance_Only')

        # 原始数据表
        raw_data = []
        for exp_name, exp_values in data_dict.items():
            size = exp_name.split('_')[2].split('.')[0]
            row = {'Experiment': exp_name, 'Size': size}
            row.update(exp_values)
            raw_data.append(row)

        raw_df = pd.DataFrame(raw_data)
        raw_df.to_excel(writer, sheet_name='Raw_Data', index=False)

    print(f"性能统计表已保存到: {output_file}")
    return mean_df, var_df, combined_df
def draw_statistics_properties_methods(method_list,graphs_properties_AUC,graph_property=None,
                            vision_mode="statistics",dataset_type="real",

                            filter=None):
    #
  #  filter=["advogato","moreno_crime_projected","opsahl-ucsocial","ego-twitter",]#,"arenas-meta","corruption","subelj_jung-j_jung-j","power-eris1176","econ-wm1"
    graph_filter=["arenas-meta","corruption","moreno_crime_projected","econ-wm1","inf-USAir97","maayan-foodweb"]#"corruption",,"moreno_crime_projected"


    # with open(file_path + '/graphs_dismantle_size_MVC_True.json', 'r') as f:
    #     graphs_dismantle_size=json.load(f)
    fig = plt.figure(num="AUC -{} 测试".format(graph_property),figsize=(12,6))
    # ax = fig.add_subplot(122)
    ax = fig.gca()
    re_graphs_dismantle_size={}

    # graphs_name = list(graphs_properties_AUC.keys())
    # if filter is not None:
    #     for graph_name in filter:
    #         index = graphs_name.index(graph_name)
    #         graphs_name.pop(index)
   # print("graphs_name:{}".format(graphs_name))
    for i, graph_name in enumerate(graph_filter):
        print("graph_name:{}".format(graph_name))
        re_graphs_dismantle_size[graph_name]={}
        for method in method_list:
            re_graphs_dismantle_size[graph_name][method]=graphs_properties_AUC[graph_name][method]
    if vision_mode=="statistics":
        if "real" in dataset_type:
            draw_dismantle_statistics_real(ax,re_graphs_dismantle_size)
        elif "synth" in dataset_type:
            draw_dismantle_statistics_synth(ax,re_graphs_dismantle_size)
    elif vision_mode=="table":
        if  "real" in dataset_type:
            draw_dismantle_methods_graphs_real(ax,re_graphs_dismantle_size)
        elif "synth" in dataset_type:
            draw_dismantle_methods_graphs_synth(ax, re_graphs_dismantle_size)


    plt.show()
    plt.tight_layout()

def draw_kendall_correlation_all(features,method_list,save_data_path,graphs_path,model_output_mode,filter="*",graphs_name=None):
    from scipy import stats
    if os.path.exists(save_data_path):
        print("records_path:{}文件夹目录已存在".format(save_data_path))

    else:
        os.mkdir(save_data_path)

    rank_names=features
    size = len(rank_names)
    fig, ax = plt.subplots(figsize=(2 * size, size), facecolor='w')  # 设置画布大小，和底色
    rank_correlation={}


    count=1
    exclude_set=set()
    if graphs_name == None:
        for file in sorted(glob(graphs_path + '/' + filter + ".graphml")):

            g = igraph.Graph.Read_GraphML(file)
            G = g.to_networkx()
            # G.remove_edges_from(nx.selfloop_edges((G)))  # 去自环
            graph_name = file[len(graphs_path) + 1:].replace(".graphml", "")
            print("graph name:{} \n number of nodes:{}".format(graph_name, G.number_of_nodes()))
            features_for_graph(G, features_list=features)
            method_rank = {}
            if method_list is not None:
                for model_name in method_list:
                    print(model_name)
                    if model_output_mode == "output":
                        method_rank[model_name] = model_rank(G, method_name=model_name,mode=model_output_mode) #模型输出指标
                        if count==1:
                            rank_correlation[model_name]={}
                    elif model_output_mode =="prefeatures" or model_output_mode =="embfeatures":
                        model_features=model_rank(G, method_name=model_name,mode=model_output_mode)
                        for i,model_feature in enumerate(model_features):
                            method_rank[str(i)]=model_feature

            for feature in features:
                feature_list = list(nx.get_node_attributes(G, name=feature).values())
                method_rank[feature] = feature_list
            for method_name_1, values_1 in method_rank.items():
                if count == 1:
                    rank_correlation[method_name_1]={}
               # print("method_name: {}".format(method_name_1))
                for method_name_2, values_2 in method_rank.items():
                    rank_cor, _ = stats.kendalltau(x=values_1, y=values_2)
                   # print("method_name_2:{}".format(method_name_2))
                   #  if rank_cor is np.nan: #某变量 全为0,则计算为nan
                   #      continue
                        #raise ValueError("rank_cor values is nan in position: ({},{}) ".format(method_name_1,method_name_2))
                    if count ==1:
                        rank_correlation[method_name_1][method_name_2]=float(rank_cor)
                    else:
                        rank_correlation[method_name_1][method_name_2]=(count-1)/count*float(rank_correlation[method_name_1][method_name_2])+1/count*float(rank_cor)

            count+=1
    else:
        fig = plt.figure()
        subplot_nums = len(graphs_name)
        subplot_numCols = 1
        subplot_numRows = int(np.ceil(subplot_nums / subplot_numCols))
        subplot_iter = 1
        for graph_name in graphs_name:
            file = graphs_path + "/{}.graphml".format(graph_name)
            g = igraph.Graph.Read_GraphML(file)
            G = g.to_networkx()
            # G.remove_edges_from(nx.selfloop_edges((G)))  # 去自环
            graph_name = file[len(graphs_path) + 1:].replace(".graphml", "")
            print("graph name:{} \nnumber of nodes:{}".format(graph_name, G.number_of_nodes()))
            features_for_graph(G, features_list=features)
            method_rank = {}
            for model_name in method_list:
                method_rank[model_name] = model_rank(G, method_name=model_name)

            for feature in features:
                feature_list = list(nx.get_node_attributes(G, name=feature).values())
                method_rank[feature] = feature_list

            size = len(method_rank.keys())
            ax = fig.add_subplot(subplot_numRows, subplot_numCols, subplot_iter)
            # fig, ax = plt.subplots(figsize=(2 * size, size), facecolor='w')  # 设置画布大小，和底色
            kendall_correlation_analysis(ax=ax, values_dic=method_rank,
                                         title=True, dataname=graph_name)
            subplot_iter += 1
        plt.show()

    import seaborn as sns
    import pandas as pd
    title=False
    dataname="test"

    df = pd.DataFrame(rank_correlation)
    # df=df.dropna(axis=0,how="all")#删除全为nan的行
    # df=df.dropna(axis=1,how="all")#删除全为nan的列

    #sns.set(font_scale=1.5)
    ax = sns.heatmap(df, ax=ax, annot=True,linewidths=0.5, linecolor="#E8E8E8",vmax=1, vmin=-1, square=False, cmap="YlGnBu", fmt='.2g', annot_kws={"fontsize": 20})
    # plt.setp(ax.get_xticklabels(), rotation="horizontal", fontsize=15)  # "vertical"竖立  horizontal 横放
    # plt.setp(ax.get_yticklabels(), fontsize=15)  # "vertical"竖立  horizontal 横放
    if title == True:
        ax.set_title("{}".format(dataname), fontsize=30)
    plt.tight_layout()
    # ax.patch.set_facecolor('MistyRose')
    # ax.patch.set_alpha(0.5)
    plt.rc('font', family='Arial', size=12)
    # fig.savefig(save_data_path + "/real_networks_features_correlation_all.eps",format="eps", bbox_inches="tight")
    # plt.close(fig)
    plt.show()
def draw_kendall_correlation(features,method_list,save_data_path,graphs_path,model_output_mode,graphs_name=None):


    if os.path.exists(save_data_path):
         print("records_path:{}文件夹目录已存在".format(save_data_path))

    else:
        os.mkdir(save_data_path)
    if graphs_name==None:
        for file in sorted(glob(graphs_path + '/' + "*" + ".graphml")):

             g = igraph.Graph.Read_GraphML(file)
             G = g.to_networkx()
            # G.remove_edges_from(nx.selfloop_edges((G)))  # 去自环
             graph_name = file[len(graphs_path) + 1:].replace(".graphml", "")
             print("graph name:{} \n number of nodes:{}".format(graph_name,G.number_of_nodes()))
             features_for_graph(G,features_list=features)
             method_rank={}
             if method_list is not None:
                 for model_name in method_list:
                     print(model_name)
                     if model_output_mode == "output":
                         method_rank[model_name] = model_rank(G, method_name=model_name,
                                                              mode=model_output_mode)  # 模型输出指标
                     elif model_output_mode == "prefeatures" or model_output_mode == "prefeatures":
                         model_features = model_rank(G, method_name=model_name, mode=model_output_mode)
                         for i, model_feature in enumerate(model_features):
                             method_rank[str(i)] = model_feature

             for feature in features:
                 feature_list=list(nx.get_node_attributes(G,name=feature).values())
                 method_rank[feature]=feature_list

             size = len(method_rank.keys())
             fig, ax = plt.subplots(figsize=(2 * size, size), facecolor='w')  # 设置画布大小，和底色
             kendall_correlation_analysis(ax=ax,values_dic=method_rank,
                                          title=True,dataname=graph_name)

             fig.savefig(save_data_path + "/{}".format(graph_name), bbox_inches="tight")

             plt.close(fig)
    else:
        fig = plt.figure()
        subplot_nums = len(graphs_name)
        subplot_numCols=1
        subplot_numRows = int(np.ceil(subplot_nums / subplot_numCols))
        subplot_iter = 1
        for graph_name in graphs_name:
            file=graphs_path+"/{}.graphml".format(graph_name)
            g = igraph.Graph.Read_GraphML(file)
            G = g.to_networkx()
            # G.remove_edges_from(nx.selfloop_edges((G)))  # 去自环
            graph_name = file[len(graphs_path) + 1:].replace(".graphml", "")
            print("graph name:{} \nnumber of nodes:{}".format(graph_name, G.number_of_nodes()))
            features_for_graph(G, features_list=features)
            method_rank = {}
            for model_name in method_list:
                method_rank[model_name] = model_rank(G, method_name=model_name)

            for feature in features:
                feature_list = list(nx.get_node_attributes(G, name=feature).values())
                method_rank[feature] = feature_list

            size = len(method_rank.keys())
            ax = fig.add_subplot(subplot_numRows, subplot_numCols, subplot_iter)
           # fig, ax = plt.subplots(figsize=(2 * size, size), facecolor='w')  # 设置画布大小，和底色
            kendall_correlation_analysis(ax=ax, values_dic=method_rank,
                                         title=True, dataname=graph_name)
            subplot_iter+=1
        plt.show()
    #


def draw_correlation_methods(target,Rein,method_list,cor_mode,methods_nodelist_path,order_rank_sort=True):
    re_graphs_dismantle_nodelist = {}
    rank_nodes_dic={}
    with open(methods_nodelist_path + '/graphs_dismantle_nodelist_target_{}_{}.json'.format(target, Rein), "r") as f:
        graphs_dismantle_nodelist = json.load(f)
    graphs_name = list(graphs_dismantle_nodelist.keys())
    method_list2=list(list(graphs_dismantle_nodelist.values())[0].keys())
    print(method_list2)

    for i, graph_name in enumerate(graphs_name):
        re_graphs_dismantle_nodelist[graph_name] = {}
        for method in method_list:
            re_graphs_dismantle_nodelist[graph_name][method] = graphs_dismantle_nodelist[graph_name][method]
    if cor_mode == "median correlation":
        for method in method_list:
            rank_nodes_dic[method]=[]
            for graph_name in graphs_name:
                if order_rank_sort:#按节点序号大小排序，进而取中位数
                    rank_nodes_dic[method].append(get_median(re_graphs_dismantle_nodelist[graph_name][method]))
                else:
                    half=len(re_graphs_dismantle_nodelist[graph_name][method])//2
                    rank_nodes_dic[method].append(re_graphs_dismantle_nodelist[graph_name][method][half])
        size = len(rank_nodes_dic.keys())
        fig, ax = plt.subplots(figsize=(2 * size, size), facecolor='w')  # 设置画布大小，和底色
        dataname=cor_mode+"_rein-{}_sort-{}".format(Rein,order_rank_sort)+"_{}".format(methods_nodelist_path)
        kendall_correlation_analysis(ax=ax, values_dic=rank_nodes_dic,title=True, dataname=dataname)
    elif cor_mode == "set correlation":
        for method in method_list:
            rank_nodes_dic[method] = {}
            for graph_name in graphs_name:
                rank_nodes_dic[method][graph_name]=set(re_graphs_dismantle_nodelist[graph_name][method])
        size = len(rank_nodes_dic.keys())
        fig, ax = plt.subplots(figsize=(2 * size, size), facecolor='w')  # 设置画布大小，和底色
        dataname = cor_mode + "_rein-{}".format(Rein) + "_{}".format(methods_nodelist_path)
        set_correlation_analysis_average(ax=ax, values_dic=rank_nodes_dic,graphs_list=graphs_name,title=False, dataname=dataname)

    plt.show()
def draw_setcorrelation_methods_graphs(target,Rein,method_list,methods_nodelist_path,save_data_path):
    if os.path.exists(save_data_path):
        print("records_path:{}文件夹目录已存在".format(save_data_path))

    else:
        os.mkdir(save_data_path)
    re_graphs_dismantle_nodelist = {}
    rank_nodes_dic = {}
    with open(methods_nodelist_path + '/graphs_dismantle_nodelist_target_{}_{}.json'.format(target, Rein), "r") as f:
        graphs_dismantle_nodelist = json.load(f)
    graphs_name = list(graphs_dismantle_nodelist.keys())

    for i, graph_name in enumerate(graphs_name):
        re_graphs_dismantle_nodelist[graph_name] = {}
        for method in method_list:

            re_graphs_dismantle_nodelist[graph_name][method] = graphs_dismantle_nodelist[graph_name][method]
    for graph_name in graphs_name:
        for method in method_list:

            rank_nodes_dic[method]= set(re_graphs_dismantle_nodelist[graph_name][method])
        size = len(rank_nodes_dic.keys())
        fig, ax = plt.subplots(figsize=(2 * size, 2*size), facecolor='w')  # 设置画布大小，和底色

        set_correlation_analysis(ax=ax, values_dic=rank_nodes_dic,  title=True,
                                         dataname=graph_name)
        fig.savefig(save_data_path + "/{}_Rein-{}".format(graph_name,Rein), bbox_inches="tight")

        plt.close(fig)




def draw_threshold_dismantlesize_curve(graphs_list, methods_list,threshold_dismantlesize_data,fig_name=None,graph_average=False,graph_property=None):

    if fig_name is None:
        fig = plt.figure("draw_threshold_dismantlesize_curve",figsize=(3,6))

    else:
        fig = plt.figure(fig_name,figsize=(3,6))

    if not graph_average:
        subplot_nums = len(graphs_list)
        subplot_numCols = 1
        subplot_numRows = int(np.ceil(subplot_nums / subplot_numCols))
        subplot_iter = 1
        for graph in graphs_list:
            methods_dismantle_size=threshold_dismantlesize_data[graph]
            ax = fig.add_subplot(subplot_numRows, subplot_numCols, subplot_iter)
            threshold_dismantlesize_curve(ax,methods_dismantle_size,methods_list,graph_property)
            ax.set_title(graph)
            subplot_iter+=1
        plt.show()
    else:
        subplot_nums = 1
        subplot_numCols = 1
        subplot_numRows = int(np.ceil(subplot_nums / subplot_numCols))
        subplot_iter = 1
        methods_dismantle_size=dict()
        for graph in graphs_list:
            for method in methods_list:
                if method not in methods_dismantle_size.keys():
                    methods_dismantle_size[method]=[]
                methods_dismantle_size[method].append(threshold_dismantlesize_data[graph][method])
        for method in methods_list:

            methods_dismantle_size[method]=average_dicts(methods_dismantle_size[method])
        ax = fig.add_subplot(subplot_numRows, subplot_numCols, subplot_iter)
        threshold_dismantlesize_curve(ax,methods_dismantle_size,methods_list,graph_property)
        if fig_name is None:
            ax.set_title("average")  # 平均值
        else:
            ax.set_title(fig_name+"_average")  # 平均值
        plt.show()
def average_dicts(list_of_dicts):
    if not list_of_dicts:
        return {}
    return {key: sum(d[key] for d in list_of_dicts) / len(list_of_dicts) for key in list_of_dicts[0].keys()}
def draw_nodes_graph(target,Rein,methods_list,methods_nodelist_path,dataset_path="dataset/dataset_real",graphs_list=None,visual_dim=2):
    with open(methods_nodelist_path + '/graphs_dismantle_nodelist_target_{}_{}.json'.format(target, Rein),"r") as f:
        graphs_dismantle_nodelist = json.load(f)
    if graphs_list==None:
        graphs_name = list(graphs_dismantle_nodelist.keys())
    else:
        graphs_name=graphs_list
    subplot_nums = len(graphs_name)*len(methods_list)
    subplot_numCols = 2
    subplot_numRows = int(np.ceil(subplot_nums / subplot_numCols))

    subplot_iter = 1
    fig = plt.figure("draw_nodes_graph topology")
    for graph_name in graphs_list:
        file=dataset_path+"/"+graph_name+".graphml"
        g = igraph.Graph.Read_GraphML(file)
        G = g.to_networkx()
        for method in methods_list:
            remove_node = graphs_dismantle_nodelist[graph_name][method]
            ax = fig.add_subplot(subplot_numRows, subplot_numCols, subplot_iter)
            draw_dimantle_nodes(ax=ax,network=G,remove_nodes=remove_node,title=True,data_name=graph_name,method_name=method,visual_dim=visual_dim)
            subplot_iter += 1
    plt.show()

def generate_configure_models(target,Rein,methods,conf_models_num,dataset_path,networks_name=None,save_path=None,**kwargs):
    features = ["degree","clustering_coefficient","kcore","chi_degree"]
    if os.path.exists(save_path):
        print("records_path:{}文件夹目录已存在".format(save_path))

    else:
        os.mkdir(save_path)
    if Rein:
        config_dataset_save_path = save_path + "/dataset"
    else:
        config_dataset_save_path=save_path+"/dataset_norein"
    if os.path.exists(config_dataset_save_path):
        print("records_path:{}文件夹目录已存在".format(config_dataset_save_path))

    else:
        os.mkdir(config_dataset_save_path)
    graph_max_nodes = sys.maxsize  # sys.maxsize# 系统最大整数值 可视为Inf
   # graph_max_nodes=10000
    dismantle_list={}
    dismantle_size={}
    if networks_name==None:
        for file in sorted(glob(dataset_path + "/*.graphml")):
            g = igraph.Graph.Read_GraphML(file)
            network = g.to_networkx()
            G_nodes = network.number_of_nodes()
            if G_nodes <= graph_max_nodes:
                network_name = file[len(dataset_path) + 1:].replace(".graphml", "")
                print("-------------------------------------------")
                print("network :{} nodes:{}".format(network_name,G_nodes))
                degree = nx.degree(network)
                degree = list(zip(*degree))
                deg_sequence = list(degree[1])
                dismantle_list[network_name] = {}
                dismantle_size[network_name] = {}

                for method in methods:
                    print("method:{}".format(method))
                    dismantle_list[network_name][method] = {}
                    dismantle_size[network_name][method] = {}
                    for i in range(conf_models_num):
                        G = nx.configuration_model(deg_sequence,create_using=nx.Graph)
                       # G = nx.Graph(G)
                        G_nodes = G.number_of_nodes()
                        features_for_graph(G, features_list=features)
                        if save_path != None:
                            nx.write_graphml_lxml(G, "{}/{}_{}_configure_{}.graphml".format(config_dataset_save_path, network_name,method, i))
                        remove_list, cost_time = NodeBreakList(G, method=method, target=target, models_REIN=Rein, **kwargs)
                        configure_model_name = "configure_model_{}".format(i)
                        dismantle_list[network_name][method][configure_model_name] = remove_list
                        dismantle_size[network_name][method][configure_model_name] = len(remove_list) / G_nodes
    else:
        for network_name in networks_name:
            file = dataset_path + "/" + network_name + ".graphml"
            g = igraph.Graph.Read_GraphML(file)
            network = g.to_networkx()
            degree = nx.degree(network)
            degree = list(zip(*degree))
            deg_sequence = list(degree[1])
            dismantle_list[network_name]={}
            dismantle_size[network_name]={}

            for method in methods:
                dismantle_list[network_name][method]={}
                dismantle_size[network_name][method]={}
                print(method)
                for i in range(conf_models_num):
                    G=nx.configuration_model(deg_sequence,create_using=nx.Graph)
                    #G=nx.Graph(G)
                    features_for_graph(G, features_list=features)
                    G_nodes=G.number_of_nodes()
                    if save_path!=None:
                        nx.write_graphml_lxml(G, "{}/{}_{}_configure_{}.graphml".format(config_dataset_save_path, network_name,method, i))

                    remove_list, cost_time = NodeBreakList(G, method=method, target=target,
                                                           CoreHD_REIN=Rein, MinSum_REIN=Rein, GND_REIN=Rein,
                                                           models_REIN=Rein,
                                                           **kwargs)
                    configure_model_name="configure_model_{}".format(i)
                    dismantle_list[network_name][method][configure_model_name] = remove_list
                    dismantle_size[network_name][method][configure_model_name] = len(remove_list) / G_nodes


    with open(save_path + '/graphs_dismantle_size_target_{}_{}.json'.format(target,Rein), 'w') as f:
        json.dump(dismantle_size, f, indent=4, separators=(',', ':'))

    with open(save_path + '/graphs_dismantle_nodelist_target_{}_{}.json'.format(target,Rein), 'w') as f:
        json.dump(dismantle_list, f, indent=4, separators=(',', ':'))

def draw_configures2real_curves(target,Rein,method,real_file_path,config_file_path,dataset_path="dataset/dataset_real",subplot_numCols=3,networks_name=None,title=True):
    if Rein==True:
        config_dataset_path = config_file_path + "/dataset"
    else:
        config_dataset_path = config_file_path + "/dataset_norein"
    with open(real_file_path + '/graphs_dismantle_nodelist_target_{}_{}.json'.format(target, Rein), "r") as f:
        real_graphs_dismantle_nodelist = json.load(f)
    with open(config_file_path + '/graphs_dismantle_nodelist_target_{}_{}.json'.format(target, Rein), "r") as f:
        config_graphs_dismantle_nodelist = json.load(f)
    if networks_name==None:
        networks_name = list(config_graphs_dismantle_nodelist.keys())
    if method == "Stable-AGDM":
        old_method = "stable_AGDM"
    else:
        old_method = method
    real_record_path=real_file_path+"/dismantle_curve_record_target_{}_{}.json".format(target,Rein)
    if os.path.exists(real_record_path):
        with open(real_record_path,"r") as f:
            real_record_dict=json.load(f)
            method_list=list(real_record_dict[networks_name[0]].keys())
            real_record=True
            if old_method in method_list:
                real_record_method = True
            else:
                real_record_method = False
    else:
        real_record_dict = {}
        real_record=False
        real_record_method = False
    config_record_path = config_file_path + "/dismantle_curve_record_target_{}_{}_{}.json".format(target, Rein,old_method)
    if os.path.exists(config_record_path):
        with open(config_record_path, "r") as f:
            config_record_dict = json.load(f)
            config_record=True
    else:
        config_record_dict={}
        config_record = False
    print(config_record)
    fig = plt.figure(figsize=(10,4))
    subplot_nums = len(networks_name)
    subplot_numRows = int(np.ceil(subplot_nums / subplot_numCols))
    subplot_iter = 1
    for graph_name in networks_name:
        ax = fig.add_subplot(subplot_numRows, subplot_numCols, subplot_iter)

        file = dataset_path + "/{}.graphml".format(graph_name)
        g = igraph.Graph.Read_GraphML(file)

        network = g.to_networkx()
        assortativity=nx.assortativity.degree_pearson_correlation_coefficient(network)
       # G.remove_edges_from(nx.selfloop_edges((G)))  # 去自环
        print("graph name:{}".format(graph_name))
        remove_nodes=real_graphs_dismantle_nodelist[graph_name][old_method]
        if real_record :
            if real_record_method:
                remove_nodes_per = real_record_dict[graph_name][old_method]["remove_nodes_per"]
                lcc_size_list = real_record_dict[graph_name][old_method]["lcc_size_list"]
            else:
                real_record_dict[graph_name][old_method] = {}
                remove_nodes_per, lcc_size_list = get_dismantl_curves_data(remove_nodes=remove_nodes, network=network)
                real_record_dict[graph_name][old_method]["remove_nodes_per"] = remove_nodes_per
                real_record_dict[graph_name][old_method]["lcc_size_list"] = lcc_size_list
        else:
            real_record_dict[graph_name] = {}
            real_record_dict[graph_name][old_method] = {}
            remove_nodes_per, lcc_size_list = get_dismantl_curves_data(remove_nodes=remove_nodes, network=network)
            real_record_dict[graph_name][old_method]["remove_nodes_per"] = remove_nodes_per
            real_record_dict[graph_name][old_method]["lcc_size_list"] = lcc_size_list

        ax.plot(remove_nodes_per, lcc_size_list,
                color="b",
                linestyle="-",
                marker="."
                )
        if config_record:
            for i, config_model_name in enumerate(config_record_dict[graph_name].keys()):
                remove_nodes_per = config_record_dict[graph_name][config_model_name]["remove_nodes_per"]
                lcc_size_list = config_record_dict[graph_name][config_model_name]["lcc_size_list"]
                ax.plot(remove_nodes_per, lcc_size_list,
                        color="lightskyblue",
                        linestyle="-",
                        )
        else:
            config_record_dict[graph_name] = {}
            for i, config_model_name in enumerate(config_graphs_dismantle_nodelist[graph_name][old_method].keys()):
                file = "{}/{}_{}_configure_{}.graphml".format(config_dataset_path, graph_name, old_method, i)
                G = nx.read_graphml(file, node_type=int)
                remove_nodes = config_graphs_dismantle_nodelist[graph_name][old_method][config_model_name]

                config_record_dict[graph_name][config_model_name] = {}
                remove_nodes_per, lcc_size_list = get_dismantl_curves_data(remove_nodes=remove_nodes, network=G)
                config_record_dict[graph_name][config_model_name]["remove_nodes_per"] = remove_nodes_per
                config_record_dict[graph_name][config_model_name]["lcc_size_list"] = lcc_size_list
                ax.plot(remove_nodes_per, lcc_size_list,
                        color="lightskyblue",
                        linestyle="-",
                        )

        ax.set_xlabel("percentage of removed nodes")
        ax.set_ylabel("LCC size")
        if title == True:
            ax.set_title("{}\n Assortativity: {} ".format(graph_name,round(assortativity,4)))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(linestyle="--")
        subplot_iter += 1
    if not real_record or not real_record_method:
        with open(real_file_path + '/dismantle_curve_record_target_{}_{}.json'.format(target,Rein), 'w') as f:
            json.dump(real_record_dict, f, indent=4, separators=(',', ':'))
    if not config_record:
        with open(config_file_path + "/dismantle_curve_record_target_{}_{}_{}.json".format(target, Rein,old_method), 'w') as f:
            json.dump(config_record_dict, f, indent=4, separators=(',', ':'))
    #plt.savefig("{}.png".format(method))
    plt.show()

def analysis_dismantle_graph_methods(best_method,target,Rein,method_list,file_path):

    with open(file_path + '/graphs_dismantle_size_target_{}_{}.json'.format(target, Rein), "r") as f:
        graphs_dismantle_size = json.load(f)


    graphs_name = list(graphs_dismantle_size.keys())
    for graph_name in graphs_name:
        best_method_graph_size = graphs_dismantle_size[graph_name][best_method]
        graphs_save=True
        for method in method_list:

            method_graph_size = graphs_dismantle_size[graph_name][method]
            if best_method_graph_size > method_graph_size:
                graphs_save=False
        if  graphs_save:
            print(graph_name)


if __name__=="__main__":
    target = 20
    Rein=True
    if Rein:
        # method_list = ["MinSum", "GND", "BPD", "CoreHD", "stable_GDM", "GDM"]
        # method_list = ["MinSum", "GND", "BPD", "CoreHD", "stable_GDM"]
         method_list = ["CI","MinSum", "GND",  "stable_GDM", "GDM"]
      #   method_list=["random_rein"]
        # method_list = ["MinSum", "GND", "stable_GDM"]
       # method_list = ["stable_GDM", "GDM"]
    else:
        # method_list = ["MinSum", "GND", "CoreHD", "stable_GDM", "GDM"]
        # method_list = ["MinSum", "GND","stable_GDM", "GDM"]
         method_list = ["random"]
       # method_list = ["stable_GDM", "GDM"]
    draw_statistics_methods(target=target,Rein=Rein,method_list=method_list,file_path='../result/dataset_real',vision_mode="table")


