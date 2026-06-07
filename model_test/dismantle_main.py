import math
from glob import glob

import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse
import torch.linalg

from analysis_visi.LCC_size_removed_nodes import draw_runtime_methods_synth_curve, draw_memory_methods_synth_curve, \
    draw_dismantle_methods_synth_curve
from analysis_visi.analysis_vision_utils import draw_LCCcurve_graphs, dismantle_statistics_methods, \
    draw_statistics_methods, draw_threshold_dismantlesize_curve, draw_statistics_properties_methods, \
    dismantle_graph_properties_ROC_methods, create_performance_table

from method.method import nodereinsert

from utils.utils import *


def file_graph_filter_setup(dataset_path,graph_filter,file_filters):
    for file_filter in file_filters:
        for file in sorted(glob(dataset_path + '/' + file_filter + ".graphml")):
            graph_name = file[len(dataset_path) + 1:].replace(".graphml", "")
            graph_filter.append(graph_name)


#网络拆解性能 算法比较
def dismantle_statistic_experiment(target,Refine,method_list,experiment_name,vision_mode="curve",dataset_type="real",nep=False,data_cover=False,DiffDGM_infer_param=None,Visual_presentation=True):

    dataset_path = "dataset/dataset_{}".format(dataset_type)

    #graph_filter = []
    # file_filter = ["ws_*", ]
    # #file_filter = ["ws_*","gnp_random_graph_n_1000_*","static_power_law_n_1000_*","barabasi_n_1000_*"]
    # file_graph_filter_setup(dataset_path=dataset_path,file_filters=file_filter,graph_filter=graph_filter)

    graph_filter=[
                   "advogato",
                  "maayan-Stelzl",
                  "foodweb-baydry",
                  "opsahl-ucsocial",
                  "foodweb-baywet"
                  ]#,
   # graph_filter = None


    file_path=os.path.join("result/{}".format(experiment_name),"dismantle_dataset_{}".format(dataset_type))
    save_path = os.path.join("result/{}".format(experiment_name), "dismantle_{}_new_method".format(dataset_type))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        if os.path.exists(save_path + '/graphs_dismantle_nodelist_target_{}_{}.json'.format(target, Refine)):
            dismantle_statistics_data_merge(target, Refine, method_list=method_list, file1_path=save_path,
                                            file2_path=file_path,
                                            save_path=file_path)
    if not Visual_presentation:

        if data_cover == False:


            if os.path.exists(
                    file_path + '/graphs_dismantle_size_target_{}_{}.json'.format(target, Refine)) :
                with open(file_path + '/graphs_dismantle_size_target_{}_{}.json'.format(target, Refine), 'r') as f:
                    graphs_dismantle_size = json.load(f)
                with open(file_path + '/graphs_dismantle_time_target_{}_{}.json'.format(target, Refine), 'r') as f:
                    graphs_dismantle_time = json.load(f)
                with open(file_path + '/graphs_dismantle_nodelist_target_{}_{}.json'.format(target, Refine), 'r') as f:
                    graphs_dismantle_nodelist = json.load(f)
            else:
                graphs_dismantle_time = {}
                graphs_dismantle_size = {}
                graphs_dismantle_nodelist = {}
        else:
            graphs_dismantle_time = {}
            graphs_dismantle_size = {}
            graphs_dismantle_nodelist = {}

        dismantle_statistics_methods(graphs_dismantle_time=graphs_dismantle_time,
                                     graphs_dismantle_size=graphs_dismantle_size,
                                     graphs_dismantle_nodelist=graphs_dismantle_nodelist,
                                     target=target, methods=method_list,
                                     dataset_path=dataset_path, save_path=save_path,
                                     Rein=Refine, DiffDGM_infer_param=DiffDGM_infer_param)
        # 主算法拆解数据与已保存的其他算法网络拆解数据合并，以便分析比较
        dismantle_statistics_data_merge(target, Refine, method_list=method_list, file1_path=save_path,
                                        file2_path=file_path,
                                        save_path=file_path)
    else:
    #可视化 算法拆解性能比较结果
        with open(file_path + '/graphs_dismantle_size_target_{}_{}.json'.format(target, Refine), 'r') as f:
            graphs_dismantle_size = json.load(f)
        draw_statistics_methods(graphs_dismantle_size=graphs_dismantle_size,method_list=method_list,
                                vision_mode=vision_mode,dataset_type=dataset_type,
                                graph_filter=graph_filter)


#综合评估在各算法的不同拆解阈值下的网络结构属性 分析实验
def evaluate_method_robustness_on_demand_experiment(graph_property,method_list,experiment_name,vision_mode="curve",dataset_type="real",data_cover=False,DiffDGM_infer_param=None,Visual_presentation=True):

    dataset_path = "dataset/dataset_{}".format(dataset_type)
    target=0
    Refine=False


    graph_filter=None


    file_path=os.path.join("result/{}".format(experiment_name),"dismantle_dataset_{}".format(dataset_type))
    if not Visual_presentation:
        save_path = os.path.join("result/{}".format(experiment_name), "dismantle_{}_new_method".format(dataset_type))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            if os.path.exists(save_path+ '/graphs_dismantle_nodelist_target_{}_{}.json'.format(target, Refine))  :
                dismantle_statistics_data_merge(target, Refine, method_list=method_list, file1_path=save_path,
                                                file2_path=file_path,
                                                save_path=file_path)
            if os.path.exists(save_path + '/graphs_properties_{}_nodelist.json'.format(graph_property)):
                graph_properties_dismantle_data_merge(graph_property, method_list, file1_path=save_path,
                                                      file2_path=file_path,
                                                      save_path=file_path)
        if data_cover == False:


            if os.path.exists(
                    file_path + '/graphs_properties_{}_nodelist.json'.format(graph_property)) :
                with open(file_path + '/graphs_properties_{}_nodelist.json'.format(graph_property), 'r') as f:
                    graphs_properties_nodelist = json.load(f)
                with open(file_path + '/graphs_properties_{}_AUC.json'.format(graph_property), 'r') as f:
                    graphs_properties_AUC = json.load(f)
            else:
                graphs_properties_nodelist = {}
                graphs_properties_AUC = {}
            if os.path.exists(file_path + '/graphs_dismantle_nodelist_target_{}_{}.json'.format(target, Refine)):
                with open(file_path + '/graphs_dismantle_time_target_{}_{}.json'.format(target, Refine), 'r') as f:
                    graphs_dismantle_time = json.load(f)
                with open(file_path + '/graphs_dismantle_nodelist_target_{}_{}.json'.format(target, Refine), 'r') as f:
                    graphs_dismantle_nodelist = json.load(f)
                with open(file_path + '/graphs_dismantle_size_target_{}_{}.json'.format(target, Refine), 'r') as f:
                    graphs_dismantle_size = json.load(f)
            else:
                graphs_dismantle_time = {}
                graphs_dismantle_nodelist = {}
                graphs_dismantle_size = {}
        else:
            graphs_dismantle_time = {}
            graphs_properties_nodelist = {}
            graphs_dismantle_nodelist = {}
            graphs_dismantle_size = {}
            graphs_properties_AUC = {}

        dismantle_graph_properties_ROC_methods(graph_property=graph_property,
                                        graphs_properties_nodelist=graphs_properties_nodelist,graphs_properties_AUC=graphs_properties_AUC,
                                    graphs_dismantle_time=graphs_dismantle_time,
                                     graphs_dismantle_size=graphs_dismantle_size,
                                     graphs_dismantle_nodelist=graphs_dismantle_nodelist,
                                     target=target, methods=method_list,
                                     dataset_path=dataset_path, save_path=save_path,
                                     Rein=Refine, DiffDGM_infer_param=DiffDGM_infer_param)

        # 主算法拆解数据与已保存的其他算法网络拆解数据合并，以便分析比较 未改
        dismantle_statistics_data_merge(target, Refine, method_list=method_list, file1_path=save_path,
                                        file2_path=file_path,
                                        save_path=file_path)
        graph_properties_dismantle_data_merge(graph_property, method_list, file1_path=save_path, file2_path=file_path,
                                              save_path=file_path)

    else:
    #可视化 算法拆解性能比较结果
        with open(file_path+'/graphs_properties_{}_AUC.json'.format(graph_property),"r") as f:
            graphs_properties_AUC = json.load(f)
        draw_statistics_properties_methods(method_list, graphs_properties_AUC,graph_property=graph_property,
                                           vision_mode=vision_mode, dataset_type=dataset_type,

                                           filter=graph_filter)




def graph_size_compute_size():
    dataset_type="synth_size"
    dismantle_threshold=1
    Refine=True
    file_path = "result/DiffGDM/dismantle_dataset_{}".format(dataset_type)
    with open(file_path + '/graphs_dismantle_size_target_{}_{}.json'.format(dismantle_threshold, Refine), 'r') as f:
        graphs_dismantle_size = json.load(f)
    with open(file_path + '/graphs_dismantle_time_target_{}_{}.json'.format(dismantle_threshold, Refine), 'r') as f:
        graphs_dismantle_time = json.load(f)
    with open(file_path + '/graphs_dismantle_memory_target_{}_{}.json'.format(dismantle_threshold, Refine), 'r') as f:
        graphs_dismantle_memory = json.load(f)

    #create_performance_table(graphs_dismantle_size, output_file=file_path+'/method_performance_sizes.xlsx')
    #create_performance_table(graphs_dismantle_time, output_file=file_path+'/method_performance_times.xlsx')
    plt.rc('font', family='Arial', size=10)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    draw_runtime_methods_synth_curve(axs[0], graphs_dismantle_time)
    # axs_twinx=axs[0].twinx()
    # draw_dismantle_methods_synth_curve(axs_twinx, graphs_dismantle_size)
    axs[0].text(-0.1, 1.05, "a", transform=axs[0].transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')
    draw_memory_methods_synth_curve(axs[1], graphs_dismantle_memory)
    axs[1].text(-0.1, 1.05, "b", transform=axs[1].transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')
    plt.tight_layout()
    plt.show()
    plt.savefig(file_path+'/method_performance_time_memory.png')



if __name__=="__main__":
    """
   
    """
    nep = False  # 节点爆炸渗流精细化节点拆解顺序
    experiment_name="DiffGDM"
    dataset_type = "real_all"
    Refine= True  # 网络节点重插
    data_cover = True  # 数据覆盖模式

    """实验一 网络拆解性能 算法比较"""
    dismantle_threshold =1 # 拆解阈值 （%）
    method_list = [
        "BPD","CoreDiffGDM","CoreDiffGDM_base"

]

    DiffDGM_infer_param = {
        "gradient_estimator_type": "cost_estimator",
        "guided_model_path": "method/modelGym/Diffusion_model/robustness_metrics_models",
        "use_tag": True,
        "parallel_sampling": 4,
    }
    # dismantle_statistic_experiment(dismantle_threshold, Refine, method_list, experiment_name, vision_mode="table",
    #                                dataset_type=dataset_type,
    #                                nep=nep, data_cover=data_cover, DiffDGM_infer_param=DiffDGM_infer_param,
    #                                Visual_presentation=False)
    """实验2 统计模型尺寸对算法性能以及运行时间，内存的影响"""

    graph_size_compute_size()

    """实验三 各算法删除节点序列对网络性质的影响比较 AUC """
    # method_list = ["NIRM","GDM","HD","HB","HSubgraph","DiffGDM_D","DiffGDM_D_no_tag_natural_connectivity"
    #                ]
    # method_list = ["NIRM","GDM","HD","HB","DiffGDM_D","DiffGDM_D_no_tag_LCC"
    #                ]
    # method_list = ["NIRM","GDM","HD","HB","HCluster","DiffGDM_D","DiffGDM_D_no_tag_global_CC", "DiffGDM_D_no_tag_average_CC"
    #                ]
    # method_list = ["NIRM","GDM","HD","HB","HClose","DiffGDM_D","DiffGDM_D_no_tag_global_efficiency"
    #                ]
    # # method_list = ["NIRM", "GDM", "HD", "HB", "HSubgraph", "DiffGDM_D", "DiffGDM_D_no_tag_natural_connectivity"
    # #                ]
    # graph_properties=["global_efficiency"]#"LCC","global_CC","average_CC","global_efficiency", natural_connectivity
    # #
    # #
    # for graph_property in graph_properties:
    #     print("graph_property:", graph_property)
    #     evaluate_method_robustness_on_demand_experiment(graph_property, method_list, experiment_name,
    #                                                     vision_mode="statistics", dataset_type=dataset_type,
    #                                                     data_cover=data_cover, DiffDGM_infer_param=None,
    #                                                     Visual_presentation=True
    #                                                     )







