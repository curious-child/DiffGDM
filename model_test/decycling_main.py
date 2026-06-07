import json
import os
from glob import glob
import networkx as nx
import igraph

from analysis_visi.analysis_vision_utils import decycle_statistics_methods, draw_statistics_methods
from utils.utils import decycle_statistics_data_merge, seed_torch


def DiffDGM_infer_param_set(method_list,gradient_estimator_type,infer_direction):
    common_direction=["LCC","global_CC","average_CC","natural_connectivity","global_efficiency","density"]
    cost_specific_direction=["MVC"]
    modelgym_methods = ["DiffGDM_C", "DiffGDM_D", "DiffGDM_DD"
                        ]
    new_method_list=method_list.copy()
    DiffDGM_methods=set(new_method_list).intersection(set(modelgym_methods))

    if gradient_estimator_type=="cost_estimator":
        if infer_direction in common_direction:
            DiffDGM_infer_param={
            "gradient_estimator_type": "cost_estimator",
            "infer_direction": infer_direction , # ,]#
            "guided_model_path": "method/modelGym/Diffusion_model/robustness_metrics_models",
                "use_tag": True,
            }

        elif infer_direction in cost_specific_direction:
            DiffDGM_infer_param = {
                "gradient_estimator_type": "cost_estimator",
                "infer_direction": infer_direction,
                "use_tag": True,
                }
        else:
            raise ValueError("infer_direction not in common_direction or cost_specific_direction")
        DiffDGM_infer_param.update(
            {"parallel_sampling": 4,

             }
        )
    elif gradient_estimator_type=="CEP_estimator":
        if infer_direction in common_direction:
            DiffDGM_infer_param = {
                "gradient_estimator_type": "diffusion_evolution_estimator",
                "infer_direction": infer_direction,
                "guided_model_path": "method/modelGym/Diffusion_model/robustness_metrics_models_cep",
                "use_tag": True,
                 }
        else:
            raise ValueError("infer_direction not in common_direction")
        DiffDGM_infer_param.update(
            {"parallel_sampling": 4, }
        )
    elif gradient_estimator_type=="no_tag":
        if infer_direction in common_direction:
            DiffDGM_infer_param = {
                "gradient_estimator_type": "no_tag",
                "infer_direction": infer_direction,
                "use_tag": False,
            }
        else:
            raise ValueError("infer_direction not in common_direction")
        DiffDGM_infer_param.update(
            {"parallel_sampling": 4, }
        )
    else:
        raise ValueError("gradient_estimator_type not in cost_estimator,CEP_estimator")
    if DiffDGM_methods:
        for DiffDGM_method in DiffDGM_methods:
            new_method_list.remove(DiffDGM_method)
            new_method_name = DiffDGM_method + "_{}_{}".format(gradient_estimator_type, infer_direction)

            new_method_list.append(new_method_name)

    return DiffDGM_infer_param,new_method_list

def decycle_statistic_experiment(target,method_list,experiment_name,vision_mode="curve",dataset_type="real",DiffDGM_infer_param=None,data_cover=False):

    dataset_path = "dataset/dataset_{}".format(dataset_type)

    #graph_filter = []
    # file_filter = ["ws_*", ]
    # #file_filter = ["ws_*","gnp_random_graph_n_1000_*","static_power_law_n_1000_*","barabasi_n_1000_*"]
    # file_graph_filter_setup(dataset_path=dataset_path,file_filters=file_filter,graph_filter=graph_filter)

    # graph_filter=["advogato","cfinder-google","econ-wm1","ego-twitter","foodweb-baydry","internet-topology",
    #               "linux","maayan-Stelzl","opsahl-openflights","opsahl-powergrid","oregon2_010526","petster-hamster","subelj_jung-j_jung-j","web-webbase-2001"]

    # graph_filter = [
    #                 "advogato",
    #               "maayan-Stelzl",
    #               "foodweb-baydry",
    #               "opsahl-ucsocial",
    #               "foodweb-baywet"
    #               ]
    graph_filter = None
    file_path=os.path.join("result/{}".format(experiment_name),"decycle_dataset_{}".format(dataset_type))

    save_path = os.path.join("result/{}".format(experiment_name), "decycle_{}_new_method".format(dataset_type))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        # 主算法拆解数据与已保存的其他算法网络拆解数据合并，以便分析比较
        if os.path.exists(save_path + "/graphs_decycle_nodelist.json"):
            print("merge data")
            decycle_statistics_data_merge( file1_path=save_path,
                                          file2_path=file_path,
                                          save_path=file_path)

    if data_cover == False:

        if os.path.exists(
                file_path + '/graphs_decycle_size.json'):
            with open(file_path + '/graphs_decycle_size.json', 'r') as f:
                graphs_dismantle_size = json.load(f)
            with open(file_path + "/graphs_decycle_time.json", 'r') as f:
                graphs_dismantle_time = json.load(f)
            with open(file_path + "/graphs_decycle_nodelist.json", 'r') as f:
                graphs_dismantle_nodelist = json.load(f)
    else:
        graphs_dismantle_time = {}
        graphs_dismantle_size = {}
        graphs_dismantle_nodelist = {}
    decycle_statistics_methods(methods=method_list,
                               graphs_dismantle_size=graphs_dismantle_size,
                               graphs_dismantle_time=graphs_dismantle_time,
                               graphs_dismantle_nodelist=graphs_dismantle_nodelist,
                                 dataset_path=dataset_path, save_path=save_path,DiffDGM_infer_param=DiffDGM_infer_param
                                 )
    decycle_statistics_data_merge( file1_path=save_path,
                                  file2_path=file_path,
                                  save_path=file_path)


    #可视化 算法拆解性能比较结果
    with open(file_path + '/graphs_decycle_size.json', 'r') as f:
        graphs_dismantle_size = json.load(f)
    draw_statistics_methods(graphs_dismantle_size=graphs_dismantle_size,method_list=method_list,task_type="decycling",
                            vision_mode=vision_mode,dataset_type=dataset_type,graph_filter=graph_filter
                           )


def resave_dataset(dataset_type):
    file2_path = "result/DiffGDM/decycle_dataset_{}".format( dataset_type)
    save_path = "result"

    with open(file2_path + '/graphs_decycle_size.json', "r") as f:
        file2_dismantle_size = json.load(f)
    with open(file2_path + '/graphs_decycle_nodelist.json', "r") as f:
        file2_dismantle_nodelist = json.load(f)
    with open(file2_path + '/graphs_decycle_time.json', "r") as f:
        file2_dismantle_time = json.load(f)
    graph_max_nodes = 200000
    dataset_path = "dataset/dataset_{}".format(dataset_type)
    save_dismantle_size = {}
    save_dismantle_nodelist = {}
    save_dismantle_time = {}
    for file in sorted(glob(dataset_path + '/' + "*" + ".graphml")):
        g = igraph.Graph.Read_GraphML(file)
        G = g.to_networkx()
        #  print("G:", G)
        G.remove_edges_from(nx.selfloop_edges((G)))  # 去自环
        G.remove_nodes_from(list(nx.isolates(G)))  # 去孤立节点
        G = nx.convert_node_labels_to_integers(G)  # 将节点编号从0开始,重新编号
        G_nodes = G.number_of_nodes()
        if G_nodes <= graph_max_nodes:
            graph_name = file[len(dataset_path) + 1:].replace(".graphml", "")
            print("graph_name:", graph_name)
            print("G nodes", G_nodes)
            save_dismantle_size[graph_name] = file2_dismantle_size[graph_name]
            save_dismantle_time[graph_name] = file2_dismantle_time[graph_name]
            save_dismantle_nodelist[graph_name] = file2_dismantle_nodelist[graph_name]
    with open(save_path + '/graphs_decycle_size.json', 'w') as f:
        json.dump(save_dismantle_size, f, indent=4, separators=(',', ':'))
    with open(save_path + '/graphs_decycle_time.json', 'w') as f:
        json.dump(save_dismantle_time, f, indent=4, separators=(',', ':'))
    with open(save_path + '/graphs_decycle_nodelist.json', 'w') as f:
        json.dump(save_dismantle_nodelist, f, indent=4, separators=(',', ':'))
def resave_method(file2_path,method_list):

    save_path = "result"
    with open(file2_path + '/graphs_decycle_size.json', "r") as f:
        file2_dismantle_size = json.load(f)
    with open(file2_path + '/graphs_decycle_nodelist.json', "r") as f:
        file2_dismantle_nodelist = json.load(f)
    with open(file2_path + '/graphs_decycle_time.json', "r") as f:
        file2_dismantle_time = json.load(f)

    save_dismantle_size = {}
    save_dismantle_nodelist = {}
    save_dismantle_time = {}
    for graph_name in file2_dismantle_size.keys():
        save_dismantle_size[graph_name] = {}
        save_dismantle_time[graph_name] = {}
        save_dismantle_nodelist[graph_name] = {}
        for method in method_list:
            save_dismantle_size[graph_name][method] = file2_dismantle_size[graph_name][method]
            save_dismantle_time[graph_name][method] = file2_dismantle_time[graph_name][method]
            save_dismantle_nodelist[graph_name][method] = file2_dismantle_nodelist[graph_name][method]
    with open(save_path + '/graphs_decycle_size.json', 'w') as f:
        json.dump(save_dismantle_size, f, indent=4, separators=(',', ':'))
    with open(save_path + '/graphs_decycle_time.json', 'w') as f:
        json.dump(save_dismantle_time, f, indent=4, separators=(',', ':'))
    with open(save_path + '/graphs_decycle_nodelist.json', 'w') as f:
        json.dump(save_dismantle_nodelist, f, indent=4, separators=(',', ':'))
if __name__=="__main__":
    """
  
    
   
    """
    # method_list = [
    #     "MinSum",
    #     # "CoreDQN",
    #     "BPD","CoreHD",
    #      "DiffGDM_D",
    #    # "DiffGDM_D2",
    #                 "DiffGDM_D_cost_estimator_MVC",
    #                # "DiffGDM_D_cost_estimator_LCC",
    #                # "DiffGDM_D_cost_estimator_natural_connectivity",
    #                # "DiffGDM_D_cost_estimator_global_CC",
    #                # "DiffGDM_D_cost_estimator_average_CC",
    #                # "DiffGDM_D_cost_estimator_global_efficiency",
    #                # "DiffGDM_D_cost_estimator_density",
    #                ]#"MinSum", "BPD","CoreDQN","DiffGDM_C","DiffGDM_D","DiffGDM_DD" ,"DiffGDM_DD_cost_estimator_MVC"
    # new_method_list=["MinSum", "BPD","CoreHD","CoreDQN","DiffGDM_C",]
    experiment_name="DiffGDM"

    target="decycle"
    data_cover=False
    dataset_type="real_all"


    # DFM:"cost_estimator","no_tag"(直面不可微函数)
    #Catergorical:"cost_estimator"
    method_list=["DiffGDM"]
    DiffDGM_infer_param = {
        "gradient_estimator_type": "cost_estimator",
        "guided_model_path": "method/modelGym/Diffusion_model/robustness_metrics_models",
        "use_tag": True,
        "parallel_sampling": 4,
    }
    decycle_statistic_experiment(target, method_list, experiment_name, vision_mode="table",
                                 dataset_type=dataset_type, data_cover=data_cover,
                                 DiffDGM_infer_param=DiffDGM_infer_param)

    ##infer direction function

    # 共有："LCC","global_CC","average_CC","natural_connectivity","global_efficiency","density"
    # method_list = ["DiffGDM_D"]
    # infer_directions =["LCC"]#"natural_connectivity",,"natural_connectivity","global_CC","average_CC","global_efficiency","density"
    # gradient_estimator_type = "cost_estimator"
    # DiffDGM_infer_param = {"rewrite": True}  # 未做 True;
    # for infer_direction in infer_directions:
    #     if DiffDGM_infer_param["rewrite"]:
    #         new_DiffDGM_infer_param,new_method_list = DiffDGM_infer_param_set(method_list,gradient_estimator_type, infer_direction)
    #         DiffDGM_infer_param.update(new_DiffDGM_infer_param)
    #     else:
    #        # DiffDGM_infer_param = {"rewrite":False}
    #         new_method_list=method_list
    #
    #     decycle_statistic_experiment(target,new_method_list,experiment_name,vision_mode="table",dataset_type=dataset_type,data_cover=data_cover,
    #                                  DiffDGM_infer_param=DiffDGM_infer_param)
    # infer_direction = "MVC"


   # resave_dataset(dataset_type)
   # resave_method(file2_path="result/DiffGDM/decycle_dataset_real_all", method_list=method_list)
   #  file_path = "result/DiffGDM/decycle_dataset_real_all"
   #  save_path = "result/DiffGDM/decycle_real_all_new_method"
   #  decycle_statistics_data_merge(file1_path=save_path,
   #                                file2_path=file_path,
   #                                save_path=file_path)