import json
import math
import os.path
import time
import warnings
from glob import glob
import yaml
import random
import sys
import igraph
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from analysis.visual_tool.LCC_size_removed_nodes import draw_robustness_curves, draw_remove_statistics
import itertools as it
import torch

from analysis.rank import nodereinsert, modelGym_diffusion_evaluation, break_rein
from utils.utils import save_checkpoint, load_diffusion_model


def seed_torch(seed=1029):
    random.seed(seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)   # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True   # 选择确定性算法

def dismantle_methods(target,graph_name, G, models_path,
                      graphs_dismantle_time, graphs_dismantle_size, graphs_dismantle_nodelist, nep=False):
    dismantle_time = {}
    dismantle_size = {}
    dismantle_list = {}


        # print(models_path)
    filelist = os.listdir(models_path)
    print(filelist)
    if "configs_record_scores.json" in filelist:
        filelist.remove("configs_record_scores.json")
    filelist.sort(key=lambda x: int(x[7:]))
 #   for i in range(11,383):
    for method in filelist:
      #  method="config_{}".format(i)
        if ".json" in method:
            continue
        print("---------methods_name-----------------")

        print(method)
        searched_method = method.replace("config_", "")
        if graphs_dismantle_size.get(graph_name) is not None:
            if graphs_dismantle_size[graph_name].get(searched_method) is not None:
                print("the result {} of {} already exists".format(method, graph_name))
                dismantle_list[searched_method] = graphs_dismantle_nodelist[graph_name][searched_method]
                dismantle_time[searched_method] = graphs_dismantle_time[graph_name][searched_method]
                dismantle_size[searched_method] = graphs_dismantle_size[graph_name][searched_method]
                continue

        seed_torch(123)
        remove_list, cost_time = modelGym_diffusion_evaluation(graph_name,G=G, method_name=method,
                                                               target=target, model_path=models_path)  #
        if nep == True:
            remove_list, _ = nodereinsert(G=G, remove_list=remove_list)
        method = method.replace("config_", "")
        dismantle_list[method] = remove_list
        dismantle_time[method] = cost_time
        dismantle_size[method] = len(remove_list) / nx.number_of_nodes(G)
    graphs_dismantle_time[graph_name] = dismantle_time
    graphs_dismantle_size[graph_name] = dismantle_size
    graphs_dismantle_nodelist[graph_name] = dismantle_list

def dismantle_statistics_methods(target,models_path="results/ND_pre_trainl/grid_search",dataset_path="analysis/analysis_dataset/dataset_real",
                                 result_save_path="analysis/result/dataset_real",filter="*",nep=False,graph_name_set=False,infer_param=None):
    graphs_name = []
    graphs_dismantle_time = {}
    graphs_dismantle_size = {}
    graphs_dismantle_nodelist = {}
    graph_max_nodes = sys.maxsize
    if os.path.exists(result_save_path):
        print("waring:result_save_path文件夹目录已存在")
    else:
        os.mkdir(result_save_path)
    if os.path.exists(result_save_path + '/graphs_dismantle_size.json'):
        with open(result_save_path + '/graphs_dismantle_size.json', 'r') as f:
            graphs_dismantle_size = json.load(f)
        with open(result_save_path + '/graphs_dismantle_time.json', 'r') as f:
            graphs_dismantle_time = json.load(f)
        with open(result_save_path + '/graphs_dismantle_nodelist.json', 'r') as f:
            graphs_dismantle_nodelist = json.load(f)

    for file in sorted(glob(dataset_path + '/' + filter + ".graphml")):
        g = igraph.Graph.Read_GraphML(file)
        G = g.to_networkx()

        G.remove_edges_from(nx.selfloop_edges((G)))  # 去自环
        G.remove_nodes_from(list(nx.isolates(G)))  # 去孤立节点
        G = nx.convert_node_labels_to_integers(G)#将节点编号从0开始,重新编号

        if G.number_of_nodes() <= graph_max_nodes:
            graph_name = file[len(dataset_path) + 1:].replace(".graphml", "")
            graphs_name.append(graph_name)
            print("-----------networks------------")
            print("{}_size:{}".format(graph_name, G.number_of_nodes()))

            dismantle_methods(target,graph_name, G,models_path,
                              graphs_dismantle_time, graphs_dismantle_size, graphs_dismantle_nodelist, nep=nep)


        with open(result_save_path + '/graphs_dismantle_size.json', 'w') as f:
            json.dump(graphs_dismantle_size, f, indent=4, separators=(',', ':'))
        with open(result_save_path + '/graphs_dismantle_time.json', 'w') as f:
            json.dump(graphs_dismantle_time, f, indent=4, separators=(',', ':'))
        with open(result_save_path + '/graphs_dismantle_nodelist.json', 'w') as f:
            json.dump(graphs_dismantle_nodelist, f, indent=4, separators=(',', ':'))

def decycling_methods(graph_name, G, models_path, graphs_decycling_time,
                      graphs_decycling_size, graphs_decycling_nodelist, nep=False,
                      ):
    decycling_time = {}
    decycling_size = {}
    decycling_list = {}

    #print(models_path)
    filelist=os.listdir(models_path)
    print(filelist)
    if "configs_record_scores.json" in filelist:
        filelist.remove("configs_record_scores.json")
    filelist.sort(key=lambda x:int(x[7:]))
   # # for i in range(134,192):
    for method in filelist:
        if ".json" in method:
            continue
        print("---------methods_name-----------------")
       # method = "config_" + str(i)
        print(method)
        searched_method = method.replace("config_", "")
        if graphs_decycling_size.get(graph_name) is not None:
            if graphs_decycling_size[graph_name].get(searched_method) is not None:
                print("the result {} of {} already exists".format(method,graph_name))
                decycling_list[searched_method] = graphs_decycling_nodelist[graph_name][searched_method]
                decycling_time[searched_method] = graphs_decycling_time[graph_name][searched_method]
                decycling_size[searched_method] = graphs_decycling_size[graph_name][searched_method]
                continue
        seed_torch(123)

        remove_list, cost_time = modelGym_diffusion_evaluation(graph_name,G, method_name=method,
                                                               target="decycling",model_path=models_path)  #
        if nep == True:
            remove_list, _ = nodereinsert(G=G, remove_list=remove_list)

        decycling_list[searched_method] = remove_list
        decycling_time[searched_method] = cost_time
        decycling_size[searched_method] = len(remove_list) / nx.number_of_nodes(G)
    graphs_decycling_time[graph_name] = decycling_time
    graphs_decycling_size[graph_name] = decycling_size
    graphs_decycling_nodelist[graph_name] = decycling_list
def decycling_statistics_methods(models_path="results/ND_pre_trainl/grid_search",
                                 dataset_path="analysis/analysis_dataset/dataset_real",
                                 result_save_path="analysis/result/dataset_real",filter="*",nep=False,
                                 graph_name_set=False):

    graphs_decycling_time = {}
    graphs_decycling_size = {}
    graphs_decycling_nodelist = {}
    graph_max_nodes = sys.maxsize
    if os.path.exists(result_save_path):
        print("waring:result_save_path文件夹目录已存在")
    else:
        os.mkdir(result_save_path)
    if os.path.exists(result_save_path + '/graphs_decycling_size.json'):
        with open(result_save_path + '/graphs_decycling_size.json', 'r') as f:
            graphs_decycling_size = json.load(f)
        with open(result_save_path + '/graphs_decycling_time.json', 'r') as f:
            graphs_decycling_time = json.load(f)
        with open(result_save_path + '/graphs_decycling_nodelist.json', 'r') as f:
            graphs_decycling_nodelist = json.load(f)
    for file in sorted(glob(dataset_path + '/' + filter + ".graphml")):
        graph_name = file[len(dataset_path) + 1:].replace(".graphml", "")
        print("-----------networks------------")
        print(graph_name)


        g = igraph.Graph.Read_GraphML(file)
        G = g.to_networkx()

        G.remove_edges_from(nx.selfloop_edges((G)))  # 去自环
        G.remove_nodes_from(list(nx.isolates(G)))  # 去孤立节点
        G = nx.convert_node_labels_to_integers(G)  # 将节点编号从0开始,重新编号
        if G.number_of_nodes() <= graph_max_nodes:

         #   graphs_name.append(graph_name)
           # print("{}_size:{}".format(graph_name, G.number_of_nodes()))

            decycling_methods(graph_name, G,models_path,
                              graphs_decycling_time, graphs_decycling_size, graphs_decycling_nodelist,
                              nep=nep)



        with open(result_save_path + '/graphs_decycling_size.json', 'w') as f:
            json.dump(graphs_decycling_size, f, indent=4, separators=(',', ':'))
        with open(result_save_path + '/graphs_decycling_time.json', 'w') as f:
            json.dump(graphs_decycling_time, f, indent=4, separators=(',', ':'))
        with open(result_save_path + '/graphs_decycling_nodelist.json', 'w') as f:
            json.dump(graphs_decycling_nodelist, f, indent=4, separators=(',', ':'))

def draw_statistics_methods(target,file_path="result",filter_graphs=None):
    #
    if filter_graphs is None:
        filter_graphs=[]
    if target=="dismantling":
        with open(file_path+"/graphs_dismantle_size.json","r") as f:
            graphs_reomve_size=json.load(f)
        fig = plt.figure(num="dismantle complex systems -dismantle_size 测试")
        # ax = fig.add_subplot(122)
    elif target=="decycling":
        with open(file_path+"/graphs_decycling_size.json","r") as f:
            graphs_reomve_size=json.load(f)
        fig = plt.figure(num="decycling complex systems -dismantle_size 测试")
    graphs_name = list(graphs_reomve_size.keys())
    filter_graphs_reomve_size={}
    for graph_name in graphs_name:
        if graph_name in filter_graphs:
            continue
        filter_graphs_reomve_size[graph_name]=graphs_reomve_size[graph_name]
    draw_remove_statistics(fig, filter_graphs_reomve_size)

    plt.show()

def filter_graphs_and_statistics(target,filter_target,filter_graphs=None,file_path="result",models_grid=None):
    if filter_graphs is None:
        filter_graphs=[]
    with open(file_path+"/graphs_{}_size.json".format(target),"r") as f:
        graphs_dismantle_size=json.load(f)
    graphs_name = list(graphs_dismantle_size.keys())
    methods_list = list(list(graphs_dismantle_size.values())[2].keys())
    methods_all_dismantle_size={}
    for method in methods_list:
        method_all_dismantle_size=np.array([
            graphs_dismantle_size[graph_name].get(method)
            for graph_name in graphs_name
            if graph_name not in filter_graphs
        ]    )
        methods_all_dismantle_size[method]=method_all_dismantle_size.mean()*100


    with open(file_path + '/graphs_all_{}_size.json'.format(target), 'w') as f:
        json.dump(methods_all_dismantle_size, f, indent=4, separators=(',', ':'))


    count=0
    dismantle_size_list=[]
    for i,dismantle_size in methods_all_dismantle_size.items():
        dismantle_size_list.append(dismantle_size)
        if dismantle_size >filter_target:
            continue
        print("________________________________________________________")
        print("methods_name:{} dismantle size:{} ".format(i,dismantle_size))
        count+=1
    print("less than target_{}:number:{}".format(filter_target,count))
    fig,ax=plt.subplots(figsize=(10, 5), num=models_grid)
    ax.boxplot(dismantle_size_list)
    plt.xticks([1], [models_grid])
    plt.show()


def grid_infer_params_model_saved(models_path, grid_infer_params, device):

    filelist = os.listdir(models_path)
    new_models_path = models_path.replace("grid_search", "grid_infer")
    if not os.path.exists(new_models_path):
        os.mkdir(new_models_path)
    else:
        warnings.warn("{}文件夹目录已存在".format(new_models_path))
        warnings.warn("code execution exits early")
        return
    print(filelist)
    if "configs_record_scores.json" in filelist:
        filelist.remove("configs_record_scores.json")
    filelist.sort(key=lambda x: int(x[7:]))
    models_count = len(filelist)
    infer_param_count = 0
    for params_values in it.product(*grid_infer_params.values()):
        infer_param = dict(zip(grid_infer_params.keys(), params_values))
        for method_name in filelist:
            method_path = models_path + "/" + method_name
            method_model_path = method_path + "/hold_out/trained_model/NDlearner_diffusion_trained"
            method_config_path = method_path + "/{}.yaml".format(method_name)
            with open(method_config_path, 'r') as f:
                method_config_param = yaml.safe_load(f)
            method_config_param["net"].update(infer_param)
            loaded_model, loaded_net_param = load_diffusion_model(path=method_model_path, device=device, infer_para=infer_param)

            searched_method_count = method_name.replace("config_", "")
            new_method_count = int(searched_method_count) + infer_param_count * models_count
            new_method_name = method_name.replace(searched_method_count, str(new_method_count))
            new_method_path = new_models_path + "/" + new_method_name
            if os.path.exists(new_method_path):
                print("{}文件夹目录已存在".format(new_method_path))
            else:
                os.mkdir(new_method_path)

            new_method_model_path = new_method_path + "/hold_out/trained_model"
            if os.path.exists(new_method_model_path):
                print("{}文件夹目录已存在".format(new_method_model_path))
            else:
                try:
                    os.makedirs(new_method_model_path, exist_ok=True)
                    print(f"Folders '{new_method_model_path}' created successfully.")
                except OSError as e:
                    print(f"Error: {e.strerror}")
            save_checkpoint(path=new_method_model_path, model_name="NDlearner_diffusion_trained", model=loaded_model,
                            net_param=loaded_net_param)
            new_method_config_path = new_method_path + "/{}.yaml".format(new_method_name)
            with open(new_method_config_path, "w") as f:
                yaml.dump(method_config_param, f)
        infer_param_count += 1
def delete_dataset_subfolder(folder_path):
    import shutil
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        print("Checking subfolder: ", subfolder_path)
        if os.path.isdir(subfolder_path):
            print("Deleting subfolder: ", subfolder_path)
            shutil.rmtree(subfolder_path)
if __name__=="__main__":
    """
    # 
    """
    seed_torch(123)
    models_grid = "trained_model_new_FastT2T2"#trained_model_new_T2T
    models_path = "results/{}/grid_search".format(models_grid)
    # models_path = "results/saved_model/{}/grid_search".format(models_grid)
    dataset_path = "analysis/analysis_dataset/dataset_real"
    #
    # filter_graphs=["cfinder-google"]
    filter_graphs = []
    target="decycling"
    #target = "dismantling"
    file_path = "analysis/result/{}_{}".format(models_grid, target)
    # file_path="analysis/result/trained_model_ND_t2t_new2_decycling"
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    if "_categorical" in models_grid:
        grid_infer_params = {
            "gradient_estimator_type": ["cost_estimator"],
            "target_weight":[0.1,1,10],# [0.1,1,10],
             "parallel_sampling": [2,4],
            "rewrite": [False, True],
            "rewrite_steps": [1,3,5],
            "rewrite_ratio": [0.1,0.5,0.9],
            "inference_steps": [10,20],

        }
    elif "_gaussian" in models_grid:
        grid_infer_params = {
            "gradient_estimator_type": ["cost_estimator"],#"CEP_estimator","diffusion_evolution_estimator"
            "target_weight":[0.1,1,10],# [0.1,1,10],
             "parallel_sampling": [2,4],
            "rewrite": [False, True],
            "rewrite_steps": [1,3,5],
            "rewrite_ratio": [0.1,0.5,0.9],
            "inference_steps": [10,20],
            "predictor_type": ["discretize_reverse_diffusion","diffusion_sampling"] , # "diffusion_sampling"
            "corrector_type": ["annealed_langevin_dynamics", "no_corrector"],
            "corrector_steps": [2,4,6]
        }
    elif "_discretFlow" in models_grid:
        grid_infer_params = {
            "gradient_estimator_type": ["cost_estimator"],
            "target_weight":[0.1,1,10],# [0.1,1,10],
             "parallel_sampling": [2,4],
            "rewrite": [False, True],
            "rewrite_steps": [1,3,5],
            "rewrite_ratio": [0.1,0.5,0.9],
            "inference_steps": [10,20],
            "use_tag": [True,False]
        }
    else:
        raise ValueError("models_grid not found")
    delete_dataset_subfolder(folder_path=dataset_path)
    grid_infer_params_model_saved(models_path, grid_infer_params, device="cuda")

    infer_models_path = "results/{}/grid_infer".format(models_grid)
    if target == "decycling":
        decycling_statistics_methods(models_path=infer_models_path, filter="*", dataset_path=dataset_path,
                                     result_save_path=file_path)
    elif target == "dismantling":
        dismantle_statistics_methods(target=1, models_path=infer_models_path, filter="*", dataset_path=dataset_path,
                                    result_save_path=file_path)
    # draw_statistics_methods(target,file_path,filter_graphs=filter_graphs) #decycle minsum34.38 bpd 34.51 CoreHD 34.61 %
    filter_graphs_and_statistics(target=target, filter_target=3.19,  file_path=file_path)  #  dismantle: minsum 36.28 coreHD 39.02 bpd 35.09 GDM 40.15 finder 37.25 CI 52.33
    #










