import math
from glob import glob

import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse
import torch.linalg
from matplotlib.ticker import PercentFormatter

from analysis_visi.analysis_vision_utils import dismantle_methods
from method.method import nodereinsert, NodeBreakList

from utils.utils import *
import matplotlib


def file_graph_filter_setup(dataset_path,graph_filter,file_filters):
    for file_filter in file_filters:
        for file in sorted(glob(dataset_path + '/' + file_filter + ".graphml")):
            graph_name = file[len(dataset_path) + 1:].replace(".graphml", "")
            graph_filter.append(graph_name)

def dismantle_statistics_methods(target,method,dataset_path,Rein,
                                 graphs_dismantle_time,graphs_dismantle_size,graphs_dismantle_nodelist,
                                 save_path="result",filter="*",**kwargs):
    if os.path.exists(save_path):
        print("records_path:{}文件夹目录已存在".format(save_path))

    else:
        os.mkdir(save_path)

    graphs_name = []
    #graphs_name_filter = ["subelj_jdk_jdk"]
    graphs_name_filter = []
    graph_max_nodes = 200000  # 系统最大整数值 可视为Inf

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
            if graph_name in graphs_name_filter:
                continue
            graphs_name.append(graph_name)
            print("-----------networks------------")
            print("{}_size:{}".format(graph_name, G_nodes))
            if graphs_dismantle_nodelist.get(graph_name) is not None:

                print("the result {}  already exists".format( graph_name))

                continue
            remove_list, cost_time = NodeBreakList(G, method=method,
                                                   graph_name=graph_name,
                                                   target=target,
                                                   models_REIN=Rein,**kwargs)
            graphs_dismantle_time[graph_name] = cost_time
            graphs_dismantle_size[graph_name] = len(remove_list) / nx.number_of_nodes(G)
            graphs_dismantle_nodelist[graph_name] = remove_list

def dismantle_model_params_experiment(experiment_name,analysis_param,params_list):
    Refine = True  # 网络节点重插
    target = 1  # 拆解阈值 （%）
    method = "DiffGDM"  #
    dataset_type = "synth_large"
    dataset_path = "dataset/dataset_{}".format(dataset_type)
    save_path = os.path.join("result/{}".format(experiment_name), "dismantle_analysis_dataset_{}".format(dataset_type))

    #params_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    if os.path.exists(
            save_path + '/dismantle_perform_{}.json'.format(analysis_param)):
        with open(save_path + '/dismantle_perform_{}.json'.format(analysis_param), "r") as f:
            dismantle_perform_params = json.load(f)
    else:
        dismantle_perform_params = {}
    for params_value in params_list:
        #  print(type(params_value))
        if str(params_value) in dismantle_perform_params.keys():
            continue

        DiffDGM_infer_param = {
            "gradient_estimator_type": "cost_estimator",
            "guided_model_path": "method/modelGym/Diffusion_model/robustness_metrics_models",
            "use_tag": True,
            "parallel_sampling": 4,
        }
        DiffDGM_infer_param.update({analysis_param: params_value})
        if os.path.exists(
                save_path + '/graphs_dismantling_size.json'):
            with open(save_path + '/graphs_dismantling_size.json', 'r') as f:
                graphs_dismantle_size = json.load(f)
            with open(save_path + "/graphs_dismantling_time.json", 'r') as f:
                graphs_dismantle_time = json.load(f)
            with open(save_path + "/graphs_dismantling_nodelist.json", 'r') as f:
                graphs_dismantle_nodelist = json.load(f)
            os.remove(os.path.join(save_path, 'graphs_dismantling_size.json'))
            os.remove(os.path.join(save_path, 'graphs_dismantling_time.json'))
            os.remove(os.path.join(save_path, 'graphs_dismantling_nodelist.json'))
        else:
            graphs_dismantle_time = {}
            graphs_dismantle_size = {}
            graphs_dismantle_nodelist = {}
        try:
            dismantle_statistics_methods(graphs_dismantle_time=graphs_dismantle_time,
                                         graphs_dismantle_size=graphs_dismantle_size,
                                         graphs_dismantle_nodelist=graphs_dismantle_nodelist,
                                         target=target, method=method,
                                         dataset_path=dataset_path, save_path=save_path,
                                         Rein=Refine, DiffDGM_infer_param=DiffDGM_infer_param)
        except Exception as e:
            with open(save_path + '/graphs_dismantling_size.json', 'w') as f:
                json.dump(graphs_dismantle_size, f, indent=4, separators=(',', ':'))
            with open(save_path + '/graphs_dismantling_time.json', 'w') as f:
                json.dump(graphs_dismantle_time, f, indent=4, separators=(',', ':'))
            with open(save_path + '/graphs_dismantling_nodelist.json', 'w') as f:
                json.dump(graphs_dismantle_nodelist, f, indent=4, separators=(',', ':'))
            raise ValueError("exception:{}".format(e))
        dismantle_perform_params[params_value] = graphs_dismantle_size
        with open(save_path + '/dismantle_perform_{}.json'.format(analysis_param), 'w') as f:
            json.dump(dismantle_perform_params, f, indent=4, separators=(',', ':'))

def plt_model_params_sensitivity(ax,model_param,save_path):
    with open(save_path + '/dismantle_perform_{}.json'.format(model_param), "r") as f:
        dismantle_perform_params = json.load(f)
    params_list = dismantle_perform_params.keys()
    plt_dismantling_static = []
    plt_dismantling_static_std=[]
    graph_types=[
        "barabasi",
        "erdos",
         "small",
        "static"
    ]
    for params_value in params_list:
        std=[]
        mean=[]
        for graph_type in graph_types:

            graph_type_record=[]
            for _ in range(100):
                average_dismantle_steps = [v for k, v in dismantle_perform_params[params_value].items()
                                            if k.startswith(graph_type)
                                           ]
                random_average_dismantle_steps = random.sample(average_dismantle_steps, 99)
                average_dismantle = np.mean(random_average_dismantle_steps)
                graph_type_record.append(average_dismantle*100)
            mean.append(np.mean(graph_type_record))
            std.append(np.std(graph_type_record))


        plt_dismantling_static.append(np.mean(mean))
        plt_dismantling_static_std.append(0.1*np.mean(std))

    plt_dismantling_static = np.array(plt_dismantling_static)
    plt_dismantling_static_std = np.array(plt_dismantling_static_std)
   # print(plt_dismantling_static_std)
    ax.plot(params_list, plt_dismantling_static ,".-"         )

    ax.fill_between(params_list, plt_dismantling_static-plt_dismantling_static_std, plt_dismantling_static+plt_dismantling_static_std, color='#ff7f0e', alpha=0.2)
    ax.set_ylabel("Average size of dismantling set (%)")

   # ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=2))
    ax.set_xlabel("Parameter Value")
    # axes[n // 2][n % 2].tick_params(axis='y', rotation=-45)
    ax.set_title(model_param.replace("_", " "))

if __name__=="__main__":
    """
   
    """
    nep = False  # 节点爆炸渗流精细化节点拆解顺序
    experiment_name="DiffGDM"


    # analysis_param ="rewrite_ratio"  #
    # params_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    # dismantle_model_params_experiment(experiment_name,analysis_param,params_list)

    # plt.rc('font', family='Arial', size=12)
    # fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # model_params_sensitivity_list=["Rewrite_Ratio","Inference_Diffusion_Steps","GNN_Dims","GNN_Layers"]
    # dataset_type = "synth_large"
    # dataset_path = "dataset/dataset_{}".format(dataset_type)
    # save_path = os.path.join("result/{}".format(experiment_name), "dismantle_analysis_model_params")
    # label=["a","b","c","d"]
    # for n,model_param in enumerate(model_params_sensitivity_list):
    #     plt_model_params_sensitivity(ax=axes[n//2][n%2],model_param=model_param,save_path=save_path)
    #     axes[n//2][n%2].text(-0.05, 1.1, label[n], transform=axes[n//2][n%2].transAxes,
    #             fontsize=16, fontweight='bold', va='top', ha='left')
    # plt.tight_layout()
    # plt.savefig(save_path + '/dismantle_model_sensitivity.png')
    #
    # plt.show()
    plt.rc('font', family='Arial', size=12)
    fig, ax = plt.subplots()
    model_param = "Rewrite_Ratio"
    dataset_type = "synth_large"
    dataset_path = "dataset/dataset_{}".format(dataset_type)
    save_path = os.path.join("result/{}".format(experiment_name), "dismantle_analysis_model_params")

    plt_model_params_sensitivity(ax=ax, model_param=model_param, save_path=save_path)
   # plt.savefig(save_path + '/Rewrite_steps_convergence.svg')
    plt.tight_layout()
    plt.show()





















