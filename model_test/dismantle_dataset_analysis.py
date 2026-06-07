import json
import math
from glob import glob

import matplotlib.pyplot as plt

from method.method import  NodeBreakList

from utils.utils import *

from matplotlib.ticker import PercentFormatter


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





def dismantle_dataset_noise_experiment(experiment_name,method_list,analysis_param):
    Refine = True  # 网络节点重插
    target = 1  # 拆解阈值 （%）
    dataset_type = "real_all"
    dataset_path = "dataset/dataset_{}".format(dataset_type)
    save_path = os.path.join("result/{}".format(experiment_name), "dismantle_analysis_dataset")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

   # analysis_param = "Missing_link"  # Spurious_link
    params_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    if os.path.exists(
            save_path + '/method_dismantle_perform_{}.json'.format(analysis_param)):
        with open(save_path + '/method_dismantle_perform_{}.json'.format(analysis_param), "r") as f:
            method_dismantle_perform_params = json.load(f)
    else:
        method_dismantle_perform_params = {}

    for method in method_list:
        print(method)
        if method not in method_dismantle_perform_params.keys():
            method_dismantle_perform_params[method] = {}
        for params_value in params_list:
            print(params_value)
            if str(params_value) in method_dismantle_perform_params[method].keys():
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
            method_dismantle_perform_params[method][params_value] = graphs_dismantle_size
            with open(save_path + '/method_dismantle_perform_{}.json'.format(analysis_param), 'w') as f:
                json.dump(method_dismantle_perform_params, f, indent=4, separators=(',', ':'))

def plt_dataset_analysis(experiment_name,method_list):
    save_path = os.path.join("result/{}".format(experiment_name), "dismantle_analysis_dataset")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.rc('font', family='Arial', size=10)
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    analysis_params = ["Missing_link", "Spurious_link"]
    params_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    lable=["a","b"]
    for n, analysis_param in enumerate(analysis_params):
        with open(save_path + '/method_dismantle_perform_{}.json'.format(analysis_param), "r") as f:
            method_dismantle_perform_params = json.load(f)

        for method in method_list:
            plt_dismantling_static = []
            for params_value in params_list:
                average_dismantle_steps = [v for k, v in
                                           method_dismantle_perform_params[method][str(params_value)].items()

                                           ]
                average_dismantle = np.mean(average_dismantle_steps)

                plt_dismantling_static.append(average_dismantle)
            axes[n].plot(params_list, plt_dismantling_static, ".-", label=method)

        axes[n].legend()
        axes[n].xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axes[n].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axes[n].set_xlabel(analysis_param.replace("_", " "))
        axes[n].set_ylabel("Average size of dismantling set")
        axes[n].text(-0.05, 1.05, lable[n], transform=axes[n].transAxes,
                    fontsize=16, fontweight='bold', va='top', ha='left')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + '/dismantle_dataset_analysis.svg')
    #  #
    plt.show()
if __name__=="__main__":
    """
   
    """
    experiment_name="DiffGDM"

    method_list = [
        "CI",
         "DiffGDM",
           "NIRM",
           "Domirank"
        ]#
    analysis_params=["Missing_link","Spurious_link"]
    # for analysis_param in analysis_params:
    #     dismantle_dataset_noise_experiment(experiment_name=experiment_name,method_list=method_list,analysis_param=analysis_param)
    plt_dataset_analysis(experiment_name=experiment_name,method_list=method_list)























