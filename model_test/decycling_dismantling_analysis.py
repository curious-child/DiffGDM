import json
from glob import glob

import igraph
import networkx as nx
from matplotlib import pyplot as plt

from analysis_visi.LCC_size_removed_nodes import draw_dismantle_methods_graphs_real, draw_statistic_graphs_real, \
    draw_dismantle_statistics_real, draw_dismantle_statistics_synth
from utils.utils import graph_properties_calculate


def plt_decycling_dismantling_analysis(dismantling_file_path,decycling_dismantling_file_path):

    # target=1
    # Refine=True
    #
    # with open(dismantling_file_path + '/graphs_dismantle_size_target_{}_{}.json'.format(target, Refine), 'r') as f:
    #     graphs_dismantle_size = json.load(f)
    # with open(decycling_dismantling_file_path + '/graphs_decycle_size.json', 'r') as f:
    #     graphs_decycle_size = json.load(f)
    # save_file_path = "result/DiffGDM/decycling_dismantling_analysis"
    # decycle_methods_list=["MinSum","DiffGDM","BPD","GFD"]
    # Core_dismantle_methods_list=["CoreDiffGDM_ens","GFD","MinSum","BPD","CoreDiffGDM",]
    # onepass_dismantle_methods_list=["DiffGDM","CI","FINDER","Domirank","NIRM","GDM"]
    # graphs_list=graphs_decycle_size.keys()
    # re_graph_decycle_size={}
    # re_graph_coredismantle_size={}
    # re_graph_onepassdismantle_size={}
    # for graph in graphs_list:
    #     re_graph_decycle_size[graph]={}
    #     re_graph_coredismantle_size[graph]={}
    #     re_graph_onepassdismantle_size[graph]={}
    #     # for decycle_method in decycle_methods_list:
    #     #     re_graph_decycle_size[graph][decycle_method] = graphs_decycle_size[graph][decycle_method]
    #     # for coredismantle_method in Core_dismantle_methods_list:
    #     #     re_graph_coredismantle_size[graph][coredismantle_method]=graphs_dismantle_size[graph][coredismantle_method]
    #     for onepass_method in onepass_dismantle_methods_list:
    #         re_graph_onepassdismantle_size[graph][onepass_method] = graphs_dismantle_size[graph][onepass_method]
    #
    # with open(save_file_path+"/graphs_onepassdismantle_size.json", 'w') as f:
    #     json.dump(re_graph_onepassdismantle_size,f, indent=4, separators=(',', ':'))


    save_file_path="result/DiffGDM/decycling_dismantling_analysis"
    with open(save_file_path+"/graphs_decycle_size.json", 'r') as f:
        re_graph_decycle_size=json.load(f)
    with open(save_file_path+"/graphs_coredismantle_size.json", 'r') as f:
        re_graph_coredismantle_size=json.load(f)
    with open(save_file_path+"/graphs_onepassdismantle_size.json", 'r') as f:
        re_graph_onepassdismantle_size=json.load(f)
    decycle_methods_list = ["MinSum",  "BPD","DiffGDM", "GFD"]
    Core_dismantle_methods_list = ["DiffGDM_ens", "GFD", "MinSum", "BPD", "DiffGDM", ]
    onepass_dismantle_methods_list = ["DiffGDM", "GDM", "FINDER","CI",  "DomiRank"]
    plt.rc('font', family='Arial', size=10)
    fig,axs=plt.subplots(1,3,figsize=(10,6.5),gridspec_kw={'width_ratios': [1, 1.4, 1.2]} )
    draw_dismantle_methods_graphs_real(axs[0], re_graph_decycle_size,method_list=decycle_methods_list,yticklabels=True,cbar=False)
    axs[0].text(-0.05, 1.02, "a", transform=axs[0].transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left')
    draw_dismantle_methods_graphs_real(axs[1], re_graph_coredismantle_size,method_list=Core_dismantle_methods_list,yticklabels=False,cbar=False)
    axs[1].text(-0.05, 1.02, "b", transform=axs[1].transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')
    draw_dismantle_methods_graphs_real(axs[2], re_graph_onepassdismantle_size,method_list=onepass_dismantle_methods_list,
                                       yticklabels=False,cbar=True)
    axs[2].text(-0.05, 1.02, "c", transform=axs[2].transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.show()


def plt_DiffGDM_direction_analysis(dismantling_file_path, decycling_dismantling_file_path):
    target = 1
    Refine = True

    # with open(dismantling_file_path + '/graphs_dismantle_size_target_{}_{}.json'.format(target, Refine), 'r') as f:
    #     graphs_dismantle_size = json.load(f)
    # with open(decycling_dismantling_file_path + '/graphs_decycle_size.json', 'r') as f:
    #     graphs_decycle_size = json.load(f)
    #
    # decycle_methods_list=[
    #    # "DiffGDM",
    #     "DiffGDM_GCC","DiffGDM_ACC","DiffGDM_NC","DiffGDM_Density","DiffGDM_LCC","DiffGDM_base"]
    # Core_dismantle_methods_list=[
    #    "CoreDiffGDM_NC",
    #     "CoreDiffGDM_LCC",
    #                              "CoreDiffGDM_Density","CoreDiffGDM_GE",
    #                              "CoreDiffGDM_ACC","CoreDiffGDM_GCC","CoreDiffGDM_base"]

    # graphs_list=graphs_decycle_size.keys()
    # re_graph_decycle_size={}
    # re_graph_coredismantle_size={}
    # re_graph_onepassdismantle_size={}
    # for graph in graphs_list:
    #     re_graph_decycle_size[graph]={}
    #     re_graph_coredismantle_size[graph]={}
    #     re_graph_onepassdismantle_size[graph]={}
    #     for decycle_method in decycle_methods_list:
    #         re_graph_decycle_size[graph][decycle_method] = graphs_decycle_size[graph][decycle_method]
    #     for coredismantle_method in Core_dismantle_methods_list:
    #         re_graph_coredismantle_size[graph][coredismantle_method]=graphs_dismantle_size[graph][coredismantle_method]
    #
    #
    # save_file_path = "result/DiffGDM/decycling_dismantling_analysis"
    # with open(save_file_path + "/DiffGDM_graphs_decycle_size.json", 'w') as f:
    #     json.dump(re_graph_decycle_size, f, indent=4, separators=(',', ':'))
    # with open(save_file_path + "/DiffGDM_graphs_decycledismantle_size_target.json", 'w') as f:
    #     json.dump(re_graph_coredismantle_size, f, indent=4, separators=(',', ':'))
    save_file_path = "result/DiffGDM/decycling_dismantling_analysis"
    with open(save_file_path + "/DiffGDM_graphs_decycle_size.json", 'r') as f:
        graphs_decycle_size=json.load( f)
    with open(save_file_path + "/DiffGDM_graphs_decycledismantle_size_target.json", 'r') as f:
        graphs_dismantle_size=json.load( f)

    plt.rc('font', family='Arial', size=10)
    fig, axs = plt.subplots(1, 2, figsize=(10,6.5), gridspec_kw={'width_ratios': [1.2, 1.6]})
    draw_dismantle_methods_graphs_real(axs[0], graphs_decycle_size, yticklabels=True, cbar=False)
    axs[0].text(-0.05, 1.02, "a", transform=axs[0].transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')
    draw_dismantle_methods_graphs_real(axs[1], graphs_dismantle_size, yticklabels=False, cbar=True)
    axs[1].text(-0.05, 1.02, "b", transform=axs[1].transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')
    plt.tight_layout()
    plt.show()
def plt_graph_property_analysis():
    save_file_path = "result/DiffGDM/decycling_dismantling_analysis"

    with open(save_file_path + '/graphs_coredismantle_size.json', 'r') as f:
        graphs_decycle_size = json.load(f)
    with open(save_file_path+ '/record_graph_property.json', 'r') as f:
        record_graph_property=json.load(f)
    graph_property_list=[
                    #"LCC",
                         "GCC","ACC","Density","NC","GE","Assortativity",
                       #  "SW_sigma","SW_omega","SF_alpha"
                         ]
    methods_list = ["DiffGDM_ens","MinSum", "BPD",]
   # methods_list = [ "CoreDiffGDM", "CoreDQN", "MinSum", "BPD","CoreHD", "GFD"]
   #  re_record_graph_property={}
   #  dataset_path="dataset/dataset_real_all"
    # for file in sorted(glob(dataset_path + '/' + "*" + ".graphml")):
    #     graph_name = file[len(dataset_path) + 1:].replace(".graphml", "")
    #     print(graph_name)
    #     # if graph_name not in record_graph_property.keys():
    #     #     record_graph_property[graph_name]={}
    #     re_record_graph_property[graph_name] = {}
    #     g = igraph.Graph.Read_GraphML(file)
    #     G = g.to_networkx()
    #     G.remove_edges_from(nx.selfloop_edges((G)))  # 去自环
    #     G.remove_nodes_from(list(nx.isolates(G)))  # 去孤立节点
    #     nx_g = nx.convert_node_labels_to_integers(G)
    #     for graph_property in graph_property_list:
    #         if graph_property in record_graph_property[graph_name].keys():
    #             re_record_graph_property[graph_name][graph_property]=record_graph_property[graph_name][graph_property]
    #         else:
    #             print(graph_property)
    #             record_graph_property[graph_name][graph_property]=graph_properties_calculate(nx_g, graph_property)
    #
    #     with open(save_file_path + '/re_record_graph_property.json', 'w') as f:
    #        json.dump(re_record_graph_property, f, indent=4, separators=(',', ':'))
    plt.rc('font', family='Arial', size=12)
    fig, axs = plt.subplots(1, 2, figsize=(9,5),
                            gridspec_kw={'width_ratios': [1.5, 1.]}
                            )
    draw_statistic_graphs_real(axs[0], record_graph_property, method_list=graph_property_list,yticklabels=True, cbar=False)
    axs[0].text(-0.05, 1.02, "a", transform=axs[0].transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')
    draw_dismantle_methods_graphs_real(axs[1], graphs_decycle_size,
                                      method_list=methods_list,
                                       yticklabels=False, cbar=False)
    axs[1].text(-0.05, 1.02, "b", transform=axs[1].transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.show()
def plt_graph_auc_analysis():
    dismantling_file_path="result/DiffGDM/decycling_dismantling_analysis"
    graph_property_list = [
         "LCC",
         "ACC", "NC", "GE"
    ]
    plt.rc('font', family='Arial', size=10)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    label = ["a", "b", "c", "d"]
    for n,graph_property in enumerate(graph_property_list):
        # if graph_property == "LCC":
        #     method_list=["NIRM","GDM","HD","HB","DiffGDM_base","DiffGDM_LCC"  ]
        # elif graph_property == "ACC":
        #     method_list = ["NIRM", "GDM", "HD", "HB", "HCluster", "DiffGDM_base", "DiffGDM_GCC",
        #                    "DiffGDM_ACC"]
        # elif graph_property == "NC":
        #     method_list = ["NIRM", "GDM", "HD", "HB", "HSubgraph", "DiffGDM_base", "DiffGDM_NC"]
        # elif graph_property == "GE":
        #     method_list = ["NIRM","GDM","HD","HB","HClose","DiffGDM_base","DiffGDM_GE"]
        # else:
        #     raise ValueError("graph_property not recognized")
        # with open(dismantling_file_path+'/graphs_properties_{}_AUC.json'.format(graph_property),"r") as f:
        #     graphs_properties_AUC = json.load(f)
        # re_graphs_dismantle_size={}
        # graph_filter = ["arenas-meta", "corruption", "moreno_crime_projected", "econ-wm1", "inf-USAir97",
        #                 "maayan-foodweb"]
        # for i, graph_name in enumerate(graph_filter):
        #     print("graph_name:{}".format(graph_name))
        #     re_graphs_dismantle_size[graph_name] = {}
        #     for method in method_list:
        #         re_graphs_dismantle_size[graph_name][method] = graphs_properties_AUC[graph_name][method]
        # with open("result/DiffGDM/decycling_dismantling_analysis"+'/graphs_properties_{}_AUC.json'.format(graph_property),"w") as f:
        #     json.dump(re_graphs_dismantle_size, f, indent=4,separators=(',', ':'))
        with open(dismantling_file_path + '/graphs_properties_{}_AUC.json'.format(graph_property), "r") as f:
           graphs_properties_AUC = json.load(f)
        draw_dismantle_statistics_real(ax=axes[n//2][n%2],
                                       graphs_dismantle_size=graphs_properties_AUC)
        axes[n//2][n%2].set_title("The AUC of {} ".format(graph_property))
        axes[n // 2][n % 2].text(-0.05, 1.1, label[n], transform=axes[n // 2][n % 2].transAxes,
                                 fontsize=16, fontweight='bold', va='top', ha='left')
    axes[0][1].legend(loc='upper right', bbox_to_anchor=(1, 1.4),frameon=False,fontsize=8,ncol=3)

    plt.tight_layout()
    plt.show()
def plt_synth_dismantling():
    decycling_dismantling_file_path = "result/DiffGDM/decycle_dataset_synth"
    dismantling_file_path = "result/DiffGDM/dismantle_dataset_synth"
    target=1
    Refine=True

    # with open(dismantling_file_path + '/graphs_dismantle_size_target_{}_{}.json'.format(target, Refine), 'r') as f:
    #     graphs_dismantle_size = json.load(f)
    # with open(decycling_dismantling_file_path + '/graphs_decycle_size.json', 'r') as f:
    #     graphs_decycle_size = json.load(f)
    # save_file_path = "result/DiffGDM/decycling_dismantling_analysis"
    # decycle_methods_list = ["MinSum", "BPD", "CoreHD","CoreDQN", "DiffGDM"]
    # Core_dismantle_methods_list=["MinSum", "BPD", "CoreHD","CoreDQN","DiffGDM",]
    # graphs_list=graphs_decycle_size.keys()
    # re_graph_decycle_size={}
    # re_graph_coredismantle_size={}
    #
    # for graph in graphs_list:
    #     re_graph_decycle_size[graph]={}
    #     re_graph_coredismantle_size[graph]={}
    #
    #     for decycle_method in decycle_methods_list:
    #         re_graph_decycle_size[graph][decycle_method] = graphs_decycle_size[graph][decycle_method]
    #     for coredismantle_method in Core_dismantle_methods_list:
    #         re_graph_coredismantle_size[graph][coredismantle_method]=graphs_dismantle_size[graph][coredismantle_method]


    # with open(save_file_path+"/synth_graphs_decycle_size.json", 'w') as f:
    #     json.dump(re_graph_decycle_size,f, indent=4, separators=(',', ':'))
    # with open(save_file_path+"/synth_graphs_dismantling_size.json", 'w') as f:
    #     json.dump(re_graph_coredismantle_size,f, indent=4, separators=(',', ':'))
    save_file_path = "result/DiffGDM/decycling_dismantling_analysis"
    decycle_methods_list = ["MinSum", "BPD", "CoreHD", "CoreDQN", "DiffGDM"]
    Core_dismantle_methods_list = ["MinSum", "BPD", "CoreHD", "CoreDQN", "DiffGDM", ]
    with open(save_file_path+"/synth_graphs_decycle_size.json", 'r') as f:
        re_graph_decycle_size=json.load(f)
    with open(save_file_path+"/synth_graphs_dismantling_size.json", 'r') as f:
        re_graph_coredismantle_size=json.load(f)
    plt.rc('font', family='Arial', size=12)
    fig, axs = plt.subplots(2, 1, figsize=(6, 6),
                            #gridspec_kw={'width_ratios': [1, 1.2, 1.3]}
                            )

    draw_dismantle_statistics_synth(axs[0], re_graph_decycle_size, )
    axs[0].text(-0.05, 1.02, "a", transform=axs[0].transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')
    draw_dismantle_statistics_synth(axs[1], re_graph_coredismantle_size, )
    axs[1].text(-0.05, 1.02, "b", transform=axs[1].transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    decycling_dismantling_file_path="result/DiffGDM/decycle_dataset_real_all"
    dismantling_file_path="result/DiffGDM/dismantle_dataset_real_all"
    plt_decycling_dismantling_analysis(dismantling_file_path, decycling_dismantling_file_path)
   # plt_DiffGDM_direction_analysis(dismantling_file_path,decycling_dismantling_file_path)
    #plt_graph_property_analysis()
    #plt_graph_auc_analysis()
   # plt_synth_dismantling()