import json
import os
import argparse
import hashlib
import math
import statistics
from pathlib import Path
from glob import glob

import igraph
import networkx as nx
from matplotlib import pyplot as plt

from analysis_visi.LCC_size_removed_nodes import draw_dismantle_methods_graphs_real, draw_statistic_graphs_real, \
    draw_dismantle_statistics_real, draw_dismantle_statistics_synth
from utils.utils import graph_properties_calculate


def prepare_dismantling_analysis_results(dismantling_file_path,
                                         save_file_path="result/DiffGDM/decycling_dismantling_analysis"):
    """Split explicit decycle/direct result keys into the two plotting files."""
    result_file = os.path.join(
        dismantling_file_path, "graphs_dismantle_size_target_1_True.json")
    with open(result_file, "r") as f:
        graphs_dismantle_size = json.load(f)

    decycle_mapping = {
        "DiffGDM_decycle": "DiffGDM",
        "T2T_decycle": "T2T",
        "Random-Decycle": "Random-Decycle",
    }
    direct_mapping = {
        "DiffGDM_direct": "DiffGDM",
        "T2T_direct": "T2T",
    }
    decycle_optional = ("GFD", "MinSum", "BPD")
    direct_optional = ("GDM", "FINDER", "CI", "DomiRank", "NIRM")

    def select_results(mapping, optional_methods):
        selected = {}
        for graph_name, method_results in graphs_dismantle_size.items():
            missing = [method for method in mapping if method not in method_results]
            if missing:
                raise KeyError("Missing results for {}: {}".format(
                    graph_name, ", ".join(missing)))
            selected[graph_name] = {
                display_name: method_results[result_name]
                for result_name, display_name in mapping.items()
            }
            for method in optional_methods:
                if method in method_results:
                    selected[graph_name][method] = method_results[method]
        return selected

    decycle_results = select_results(decycle_mapping, decycle_optional)
    direct_results = select_results(direct_mapping, direct_optional)
    os.makedirs(save_file_path, exist_ok=True)
    with open(os.path.join(save_file_path, "graphs_coredismantle_size.json"), "w") as f:
        json.dump(decycle_results, f, indent=4, separators=(',', ':'))
    with open(os.path.join(save_file_path, "graphs_onepassdismantle_size.json"), "w") as f:
        json.dump(direct_results, f, indent=4, separators=(',', ':'))
    return decycle_results, direct_results


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
    # Core_dismantle_methods_list=["DiffGDM_ens","GFD","MinSum","BPD","DiffGDM",]
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
    prepare_dismantling_analysis_results(dismantling_file_path, save_file_path)
    with open(save_file_path+"/graphs_decycle_size.json", 'r') as f:
        re_graph_decycle_size=json.load(f)
    with open(save_file_path+"/graphs_coredismantle_size.json", 'r') as f:
        re_graph_coredismantle_size=json.load(f)
    with open(save_file_path+"/graphs_onepassdismantle_size.json", 'r') as f:
        re_graph_onepassdismantle_size=json.load(f)
    decycle_methods_list = ["MinSum",  "BPD","DiffGDM", "GFD"]
    Core_dismantle_methods_list = [
        method for method in ["DiffGDM", "T2T", "Random-Decycle", "GFD", "MinSum", "BPD"]
        if all(method in results for results in re_graph_coredismantle_size.values())
    ]
    onepass_dismantle_methods_list = [
        method for method in ["DiffGDM", "T2T", "GDM", "FINDER", "CI", "DomiRank", "NIRM"]
        if all(method in results for results in re_graph_onepassdismantle_size.values())
    ]
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


def plt_DiffGDM_direction_analysis(dismantling_file_path, decycling_dismantling_file_path,
                                  *, result_dir=None, show=True, seed_file=None,
                                  return_stats=False, verified_payload=None):
    if verified_payload is not None:
        if seed_file is not None or verified_payload.get("profile") != "original-plus-controls":
            raise ValueError("Historical guidance heatmaps require their explicit scalar profile")
        from analysis_visi.LCC_size_removed_nodes import draw_verified_guidance_comparison
        _verified_hash_guard(verified_payload["input_SHA256"])
        fig = draw_verified_guidance_comparison(verified_payload)
        _verified_hash_guard(verified_payload["input_SHA256"])
        if show:
            plt.show()
        return (fig, verified_payload) if return_stats else fig
    if seed_file is not None:
        directory = (Path(result_dir) if result_dir is not None else
                     Path("result/DiffGDM/decycling_dismantling_analysis"))
        payload = verified_figure6_payload(directory, seed_file)
        _verified_hash_guard(payload["input_SHA256"])
        with plt.rc_context({'font.family': 'Arial', 'font.size': 10,
                             'pdf.fonttype': 42, 'text.usetex': False}):
            fig, axes = plt.subplots(1, 2, figsize=(11, 9),
                                     gridspec_kw={'width_ratios': [6, 7]})
            numeric_artists = []
            for index, (panel, ax) in enumerate(zip(('a', 'b'), axes)):
                details = payload['panels'][panel]
                mesh = draw_dismantle_methods_graphs_real(
                    ax, details['graph_values'], method_list=details['methods'],
                    yticklabels=index == 0, cbar=False,
                    verified_cells=details['graph_values'], average=True,
                    graph_order=payload['graphs'],
                    average_statistics=details['average_statistics'])
                numeric_artists.extend({'panel': panel, **item}
                                       for item in mesh._verified_numeric_artists)
                for item in mesh._verified_numeric_artists:
                    item['artist'].set_fontsize(8)
                    item['artist'].set_linespacing(1.0)
                ax.set_xticks([i + .5 for i in range(len(details['methods']))],
                              [method.replace('DiffGDM_', 'DiffGDM\n')
                               for method in details['methods']], fontsize=8)
                ax.set_yticks([i + .5 for i in range(len(payload['graphs']) + 1)],
                              [payload['display_names'][graph]
                               for graph in payload['graphs']] + ['Average'],
                              fontsize=8.5)
                ax.tick_params(axis='y', labelleft=index == 0)
                for label in ax.get_yticklabels():
                    label.set_linespacing(1.0)
                if index == 0:
                    ax.get_yticklabels()[-1].set_fontweight('bold')
                ax.text(0, 1.07, panel, transform=ax.transAxes,
                        fontsize=11, fontweight='bold', va='bottom', ha='left')
                ax.text(.08, 1.07, details['title'], transform=ax.transAxes,
                        fontsize=10, va='bottom', ha='left')
                ax.axhline(len(payload['graphs']), color='#555555', linewidth=.8)
            fig.tight_layout(pad=1.25, w_pad=1.0)
        fig._verified_numeric_artists = numeric_artists
        fig._verified_data_axes = tuple(axes)
        _verified_hash_guard(payload['input_SHA256'])
        if show:
            plt.show()
        return (fig, payload) if return_stats else fig
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
    #    "DiffGDM_NC",
    #     "DiffGDM_LCC",
    #                              "DiffGDM_Density","DiffGDM_GE",
    #                              "DiffGDM_ACC","DiffGDM_GCC","DiffGDM_base"]

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
    save_file_path = (Path(result_dir) if result_dir is not None else
                      Path("result/DiffGDM/decycling_dismantling_analysis"))
    with open(save_file_path / "DiffGDM_graphs_decycle_size.json", 'r') as f:
        graphs_decycle_size=json.load( f)
    with open(save_file_path / "DiffGDM_graphs_decycledismantle_size_target.json", 'r') as f:
        graphs_dismantle_size=json.load( f)

    plt.rc('font', family='Arial', size=10)
    fig, axs = plt.subplots(1, 2, figsize=(10,6.5) if show else (11,9),
                            gridspec_kw={'width_ratios': [1.2, 1.6]})
    draw_dismantle_methods_graphs_real(axs[0], graphs_decycle_size, yticklabels=True, cbar=False)
    axs[0].text(-0.05, 1.02, "a", transform=axs[0].transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')
    draw_dismantle_methods_graphs_real(axs[1], graphs_dismantle_size, yticklabels=False, cbar=True)
    axs[1].text(-0.05, 1.02, "b", transform=axs[1].transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')
    if not show:
        for ax in axs:
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8.5)
            for label in ax.get_yticklabels():
                label.set_linespacing(1.0)
        fig.tight_layout(pad=1.25, w_pad=1.0)
    else:
        plt.tight_layout()
    if show:
        plt.show()
    return fig
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
   # methods_list = [ "DiffGDM", "CoreDQN", "MinSum", "BPD","CoreHD", "GFD"]
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
def plt_graph_auc_analysis(*, result_dir=None, show=True):
    dismantling_file_path = (Path(result_dir) if result_dir is not None else
                             Path("result/DiffGDM/decycling_dismantling_analysis"))
    graph_property_list = [
         "LCC",
         "ACC", "NC", "GE"
    ]
    plt.rc('font', family='Arial', size=8)
    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 135 / 25.4))
    fig.subplots_adjust(left=.045, right=.985, bottom=.13, top=.865,
                        hspace=.52, wspace=.12)
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
        with open(dismantling_file_path / 'graphs_properties_{}_AUC.json'.format(graph_property), "r") as f:
           graphs_properties_AUC = json.load(f)
        draw_dismantle_statistics_real(ax=axes[n//2][n%2],
                                       graphs_dismantle_size=graphs_properties_AUC,
                                       label_fontsize=7, apply_tight_layout=False)
        ax = axes[n // 2][n % 2]
        ax.set_title("AUC of {}".format(graph_property), fontsize=9, pad=7)
        ax.text(-.035, 1.045, label[n], transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='bottom', ha='left')
        methods = list(next(iter(graphs_properties_AUC.values())))
        ax.set_xticks(range(len(methods)), methods, rotation=35, ha='right',
                      rotation_mode='anchor', fontsize=7)
        ax.tick_params(axis='x', labelsize=7, pad=4)
        ax.margins(y=.13)
        ax._auc_metric = graph_property
    handles, labels = axes[0][0].get_legend_handles_labels()
    labels = [label.replace('\\n', '').replace('\n', '').replace('\r', '') for label in labels]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .995),
               frameon=False, fontsize=7, ncol=6, columnspacing=.85,
               handlelength=1.05, handletextpad=.35)
    fig._verified_data_axes = tuple(axes.flat)
    fig._exact_export_size = True
    from analysis_visi.LCC_size_removed_nodes import _publication_font_floor
    _publication_font_floor(fig, 7.4, 173.8)
    if show:
        plt.show()
    return fig
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

def _verified_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _verified_read(path, inputs):
    path = Path(path).resolve()
    inputs[str(path)] = _verified_sha(path)
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _verified_hash_guard(expected):
    actual = {str(path): _verified_sha(path) for path in sorted(expected)}
    changed = [path for path in expected if actual[path] != expected[path]]
    if changed:
        raise ValueError("Read-only figure inputs changed: " + ", ".join(changed[:3]))
    return _verified_digest(actual)


def _verified_canonical_graphs(data):
    clean = {}
    for name, values in data.items():
        key = name.replace("\\n", "").replace("\n", "").replace("\r", "")
        if key in clean:
            raise ValueError("Duplicate graph alias: " + key)
        clean[key] = values
    return clean


def _verified_ten_seed_cell(values, trace):
    if len(values) != 10 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("A verified cell must contain ten finite seed fractions")
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values),
            "variance": statistics.variance(values), "seed_num": 10,
            "sampling": "10 inference seeds, not training replicates",
            "seed_values": values, "source_trace": trace}


def _verified_average_statistics(graphs, panel, methods, graph_values):
    """Equal-network means and mean within-network ten-seed sample variances.

    Variance is stored in fraction-squared units. Its square root is the RMS
    within-network SD, not the uncertainty or SEM of the network-average mean.
    """
    eligible = {"a": {"DiffGDM", "Random-Decycle", "T2T-FVS"},
                "b": {"DiffGDM_ens", "DiffGDM", "Random-Decycle", "T2T_decycle"},
                "c": {"DiffGDM", "T2T_direct"}}[panel]
    if (len(graphs) != 27 or len(set(graphs)) != 27
            or set(graph_values) != set(graphs)
            or len(set(methods)) != len(methods) or not eligible <= set(methods)):
        raise ValueError("Average requires the complete 27-network method domain")
    summaries = {}
    for method in methods:
        cells = [graph_values[graph][method] for graph in graphs]
        summary = {"mean": statistics.mean(cell["mean"] for cell in cells),
                   "network_num": 27, "seed_num": None, "variance_ddof": None,
                   "mean_within_network_variance": None, "rms_within_network_sd": None}
        if method in eligible:
            variances = []
            for cell in cells:
                trace = cell["source_trace"]
                if (cell.get("seed_num") != 10 or trace.get("seeds") != list(range(10))
                        or trace.get("metric") != ("raw_fvs_ratio" if panel == "a" else "cfn")):
                    raise ValueError("Average variance requires matched seeds 0-9 for every network")
                rebuilt = _verified_ten_seed_cell(cell["seed_values"], trace)
                for key in ("mean", "variance", "sd"):
                    value = cell.get(key)
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value) or value < 0
                            or not math.isclose(value, rebuilt[key], rel_tol=0, abs_tol=1e-14)):
                        raise ValueError("Stored ten-seed statistics differ from per-seed values")
                variances.append(rebuilt["variance"])
            mean_variance = statistics.mean(variances)
            summary.update(seed_num=10, variance_ddof=1,
                           mean_within_network_variance=mean_variance,
                           rms_within_network_sd=math.sqrt(mean_variance))
        elif any(cell.get("seed_num") is not None or cell.get("variance") is not None
                 or cell.get("sd") is not None for cell in cells):
            raise ValueError("Historical scalar methods must not receive an inferred Average SD")
        summaries[method] = summary
    return summaries


def verified_figure6_payload(result_dir, seed_file):
    """Read Figure 6's original order and authorized seed-level measurements.

    A fully committed paired publication supplies same-call FVS/CFN samples.
    Otherwise, retain the historical FVS scalar/previous ten-seed CFN branch.
    """
    from real_seed_variance import paired_publication_snapshot
    directory = Path(result_dir).resolve()
    canonical = Path(seed_file).resolve().parent
    paired = paired_publication_snapshot(canonical)
    sources = {
        'a': directory / 'DiffGDM_graphs_decycle_size.json',
        'b': directory / 'DiffGDM_graphs_decycledismantle_size_target.json',
        'seeds': Path(seed_file).resolve(),
    }
    if paired is not None:
        sources['raw_fvs'] = canonical / 'raw_fvs_seed_target_1_True.json'
    inputs = {}

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate Figure 6 input key: ' + key)
            result[key] = value
        return result

    raw = {}
    for name, path in sources.items():
        inputs[str(path)] = _verified_sha(path)
        with path.open(encoding='utf-8') as stream:
            raw[name] = json.load(stream, object_pairs_hook=unique_object)
        if not isinstance(raw[name], dict):
            raise ValueError('Figure 6 requires graph-keyed objects: ' + str(path))
    data = {name: _verified_canonical_graphs(value) for name, value in raw.items()}
    graphs = list(data['a'])
    if len(graphs) != 27 or any(set(value) != set(graphs) for value in data.values()):
        raise ValueError('Figure 6 requires matching complete 27-network domains')
    if paired is not None:
        if (len(paired['graphs']) != 27 or set(paired['graphs']) != set(graphs)
                or set(paired['node_counts']) != set(graphs)):
            raise ValueError('Published Figure 6 paired run has a different graph domain')
        for path, expected_sha in paired['input_SHA256'].items():
            if path in inputs and inputs[path] != expected_sha:
                raise ValueError('Figure 6 paired snapshot changed during loading: ' + path)
        inputs.update(paired['input_SHA256'])
    display_names = {
        name.replace('\\n', '').replace('\n', '').replace('\r', ''):
        name.replace('\\n', '\n').replace('\r', '')
        for name in raw['a']
    }
    expected_methods = {
        'a': {'DiffGDM_GCC', 'DiffGDM_ACC', 'DiffGDM_NC', 'DiffGDM_Density',
              'DiffGDM_LCC', 'DiffGDM_base'},
        'b': {'DiffGDM_NC', 'DiffGDM_LCC', 'DiffGDM_Density', 'DiffGDM_GE',
              'DiffGDM_ACC', 'DiffGDM_GCC', 'DiffGDM_base'},
    }

    def paired_trial(graph, method, seed):
        key = graph + '\0' + method + '\0' + str(seed)
        record = paired['records'].get(key)
        node_count = paired['node_counts'][graph]
        if (not isinstance(record, dict) or record.get('status') != 'success'
                or record.get('graph') != graph or record.get('method') != method
                or type(record.get('seed')) is not int or record['seed'] != seed
                or type(node_count) is not int or node_count <= 0
                or record.get('node_count') != node_count
                or record.get('same_call_id') != key):
            raise ValueError('Invalid same-call Figure 6 record: ' + key.replace('\0', '/'))
        seed_key = method + '_' + str(seed)
        nodes = data['raw_fvs'][graph][seed_key]
        cfn = data['seeds'][graph][seed_key]
        if (not isinstance(nodes, list) or len(nodes) != len(set(nodes))
                or any(type(node) is not int or not 0 <= node < node_count for node in nodes)
                or nodes != record.get('raw_fvs_nodes')
                or record.get('raw_fvs_ratio') != len(nodes) / node_count
                or cfn != record.get('cfn')):
            raise ValueError('Canonical FVS/CFN differs from its same-call record: '
                             + key.replace('\0', '/'))
        return record

    panels = {}
    for panel in ('a', 'b'):
        first = data[panel][graphs[0]]
        if not isinstance(first, dict) or set(first) != expected_methods[panel]:
            raise ValueError('Figure 6 original method domain differs for panel ' + panel)
        methods = list(first)
        graph_values = {}
        for graph in graphs:
            if (not isinstance(data[panel][graph], dict)
                    or set(data[panel][graph]) != set(methods)):
                raise ValueError('Figure 6 original method coverage differs for ' + graph)
            graph_values[graph] = {}
            for method in methods:
                trace = {'source_group': ('published_paired' if paired is not None else
                                         'historical' if panel == 'a' else 'real_seed_variance'),
                         'source_file': str(sources['raw_fvs' if panel == 'a' and paired is not None
                                                   else panel if panel == 'a' else 'seeds']),
                         'source_method': method, 'graph': graph,
                         'metric': 'raw_fvs_ratio' if panel == 'a' else 'cfn'}
                if panel == 'a' and paired is None:
                    value = data[panel][graph][method]
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value) or not 0 <= value <= 1):
                        raise ValueError('Invalid Figure 6 historical FVS fraction')
                    cell = {'mean': value, 'sd': None, 'variance': None, 'seed_num': None,
                            'sampling': 'historical scalar FVS; ten-seed FVS unavailable',
                            'source_trace': trace}
                else:
                    leaf = data['raw_fvs' if panel == 'a' else 'seeds'][graph]
                    if not isinstance(leaf, dict):
                        raise ValueError('Figure 6 seed values must be graph-keyed objects')
                    keys = {method + '_' + str(seed) for seed in range(10)}
                    actual = {key for key in leaf if key.startswith(method + '_')}
                    if actual != keys:
                        raise ValueError('Figure 6 requires exact seeds 0-9 for ' + graph + '/' + method)
                    trace['seeds'] = list(range(10))
                    if paired is not None:
                        records = [paired_trial(graph, method, seed) for seed in range(10)]
                        trace.update(run_id=paired['run_id'], node_count=paired['node_counts'][graph],
                                     same_call_ids=[record['same_call_id'] for record in records])
                        values = [record['raw_fvs_ratio' if panel == 'a' else 'cfn']
                                  for record in records]
                    else:
                        values = [leaf[method + '_' + str(seed)] for seed in range(10)]
                    cell = _verified_ten_seed_cell(values, trace)
                graph_values[graph][method] = cell
        average_statistics = {}
        for method in methods:
            cells = [graph_values[graph][method] for graph in graphs]
            summary = {'mean': statistics.mean(cell['mean'] for cell in cells),
                       'network_num': 27, 'seed_num': None, 'variance_ddof': None,
                       'mean_within_network_variance': None, 'rms_within_network_sd': None}
            if panel == 'b' or paired is not None:
                mean_variance = statistics.mean(cell['variance'] for cell in cells)
                summary.update(seed_num=10, variance_ddof=1,
                               mean_within_network_variance=mean_variance,
                               rms_within_network_sd=math.sqrt(mean_variance))
            average_statistics[method] = summary
        panels[panel] = {
            'methods': methods, 'metric': 'raw_fvs_ratio' if panel == 'a' else 'cfn',
            'title': ('FVS (%)' if paired is not None else 'FVS (%, historical values)')
                     if panel == 'a' else 'Dismantling CFN (%)',
            'graph_values': graph_values, 'average_statistics': average_statistics,
        }
    _verified_hash_guard(inputs)
    return {'graphs': graphs, 'display_names': display_names, 'panels': panels,
            'paired_publication': paired is not None,
            'paired_run_id': None if paired is None else paired['run_id'],
            'cell_counts': {'historical_scalar_cells': 162 if paired is None else 0,
                            'ten_seed_cells': 189 if paired is None else 351,
                            'ten_seed_FVS_cells': 0 if paired is None else 162,
                            'ten_seed_CFN_cells': 189,
                            'average_cells': 13, 'rendered_numeric_labels': 364},
            'input_SHA256': inputs,
            'new_cells': ('FVS and CFN: mean +/- sample SD of ten paired same-call inference seeds 0-9, ddof=1'
                          if paired is not None else
                          'CFN: mean +/- sample SD of ten inference seeds 0-9, ddof=1; FVS: historical scalar only'),
            'average_rule': 'Equal weight over 27 network means; ten-seed columns display sqrt(mean of 27 within-network ddof=1 seed variances), not mean SD, network-between SD, SEM or uncertainty of the Average mean',
            'unit_rule': 'Values are fractions; SD displays in percentage points (*100); variance in fraction squared, or percentage-point squared (*10000)'}


def verified_result_hashes(canonical_dir, expected_hashes):
    """Follow audited endpoint migrations without trusting current file hashes."""
    import sys
    canonical_dir = Path(canonical_dir).resolve()
    workspace = canonical_dir.parents[4]
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from experiment_ops.merge_server_results import canonical_t2t4_hashes
    from real_seed_variance import paired_authorized_hashes
    approved = canonical_t2t4_hashes(canonical_dir, expected_hashes)
    approved = paired_authorized_hashes(canonical_dir, approved)
    return single_controls_authorized_hashes(canonical_dir.parent, approved)


def _original_scalar_cell(value, trace, single=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("Invalid original/single-run figure value")
    return {"mean": value, "sd": None, "variance": None,
            "seed_num": 1 if single else None,
            "sampling": "one recorded run" if single else "retained historical scalar",
            "source_trace": trace}


def _original_average_statistics(graphs, methods, values):
    return {method: {"mean": math.fsum(values[g][method]["mean"] for g in graphs) / len(graphs),
                     "network_num": len(graphs), "seed_num": None, "variance_ddof": None,
                     "mean_within_network_variance": None, "rms_within_network_sd": None}
            for method in methods}


def _guidance_component_case(case_id, graph, methods, network, node_sets, endpoints, aggregation):
    """Construct an exact static correspondence, never a postprocessing trace.

    Node IDs retain the original processed-graph identity. A displayed pool is
    a union of separately recorded connected components, not a single tree.
    """
    n = network.number_of_nodes()
    threshold = (n + 99) // 100
    selected = {side: frozenset(node_sets[method])
                for side, method in (("base", "base"), ("guided", methods[0]))}
    for method in methods[1:]:
        if (frozenset(node_sets[method]) != selected["guided"]
                or endpoints[method] != {**endpoints[methods[0]], "graph": graph}):
            raise ValueError("Case guides are not identical candidates and endpoints: " + graph)
    components, blocks, remaining = {}, {}, {}
    for side in ("base", "guided"):
        remaining[side] = set(network) - selected[side]
        forest = network.subgraph(remaining[side])
        groups = sorted((sorted(group) for group in nx.connected_components(forest)),
                        key=lambda nodes: (-len(nodes), nodes))
        if forest.number_of_edges() != len(remaining[side]) - len(groups):
            raise ValueError("Component case is not a forest: " + graph + "/" + side)
        components[side] = [{"id": side + "_component_" + str(index + 1),
                             "nodes": nodes, "size": len(nodes),
                             "oversized": len(nodes) > threshold}
                            for index, nodes in enumerate(groups)]
    identical = ({tuple(component["nodes"]) for component in components["base"]}
                 & {tuple(component["nodes"]) for component in components["guided"]})
    if aggregation not in ("identical_components", "small_components"):
        raise ValueError("Unknown component-case aggregation")
    for side in ("base", "guided"):
        pooled, singles = [], []
        for component in components[side]:
            use_pool = (tuple(component["nodes"]) in identical
                        if aggregation == "identical_components"
                        else component["size"] <= threshold)
            (pooled if use_pool else singles).append(component)

        def block(block_id, kind, members):
            nodes = sorted(node for component in members for node in component["nodes"])
            return {"id": block_id, "kind": kind, "nodes": nodes, "size": len(nodes),
                    "component_ids": [component["id"] for component in members],
                    "component_count": len(members),
                    "oversized_count": sum(component["oversized"] for component in members),
                    "component_sizes": [component["size"] for component in members]}

        blocks[side] = [block(side + "_block_" + str(index + 1), "single", [component])
                        for index, component in enumerate(singles)]
        if pooled:
            kind = "identical_pool" if aggregation == "identical_components" else "small_pool"
            blocks[side].append(block(side + "_" + kind, kind, pooled))
        flattened = [node for item in blocks[side] for node in item["nodes"]]
        if len(flattened) != len(set(flattened)) or set(flattened) != remaining[side]:
            raise ValueError("Display blocks do not partition retained nodes: " + graph)
    shared = remaining["base"] & remaining["guided"]
    links = []
    for base_block in blocks["base"]:
        for guided_block in blocks["guided"]:
            nodes = sorted(set(base_block["nodes"]) & set(guided_block["nodes"]))
            if nodes:
                links.append({"base_id": base_block["id"], "guided_id": guided_block["id"],
                              "count": len(nodes), "nodes": nodes})
    flattened_links = [node for link in links for node in link["nodes"]]
    if len(flattened_links) != len(set(flattened_links)) or set(flattened_links) != shared:
        raise ValueError("Component links do not partition shared nodes: " + graph)
    exclusive, terminal = {}, {}
    for side, other, method in (("base", "guided", "base"), ("guided", "base", methods[0])):
        nodes = remaining[side] - remaining[other]
        by_block = {item["id"]: sorted(set(item["nodes"]) & nodes) for item in blocks[side]}
        exclusive[side] = {"count": len(nodes), "nodes": sorted(nodes), "by_block": by_block}
        endpoint = endpoints[method]
        terminal[side] = {"fvs_nodes": len(selected[side]),
                          "K": sum(item["oversized"] for item in components[side]),
                          "final_nodes": endpoint["final_nodes"],
                          "remaining_nodes": len(remaining[side]),
                          "component_count": len(components[side])}
        if len(remaining[side]) + len(selected[side]) != n:
            raise ValueError("Case selected/retained nodes do not partition graph")
        for item in blocks[side]:
            linked = sum(link["count"] for link in links if link[side + "_id"] == item["id"])
            if linked + len(by_block[item["id"]]) != item["size"]:
                raise ValueError("Shared/exclusive nodes do not conserve display block size")
    if (len(selected["guided"]) - len(selected["base"])
            != exclusive["base"]["count"] - exclusive["guided"]["count"]):
        raise ValueError("Case FVS difference does not match node exchange")
    component_membership = {"fvs": {side: sorted(nodes) for side, nodes in selected.items()},
                            "components": components}
    membership_sha = hashlib.sha256(json.dumps(component_membership, sort_keys=True,
        separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    return {"id": case_id, "graph": graph, "methods": methods, "N": n,
            "threshold": threshold, **terminal, **component_membership,
            "blocks": blocks, "links": links, "exclusive": exclusive,
            "shared_retained": {"count": len(shared), "nodes": sorted(shared)},
            "aggregation": aggregation, "membership_SHA256": membership_sha,
            "validation": {"methods_have_identical_candidates_and_endpoints": True,
                "both_residuals_are_forests": True, "original_node_ids_preserved": True,
                "all_components_and_blocks_partition_retained_nodes": True,
                "shared_and_exclusive_nodes_conserve_every_block": True,
                "all_link_nodes_are_exact_intersections": True}}


def guidance_benefit_payload(workspace, *, include_metric_summary=False):
    """Validate historical guidance endpoints and saved forests, without inference.

    This independent analysis never prepares plotting inputs or reads seed means.
    All seven saved configurations remain validated. The three displayed guides
    were selected by the earlier exploratory analysis, not independent repeats.
    """
    def midranks(values):
        order = sorted(range(len(values)), key=values.__getitem__)
        ranks = [0.0] * len(values)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and values[order[stop]] == values[order[start]]:
                stop += 1
            rank = (start + 1 + stop) / 2
            for index in order[start:stop]:
                ranks[index] = rank
            start = stop
        return ranks

    def centered(values):
        mean = math.fsum(values) / len(values)
        return [value - mean for value in values]

    def correlation(x, y):
        x, y = centered(x), centered(y)
        xx, yy = math.fsum(v * v for v in x), math.fsum(v * v for v in y)
        if xx <= 1e-20 or yy <= 1e-20:
            return None
        value = math.fsum(a * b for a, b in zip(x, y)) / math.sqrt(xx * yy)
        return max(-1.0, min(1.0, value))

    def spearman(x, y):
        return correlation(midranks(x), midranks(y))

    def association(rows, field):
        x = [row[field] for row in rows]
        y = [row["delta_cfn_pp"] for row in rows]
        z = [row["delta_fvs_pp"] for row in rows]
        rho = spearman(x, y)
        leave_one_out = [{"omitted_graph": row["graph"],
                          "rho": spearman(x[:index] + x[index + 1:],
                                          y[:index] + y[index + 1:])}
                         for index, row in enumerate(rows)]
        defined = [item["rho"] for item in leave_one_out if item["rho"] is not None]
        direction = ("undefined" if rho is None else "negative" if rho < 0
                     else "positive" if rho > 0 else "zero")
        same_direction = (None if rho is None else sum(
            (value > 0) - (value < 0) == (rho > 0) - (rho < 0) for value in defined))
        xr, yr, zr = (centered(midranks(values)) for values in (x, y, z))
        zz = math.fsum(value * value for value in zr)
        if zz <= 1e-20:
            partial_rho, partial_reason = None, "control ranks are constant"
        else:
            x_slope = math.fsum(a * b for a, b in zip(xr, zr)) / zz
            y_slope = math.fsum(a * b for a, b in zip(yr, zr)) / zz
            partial_rho = correlation([a - x_slope * b for a, b in zip(xr, zr)],
                                      [a - y_slope * b for a, b in zip(yr, zr)])
            partial_reason = None if partial_rho is not None else "one residual rank vector is constant"
        return {"n": len(rows), "rho": rho,
                "leave_one_out": {"values": leave_one_out,
                    "min": min(defined) if defined else None,
                    "max": max(defined) if defined else None,
                    "defined_count": len(defined), "same_direction_as_full": same_direction,
                    "direction_stable": (rho is not None and len(defined) == len(rows)
                                         and same_direction == len(rows)),
                    "direction": direction},
                "partial_rank": {"control": "delta_fvs_pp", "rho": partial_rho,
                                 "defined": partial_rho is not None, "reason": partial_reason}}

    workspace = Path(workspace).resolve()
    model_root = workspace / "DiffGDM code/model_test"
    results = model_root / "result/DiffGDM"
    inputs = {}
    fvs = _verified_canonical_graphs(_verified_read(
        results / "decycle_dataset_real_all/graphs_decycle_size.json", inputs))
    candidates = _verified_canonical_graphs(_verified_read(
        results / "decycle_dataset_real_all/graphs_decycle_nodelist.json", inputs))
    final = _verified_canonical_graphs(_verified_read(
        results / "decycling_dismantling_analysis/DiffGDM_graphs_decycledismantle_size_target.json", inputs))
    # Preserve the original paper-domain SHA check, including its Figure 6 values.
    historical = original_controls_payload(workspace, 6)
    inputs.update(historical["input_SHA256"])
    graphs = historical["graphs"]
    methods = ["NC", "Density", "GE"]
    node_keys = {
        "base": "DiffGDM_base",
        "NC": "DiffGDM_D_cost_estimator_natural_connectivity",
        "LCC": "DiffGDM_D_cost_estimator_LCC",
        "GCC": "DiffGDM_D_cost_estimator_global_CC",
        "ACC": "DiffGDM_D_cost_estimator_average_CC",
        "Density": "DiffGDM_D_cost_estimator_density",
        "GE": "DiffGDM_D_cost_estimator_global_efficiency",
    }
    if set(final) != set(graphs) or len(graphs) != 27:
        raise ValueError("Guidance analysis requires the original 27-network domain")
    endpoints, all_forests = {}, {}
    case_networks, case_candidates = {}, {}
    for graph in graphs:
        path = model_root / "dataset/dataset_real_all" / (graph + ".graphml")
        inputs[str(path.resolve())] = _verified_sha(path)
        network = igraph.Graph.Read_GraphML(str(path)).to_networkx()
        network.remove_edges_from(list(nx.selfloop_edges(network)))
        network.remove_nodes_from(list(nx.isolates(network)))
        network = nx.convert_node_labels_to_integers(network)
        if network.is_directed() or network.is_multigraph():
            raise ValueError("Unexpected graph type: " + graph)
        n = network.number_of_nodes()
        endpoints[graph], all_forests[graph] = {}, {}
        if graph in ("econ-wm1", "opsahl-openflights"):
            case_networks[graph], case_candidates[graph] = network, {}
        native = igraph.Graph(n=n, edges=list(network.edges()), directed=False)
        for method, key in node_keys.items():
            raw = candidates[graph][key]
            if (any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < n for v in raw)
                    or len(set(raw)) != len(raw)):
                raise ValueError("Invalid FVS node identity: " + graph + "/" + method)
            selected = frozenset(raw)
            if graph in case_candidates:
                case_candidates[graph][method] = sorted(selected)
            forest = native.induced_subgraph([v for v in range(n) if v not in selected])
            components = forest.connected_components().sizes()
            if not forest.is_simple() or forest.ecount() != forest.vcount() - len(components):
                raise ValueError("Saved candidate does not leave a forest: " + graph + "/" + method)
            fvs_ratio, cfn = fvs[graph]["DiffGDM_" + method], final[graph]["DiffGDM_" + method]
            _original_scalar_cell(fvs_ratio, {})
            _original_scalar_cell(cfn, {})
            if not math.isclose(fvs_ratio, len(raw) / n, rel_tol=0, abs_tol=1e-12):
                raise ValueError("FVS size/node-list mismatch: " + graph + "/" + method)
            final_nodes = round(cfn * n)
            if not math.isclose(cfn, final_nodes / n, rel_tol=0, abs_tol=1e-12):
                raise ValueError("CFN does not represent an integer node count: " + graph + "/" + method)
            endpoints[graph][method] = {"graph": graph, "N": n,
                "fvs_nodes": len(raw), "final_nodes": final_nodes,
                "fvs_ratio": fvs_ratio, "cfn": cfn}
            all_forests[graph][method] = sorted(components, reverse=True)
    points, outcomes, direction_counts, associations = {}, {}, {}, {}
    for method in methods:
        points[method] = []
        for graph in graphs:
            row, base = endpoints[graph][method], endpoints[graph]["base"]
            threshold = (row["N"] + 99) // 100
            oversized = [size for size in all_forests[graph][method] if size > threshold]
            base_oversized = [size for size in all_forests[graph]["base"] if size > threshold]
            delta_trees = len(oversized) - len(base_oversized)
            points[method].append({**row, "base_fvs_nodes": base["fvs_nodes"],
                "base_final_nodes": base["final_nodes"],
                "threshold": threshold, "oversized_trees": len(oversized),
                "base_oversized_trees": len(base_oversized),
                "oversized_sizes": oversized, "base_oversized_sizes": base_oversized,
                "delta_trees": delta_trees, "delta_tree_density": delta_trees / row["N"],
                "delta_fvs_pp": 100 * (row["fvs_nodes"] - base["fvs_nodes"]) / row["N"],
                "delta_cfn_pp": 100 * (row["final_nodes"] - base["final_nodes"]) / row["N"]})
        outcomes[method] = {label: sum((r["final_nodes"] > r["base_final_nodes"])
            - (r["final_nodes"] < r["base_final_nodes"]) == sign for r in points[method])
            for label, sign in (("improved", -1), ("tied", 0), ("worse", 1))}
        direction_counts[method] = {outcome: {change: 0 for change in ("decreased", "tied", "increased")}
                                    for outcome in ("improved", "tied", "worse")}
        for row in points[method]:
            outcome = ("improved" if row["final_nodes"] < row["base_final_nodes"]
                       else "worse" if row["final_nodes"] > row["base_final_nodes"] else "tied")
            change = ("decreased" if row["delta_trees"] < 0
                      else "increased" if row["delta_trees"] > 0 else "tied")
            direction_counts[method][outcome][change] += 1
        if sum(sum(row.values()) for row in direction_counts[method].values()) != 27:
            raise ValueError("Guidance direction-count table does not sum to 27")
        associations[method] = {"raw_count": association(points[method], "delta_trees"),
                                "normalized_count": association(points[method], "delta_tree_density")}
    _verified_hash_guard(inputs)
    payload = {"schema_version": 2, "analysis": "historical-guidance-oversized-tree-association",
        "graphs": graphs, "methods": methods, "points": points, "outcomes": outcomes,
        "direction_counts": direction_counts, "associations": associations,
        "validation": {"networks": 27, "paired_comparisons": 81,
                       "plotted_comparisons": 81, "validated_configurations_per_network": 7,
                       "saved_FVS_forests": 189, "all_FVS_forests_valid": True,
                       "all_endpoint_node_counts_checked": True,
                       "all_direction_tables_sum_to_27": True, "all_input_SHA256_unchanged": True},
        "statistical_unit": "The same 27 networks and base are shared by NC, Density and GE; not 81 independent networks",
        "selection": "Exploratory: these three guides were selected by the earlier six-guide analysis; they are not three independent validations",
        "dispersion": "Historical single-run endpoints only; no seed SD, p-values, confidence intervals or significance tests",
        "threshold_definition": "C=ceil(0.01*N), where N is the original processed graph size; count residual trees strictly larger than C; retain newly isolated nodes",
        "association_definition": "Spearman with average ranks for ties; leave one network out; partial rank correlation uses rank residuals controlling only delta_fvs_pp; raw and N-normalized tree counts reported separately",
        "interpretation": "Descriptive association only; all 27 networks, favourable and unfavourable outcomes, and ties are retained; no causal or universally favourable forest criterion inferred",
        "input_SHA256": inputs}
    legacy_sha = hashlib.sha256(json.dumps(payload, sort_keys=True,
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    cases = [_guidance_component_case(case_id, graph, case_methods,
                case_networks[graph], case_candidates[graph], endpoints[graph], aggregation)
             for case_id, graph, case_methods, aggregation in (
                 ("d", "econ-wm1", ["NC", "Density", "GE"], "identical_components"),
                 ("e", "opsahl-openflights", ["Density", "GE"], "small_components"))]
    _verified_hash_guard(inputs)
    payload.update(schema_version=3, legacy_schema2_SHA256=legacy_sha, component_cases=cases,
        component_case_definition=("Exact static component membership correspondence between "
            "saved base/guided FVS candidates, not a denoising chronology, postprocessing trace "
            "or causal attribution; links contain shared retained original node IDs; pools are "
            "collections of separate components, never single trees"),
        component_case_validation={"case_count": 2, "case_ids": ["d", "e"],
            "all_component_node_accounting_checked": True,
            "all_shared_exclusive_block_conservation_checked": True,
            "all_candidate_identity_equivalences_checked": True,
            "legacy_schema2_fields_unchanged": True, "all_input_SHA256_unchanged": True})
    if include_metric_summary:
        legacy_schema3_sha = hashlib.sha256(json.dumps(payload, sort_keys=True,
            ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
        summary = _guidance_metric_decrease_summary(workspace, payload, all_forests,
                                                      legacy_schema3_sha)
        payload.update(schema_version=4, legacy_schema3_SHA256=legacy_schema3_sha,
                       metric_decrease_summary=summary)
    return payload


def _guidance_metric_decrease_summary(workspace, payload, all_forests, legacy_schema3_sha):
    """Validate an existing exact-audit cache without recomputing NC or GE.

    Ordinary plotting reconstructs only the already saved candidate forests.
    Spectra and shortest paths remain exclusive to the explicit read-only audit.
    """
    from fractions import Fraction

    def semantic_sha(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
            allow_nan=False).encode("utf-8")).hexdigest()

    source = (Path(workspace) / "DiffGDM code/experiments_figure/outputs"
              / "Guidance_Benefit_Analysis_source.json")
    source_sha = _verified_sha(source)
    with source.open(encoding="utf-8-sig") as stream:
        saved = json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("Nonfinite guidance metric-cache value: " + value)))
    cache = saved.get("metric_decrease_summary")
    projection = {key: value for key, value in saved.items()
                  if key not in ("metric_decrease_summary", "legacy_schema3_SHA256")}
    projection["schema_version"] = 3
    if (semantic_sha(projection) != legacy_schema3_sha or not isinstance(cache, dict)
            or cache.get("schema_version") != 1 or cache.get("status") != "COMPLETE"
            or cache.get("legacy_schema3_SHA256") != legacy_schema3_sha
            or cache.get("graphs") != payload["graphs"]
            or cache.get("methods") != payload["methods"]
            or cache.get("input_SHA256") != payload["input_SHA256"]):
        raise ValueError("The exact metric cache does not match the original saved-forest analysis")
    fields = ["graph", "N", "base_metric_value", "metric_value", "delta_metric",
              "metric_direction", "delta_fvs_count", "delta_K", "delta_Q", "delta_L",
              "delta_cfn_pp", "outcome"]
    if (cache.get("row_fields") != fields or set(cache.get("rows", {})) != set(payload["methods"])
            or cache.get("records_SHA256") != semantic_sha(cache["rows"])):
        raise ValueError("The exact metric cache has an invalid record schema or digest")
    evidence = cache.get("audit_evidence", {})
    validation = evidence.get("validation", {})
    if (evidence.get("completed_forests") != 108
            or validation.get("networks") != 27 or validation.get("paired_comparisons") != 81
            or validation.get("original_saved_candidates_validated") != 189
            or validation.get("all_input_SHA256_unchanged") is not True
            or evidence.get("runtime", {}).get("approximation") is not False):
        raise ValueError("The metric cache is not backed by the completed exact 27-network audit")
    directions, outcomes = ("decreased", "unchanged", "increased"), ("improved", "unchanged", "worse")
    counts, decreased_outcomes = {}, {}
    for method in payload["methods"]:
        raw_rows = cache["rows"][method]
        if (len(raw_rows) != 27 or any(not isinstance(row, list) or len(row) != len(fields)
                                     for row in raw_rows)
                or [row[0] for row in raw_rows] != payload["graphs"]):
            raise ValueError("The exact metric cache must retain all 27 networks in the original order")
        points = {row["graph"]: row for row in payload["points"][method]}
        counts[method] = {direction: {outcome: 0 for outcome in outcomes} for direction in directions}
        for raw in raw_rows:
            row = dict(zip(fields, raw))
            graph, point = row["graph"], points[row["graph"]]
            n = point["N"]
            for key in ("N", "base_metric_value", "metric_value", "delta_metric", "delta_fvs_count",
                        "delta_K", "delta_Q", "delta_L", "delta_cfn_pp"):
                if (isinstance(row[key], bool) or not isinstance(row[key], (int, float))
                        or not math.isfinite(row[key])):
                    raise ValueError("Invalid finite metric-cache record: " + graph + "/" + method)
            forest, base = all_forests[graph][method], all_forests[graph]["base"]
            threshold = point["threshold"]
            wanted = {"N": n,
                "delta_fvs_count": point["fvs_nodes"] - point["base_fvs_nodes"],
                "delta_K": point["delta_trees"],
                "delta_Q": (sum(size for size in forest if size > threshold)
                            - sum(size for size in base if size > threshold)) / n,
                "delta_L": (max(forest, default=0) - max(base, default=0)) / n,
                "delta_cfn_pp": point["delta_cfn_pp"]}
            if any(not math.isclose(row[key], value, rel_tol=0, abs_tol=1e-12)
                   for key, value in wanted.items()):
                raise ValueError("Metric-cache graph, forest or endpoint mismatch: " + graph + "/" + method)
            base_value, value = row["base_metric_value"], row["metric_value"]
            if base_value < 0 or value < 0 or (method != "NC" and max(base_value, value) > 1):
                raise ValueError("Metric-cache value outside the metric domain")
            if method == "Density":
                def density_fraction(sizes):
                    remaining = sum(sizes)
                    edges = remaining - len(sizes)
                    return (Fraction(2 * edges, remaining * (remaining - 1))
                            if remaining > 1 else Fraction(0))
                expected_base, expected_value = density_fraction(base), density_fraction(forest)
                if (not math.isclose(base_value, float(expected_base), rel_tol=0, abs_tol=1e-15)
                        or not math.isclose(value, float(expected_value), rel_tol=0, abs_tol=1e-15)):
                    raise ValueError("Cached density differs from the exact saved forest n/m identity")
                exact_delta = expected_value - expected_base
                sign = (exact_delta > 0) - (exact_delta < 0)
                expected_delta = float(exact_delta)
            else:
                raw_delta = value - base_value
                tolerance = 1e-10 * max(1.0, abs(base_value), abs(value))
                sign = 0 if abs(raw_delta) <= tolerance else (raw_delta > 0) - (raw_delta < 0)
                expected_delta = raw_delta if sign else 0.0
            direction = "decreased" if sign < 0 else "increased" if sign > 0 else "unchanged"
            delta_final = point["final_nodes"] - point["base_final_nodes"]
            outcome = "improved" if delta_final < 0 else "worse" if delta_final > 0 else "unchanged"
            if (row["metric_direction"] != direction or row["outcome"] != outcome
                    or not math.isclose(row["delta_metric"], expected_delta, rel_tol=0, abs_tol=1e-15)):
                raise ValueError("Metric-cache direction or CFN classification mismatch: " + graph + "/" + method)
            counts[method][direction][outcome] += 1
        if sum(sum(values.values()) for values in counts[method].values()) != 27:
            raise ValueError("Every metric-direction table must retain all 27 networks")
        selected = counts[method]["decreased"]
        decreased_outcomes[method] = {"decreased_count": sum(selected.values()),
            "counts": {"improved": selected["improved"], "tied": selected["unchanged"],
                       "worse": selected["worse"]}}
    if counts != cache.get("direction_counts"):
        raise ValueError("Metric-cache direction tables differ from all 81 paired records")
    if ("decreased_outcomes" in cache and cache["decreased_outcomes"] != decreased_outcomes):
        raise ValueError("Metric-decreased subset summaries differ from the exact paired counts")
    _verified_hash_guard(payload["input_SHA256"])
    if _verified_sha(source) != source_sha:
        raise ValueError("The metric cache changed while being validated")
    return {**cache, "decreased_outcomes": decreased_outcomes}


def _guidance_audit_ranks(values):
    """Average ranks, with no random tie breaking."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks, start = [0.0] * len(values), 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        for index in order[start:stop]:
            ranks[index] = (start + 1 + stop) / 2
        start = stop
    return ranks


def _guidance_audit_correlation(x, y):
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    xc, yc = [v - mx for v in x], [v - my for v in y]
    xx, yy = math.fsum(v * v for v in xc), math.fsum(v * v for v in yc)
    if xx <= 1e-20 or yy <= 1e-20:
        return None
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(xc, yc))
                         / math.sqrt(xx * yy)))


def _guidance_audit_association(rows, x_field, y_field):
    """A descriptive paired-network association; no p-value is calculated."""
    x, y = ([row[field] for row in rows] for field in (x_field, y_field))
    z = [row["delta_fvs_count"] for row in rows]

    def spearman(a, b):
        return _guidance_audit_correlation(_guidance_audit_ranks(a),
                                            _guidance_audit_ranks(b))

    rho = spearman(x, y)
    loo = [{"omitted_graph": row["graph"],
            "rho": spearman(x[:i] + x[i + 1:], y[:i] + y[i + 1:])}
           for i, row in enumerate(rows)]
    defined = [item["rho"] for item in loo if item["rho"] is not None]
    same_direction = (None if rho is None else sum(
        (value > 0) - (value < 0) == (rho > 0) - (rho < 0) for value in defined))
    ranked = [_guidance_audit_ranks(values) for values in (x, y, z)]
    centered = [[v - math.fsum(values) / len(values) for v in values]
                if values else [] for values in ranked]
    xr, yr, zr = centered
    zz = math.fsum(v * v for v in zr)
    if zz <= 1e-20:
        partial, reason = None, "control ranks are constant"
    else:
        bx = math.fsum(a * b for a, b in zip(xr, zr)) / zz
        by = math.fsum(a * b for a, b in zip(yr, zr)) / zz
        partial = _guidance_audit_correlation([a - bx * b for a, b in zip(xr, zr)],
                                              [a - by * b for a, b in zip(yr, zr)])
        reason = None if partial is not None else "a residual rank vector is constant"
    return {"n": len(rows), "x": x_field, "y": y_field, "rho": rho,
            "defined": rho is not None,
            "reason": None if rho is not None else "fewer than two pairs or constant ranks",
            "leave_one_out": {"values": loo,
                "min": min(defined) if defined else None,
                "max": max(defined) if defined else None,
                "defined_count": len(defined), "same_direction_as_full": same_direction,
                "direction_stable": (rho is not None and len(defined) == len(rows)
                                     and same_direction == len(rows))},
            "partial_rank": {"control": "delta_fvs_count", "rho": partial,
                             "defined": partial is not None, "reason": reason}}


def _guidance_audit_forest_metric(forest, metric, check=None, notify=None):
    """Exact whole-forest NC/density/GE, including isolated retained nodes.

    Eigenvalues are complete within every component. GE visits every ordered
    connected pair in bounded batches; different components contribute zero.
    No model, decoder, TreeBreak, or reinsertion is invoked.
    """
    import numpy as np
    from scipy.linalg import eigvalsh
    from scipy.special import logsumexp

    check = check or (lambda: None)
    notify = notify or (lambda event: None)
    check()
    n = forest.vcount()
    components = forest.connected_components()
    if (not forest.is_simple() or forest.is_directed()
            or forest.ecount() != n - len(components)):
        raise ValueError("Topology audit requires a simple undirected forest")
    if n == 0:
        raise ValueError("Natural connectivity is undefined for an empty residual graph")
    if metric == "Density":
        density = 2 * forest.ecount() / (n * (n - 1)) if n > 1 else 0.0
        identity = 2 * (n - len(components)) / (n * (n - 1)) if n > 1 else 0.0
        if density != identity:
            raise ValueError("Forest density identity failed")
        return density
    if metric not in ("NC", "GE"):
        raise ValueError("Unknown topology metric: " + str(metric))
    if metric == "GE" and n < 2:
        return 0.0
    log_traces, inverse_distance_sums = [], []
    singleton_count = 0
    for index, vertices in enumerate(components):
        check()
        size = len(vertices)
        if size == 1:
            singleton_count += 1
            continue
        component = forest.induced_subgraph(vertices)
        notify({"stage": "component_metric", "metric": metric,
                "component_index": index, "component_nodes": size})
        if metric == "NC":
            adjacency = np.zeros((size, size), dtype=np.float64, order="F")
            edges = np.asarray(component.get_edgelist(), dtype=np.intp)
            adjacency[edges[:, 0], edges[:, 1]] = 1.0
            adjacency[edges[:, 1], edges[:, 0]] = 1.0
            eigenvalues = eigvalsh(adjacency, overwrite_a=True, check_finite=False,
                                  driver="evr")
            del adjacency
            if len(eigenvalues) != size or not np.isfinite(eigenvalues).all():
                raise ValueError("Incomplete or nonfinite adjacency spectrum")
            log_traces.append(float(logsumexp(eigenvalues)))
            del eigenvalues
        else:
            for start in range(0, size, 64):
                check()
                distances = np.asarray(component.distances(
                    source=list(range(start, min(start + 64, size)))), dtype=np.float64)
                if not np.isfinite(distances).all():
                    raise ValueError("A forest component contains unreachable node pairs")
                inverse = np.zeros_like(distances)
                np.divide(1.0, distances, out=inverse, where=distances > 0)
                inverse_distance_sums.append(float(np.sum(inverse, dtype=np.float64)))
                del distances, inverse
        check()
    if metric == "NC":
        if singleton_count:
            log_traces.append(math.log(singleton_count))
        result = float(logsumexp(log_traces) - math.log(n))
    else:
        result = math.fsum(inverse_distance_sums) / (n * (n - 1))
    if not math.isfinite(result):
        raise ValueError("Nonfinite exact forest metric")
    return result


def guidance_topology_audit(workspace, progress=None, *, deadline=None, resource_check=None):
    """Read-only audit of the three selected guides on all 27 saved networks.

    This explicit entrypoint is separate from ordinary plotting. Its caller may
    enforce a 30-minute / 4-GiB / four-thread process budget. A deadline is an
    absolute time.monotonic value. Limit exceptions are never turned into a
    successful result, and no output, manifest, data, or cache is written here.
    """
    import time
    from fractions import Fraction

    started = time.monotonic()
    progress = progress or (lambda event: None)
    completed = 0

    def check():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Exact topology audit exceeded its approved time budget")
        if resource_check is not None:
            resource_check()

    def emit(stage, **details):
        check()
        progress({"stage": stage, "completed_forests": completed, **details})

    workspace = Path(workspace).resolve()
    emit("validate_saved_candidates")
    validated = guidance_benefit_payload(workspace)
    inputs = dict(validated["input_SHA256"])
    graphs, methods = validated["graphs"], validated["methods"]
    model_root = workspace / "DiffGDM code/model_test"
    candidates = _verified_canonical_graphs(_verified_read(model_root /
        "result/DiffGDM/decycle_dataset_real_all/graphs_decycle_nodelist.json", inputs))
    keys = {"base": "DiffGDM_base",
            "NC": "DiffGDM_D_cost_estimator_natural_connectivity",
            "Density": "DiffGDM_D_cost_estimator_density",
            "GE": "DiffGDM_D_cost_estimator_global_efficiency"}
    point_lookup = {method: {row["graph"]: row for row in validated["points"][method]}
                    for method in methods}
    rows = {method: [] for method in methods}
    largest_component = 0
    for graph in graphs:
        emit("load_graph", graph=graph)
        path = model_root / "dataset/dataset_real_all" / (graph + ".graphml")
        if _verified_sha(path) != inputs[str(path.resolve())]:
            raise ValueError("Graph fingerprint changed during topology audit: " + graph)
        network = igraph.Graph.Read_GraphML(str(path)).to_networkx()
        network.remove_edges_from(list(nx.selfloop_edges(network)))
        network.remove_nodes_from(list(nx.isolates(network)))
        network = nx.convert_node_labels_to_integers(network)
        N = network.number_of_nodes()
        native = igraph.Graph(n=N, edges=list(network.edges()), directed=False)
        forests, summaries, metric_cache = {}, {}, {}
        for method, key in keys.items():
            check()
            selected = frozenset(candidates[graph][key])
            forest = native.induced_subgraph([node for node in range(N) if node not in selected])
            sizes = sorted(forest.connected_components().sizes(), reverse=True)
            if (not forest.is_simple() or forest.is_directed()
                    or forest.ecount() != forest.vcount() - len(sizes)):
                raise ValueError("Invalid saved residual forest: " + graph + "/" + method)
            threshold = (N + 99) // 100
            over = [size for size in sizes if size > threshold]
            summaries[method] = {"remaining_nodes": forest.vcount(),
                "remaining_edges": forest.ecount(), "components": len(sizes),
                "fvs_count": len(selected), "K": len(over),
                "oversized_nodes": sum(over), "largest_tree_nodes": max(sizes, default=0)}
            forests[method] = (selected, forest)
            largest_component = max(largest_component, max(sizes, default=0))
            emit("forest_validated", graph=graph, method=method)
        for method in methods:
            values = {}
            for side in ("base", method):
                selected, forest = forests[side]
                identity = (selected, method)
                if identity not in metric_cache:
                    emit("exact_metric", graph=graph, method=method, side=side)
                    metric_cache[identity] = _guidance_audit_forest_metric(forest, method,
                        check=check, notify=lambda event, g=graph, m=method, s=side:
                            emit(event.pop("stage"), graph=g, method=m, side=s, **event))
                values[side] = metric_cache[identity]
            base, guided = summaries["base"], summaries[method]
            point = point_lookup[method][graph]
            if (N != point["N"] or guided["fvs_count"] != point["fvs_nodes"]
                    or base["fvs_count"] != point["base_fvs_nodes"]
                    or guided["K"] != point["oversized_trees"]
                    or base["K"] != point["base_oversized_trees"]):
                raise ValueError("Topology reconstruction disagrees with original scatter data")
            raw_delta = values[method] - values["base"]
            tolerance = (1e-10 * max(1.0, abs(values[method]), abs(values["base"]))
                         if method in ("NC", "GE") else 0.0)
            if method == "Density":
                def density_fraction(summary):
                    n = summary["remaining_nodes"]
                    return Fraction(2 * summary["remaining_edges"], n * (n - 1)) if n > 1 else Fraction(0)
                exact_delta = density_fraction(guided) - density_fraction(base)
                direction = (exact_delta > 0) - (exact_delta < 0)
                raw_delta = float(exact_delta)
            else:
                direction = 0 if abs(raw_delta) <= tolerance else (raw_delta > 0) - (raw_delta < 0)
            delta = raw_delta if direction else 0.0
            final_delta = point["final_nodes"] - point["base_final_nodes"]
            outcome = "improved" if final_delta < 0 else "worse" if final_delta > 0 else "unchanged"
            row = {"graph": graph, "method": method, "N": N, "threshold": (N + 99) // 100,
                "metric": method, "metric_value": values[method], "base_metric_value": values["base"],
                "delta_metric_raw": raw_delta, "delta_metric": delta, "metric_tie_tolerance": tolerance,
                "metric_direction": "decreased" if direction < 0 else "increased" if direction > 0 else "unchanged",
                "forest": guided, "base_forest": base,
                "delta_K": guided["K"] - base["K"],
                "delta_K_per_N": (guided["K"] - base["K"]) / N,
                "delta_oversized_nodes": guided["oversized_nodes"] - base["oversized_nodes"],
                "Q": guided["oversized_nodes"] / N, "base_Q": base["oversized_nodes"] / N,
                "delta_Q": (guided["oversized_nodes"] - base["oversized_nodes"]) / N,
                "L": guided["largest_tree_nodes"] / N, "base_L": base["largest_tree_nodes"] / N,
                "delta_largest_tree_nodes": guided["largest_tree_nodes"] - base["largest_tree_nodes"],
                "delta_L": (guided["largest_tree_nodes"] - base["largest_tree_nodes"]) / N,
                "fvs_count": guided["fvs_count"], "base_fvs_count": base["fvs_count"],
                "delta_fvs_count": guided["fvs_count"] - base["fvs_count"],
                "delta_fvs_ratio": (guided["fvs_count"] - base["fvs_count"]) / N,
                "final_nodes": point["final_nodes"], "base_final_nodes": point["base_final_nodes"],
                "delta_final_nodes": final_delta, "delta_cfn_pp": point["delta_cfn_pp"], "outcome": outcome}
            rows[method].append(row)
            completed += 1
            emit("forest_complete", graph=graph, method=method)
        completed += 1
        emit("forest_complete", graph=graph, method="base")
        del network, native, forests
    emit("paired_network_statistics")
    summaries = {}
    pairs = (("metric_to_K", "delta_metric", "delta_K_per_N"),
             ("metric_to_Q", "delta_metric", "delta_Q"),
             ("metric_to_L", "delta_metric", "delta_L"),
             ("metric_to_CFN", "delta_metric", "delta_cfn_pp"),
             ("K_to_CFN", "delta_K_per_N", "delta_cfn_pp"),
             ("Q_to_CFN", "delta_Q", "delta_cfn_pp"),
             ("L_to_CFN", "delta_L", "delta_cfn_pp"))
    for method in methods:
        check()
        selected = rows[method]
        if len(selected) != 27 or [row["graph"] for row in selected] != graphs:
            raise ValueError("Topology audit lost a paired network")
        counts = {direction: {outcome: 0 for outcome in ("improved", "unchanged", "worse")}
                  for direction in ("decreased", "unchanged", "increased")}
        for row in selected:
            counts[row["metric_direction"]][row["outcome"]] += 1
        if sum(sum(values.values()) for values in counts.values()) != 27:
            raise ValueError("Topology direction table does not sum to 27")
        equal_fvs = [row for row in selected if row["delta_fvs_count"] == 0]
        summaries[method] = {"n": 27, "metric_direction_counts": counts,
            "associations": {name: _guidance_audit_association(selected, x, y) for name, x, y in pairs},
            "equal_fvs_subset": {"n": len(equal_fvs), "graphs": [row["graph"] for row in equal_fvs],
                "outcomes": {outcome: sum(row["outcome"] == outcome for row in equal_fvs)
                             for outcome in ("improved", "unchanged", "worse")},
                "rows": equal_fvs,
                "associations": {name: _guidance_audit_association(equal_fvs, x, y) for name, x, y in pairs},
                "interpretation": "Small descriptive subset; retain ties and undefined associations; not a separate validation"}}
    check()
    _verified_hash_guard(inputs)
    return {"schema_version": 1, "analysis": "historical-guidance-topology-audit", "status": "COMPLETE",
        "graphs": graphs, "methods": methods, "completed_forests": completed, "uncompleted": [],
        "rows": rows, "summaries": summaries, "input_SHA256": inputs,
        "validation": {"networks": 27, "paired_comparisons": 81, "reconstructed_forests": 108,
            "original_saved_candidates_validated": 189, "all_forests_valid": True,
            "all_graph_and_candidate_identities_checked": True, "all_direction_tables_sum_to_27": True,
            "newly_isolated_nodes_retained": True, "density_forest_identity_checked": True,
            "all_input_SHA256_unchanged": True},
        "definitions": {"NC": "log(sum(exp(lambda_j))/n), complete adjacency eigenvalues of every residual component merged by log-sum-exp",
            "Density": "2*m/(n*(n-1)); in a forest m=n-components; zero for n<2",
            "GE": "sum_{i!=j}(1/d_ij)/(n*(n-1)); unreachable pairs contribute zero; exact connected-pair distances; zero for n<2",
            "residual_denominator": "n is every retained residual node, including newly isolated nodes",
            "forest_distribution_denominator": "N is the original processed graph size, not residual n",
            "K": "count of residual trees strictly larger than C=ceil(0.01*N); correlations use delta_K/N",
            "Q": "sum of nodes in residual trees strictly larger than C, divided by N",
            "L": "largest residual tree size divided by N",
            "changes": "guided minus the same base on each network; CFN decrease is improvement",
            "ties": "NC/GE: abs(delta)<=1e-10*max(1,abs(base),abs(guided)); delta_metric is zero for such ties and raw delta is also retained. Density direction uses exact rational comparison. K,Q,L,FVS and CFN directions use integer counts",
            "association": "Spearman average ranks with ties, all leave-one-network-out values, and partial rank residual correlation controlling only delta_fvs_count; no p-values",
            "statistical_unit": "The same 27 networks and base are shared by the three guides; not 81 independent networks",
            "boundary": "Historical saved single-run endpoints; no seed statistics, inference trajectory, postprocessing replay, causal attribution or inferred best original-network topology"},
        "runtime": {"elapsed_seconds": time.monotonic() - started,
                    "largest_component_nodes": largest_component,
                    "GE_source_batch_size": 64, "approximation": False}}


def original_controls_payload(workspace, figure_number):
    """Read historical paper inputs plus authenticated single-run controls.

    This explicit profile never reads a ten-seed average, executes inference,
    prepares legacy analysis files, or auto-selects a paired Figure 6 release.
    """
    workspace, figure_number = Path(workspace).resolve(), int(figure_number)
    if figure_number not in (2, 3, 6):
        raise ValueError("original-plus-controls only defines Figures 2, 3 and 6")
    inputs = {}
    model_root = workspace / "DiffGDM code/model_test"
    results = model_root / "result/DiffGDM"
    analysis = results / "decycling_dismantling_analysis"
    preservation = _verified_read(
        workspace / "server_result_import/20261010_local_preservation/before_manifest.json", inputs)
    filenames = ("synth_graphs_decycle_size.json", "synth_graphs_dismantling_size.json",
                 "graphs_decycle_size.json", "graphs_coredismantle_size.json",
                 "graphs_onepassdismantle_size.json", "DiffGDM_graphs_decycle_size.json",
                 "DiffGDM_graphs_decycledismantle_size_target.json")
    legacy = {}
    for filename in filenames:
        path = analysis / filename
        legacy[filename] = _verified_canonical_graphs(_verified_read(path, inputs))
        preserved = preservation["files"]["result/DiffGDM/decycling_dismantling_analysis/" + filename]
        if inputs[str(path.resolve())] != preserved["sha256"]:
            raise ValueError("Original paper input changed: " + filename)
    synth = list(legacy[filenames[0]])
    real = list(legacy[filenames[2]])
    if len(synth) != 140 or len(real) != 27:
        raise ValueError("Original paper domain must be 140 synthetic / 27 real graphs")
    if set(legacy[filenames[1]]) != set(synth) or any(
            set(legacy[name]) != set(real) for name in filenames[2:]):
        raise ValueError("Original figure graph domains disagree")
    common = {"profile": "original-plus-controls", "figure": figure_number,
              "dispersion": "No inferred SD: historical scalars and selected single runs only",
              "average_rule": "Equal graph-weighted mean of the displayed graph-level values; no SD inferred",
              "input_SHA256": inputs}

    if figure_number == 6:
        orders = {"a": ["DiffGDM_GCC", "DiffGDM_ACC", "DiffGDM_NC", "DiffGDM_Density",
                         "DiffGDM_LCC", "DiffGDM_base"],
                  "b": ["DiffGDM_NC", "DiffGDM_LCC", "DiffGDM_Density", "DiffGDM_GE",
                         "DiffGDM_ACC", "DiffGDM_GCC", "DiffGDM_base"]}
        panels = {}
        for key, filename in zip("ab", filenames[-2:]):
            source = legacy[filename]
            if any(list(source[g]) != orders[key] for g in real):
                raise ValueError("Original Figure 6 method order changed")
            values = {g: {m: _original_scalar_cell(source[g][m], {"source_file": filename,
                          "source_method": m}) for m in orders[key]} for g in real}
            panels[key] = {"methods": orders[key], "graph_values": values,
                           "metric": "raw_fvs_ratio" if key == "a" else "cfn",
                           "average_statistics": _original_average_statistics(real, orders[key], values)}
        _verified_hash_guard(inputs)
        return {**common, "graphs": real, "panels": panels, "paired_publication": False,
                "cell_counts": {"historical_scalar_cells": 351, "single_run_cells": 0}}

    manifest_path = results / "dismantle_dataset_real_all/single_run_controls.json"
    manifest = _verified_read(manifest_path, inputs)
    if (manifest.get("status") != "PUBLISHED" or manifest.get("expected_records") != 248
            or manifest.get("expected_raw_fvs") != 194
            or manifest.get("source_sync", {}).get("verified") is not True):
        raise ValueError("Single-run controls are not a complete local PUBLISHED 248/194 release")
    acceptance = manifest.get("acceptance", {})
    if (acceptance.get("structural_acceptance") is not True or acceptance.get("records") != 248
            or acceptance.get("raw_fvs") != 194):
        raise ValueError("Single-run controls lack their complete structural acceptance receipt")
    # The publication adapter independently authenticates old/new endpoints and
    # unchanged non-control projections, rather than blessing current SHA values.
    approved = single_controls_authorized_hashes(results, {})
    if not approved:
        raise ValueError("Single-run controls have no authenticated publication endpoints")
    active = {}
    for dataset in ("synth", "real_all"):
        active[dataset] = {}
        for key, folder, filename in (
                ("cfn", "dismantle", "graphs_dismantle_size_target_1_True.json"),
                ("remove_nodes", "dismantle", "graphs_dismantle_nodelist_target_1_True.json"),
                ("time_model_seconds", "dismantle", "graphs_dismantle_time_target_1_True.json"),
                ("raw_fvs_ratio", "decycle", "graphs_decycle_size.json"),
                ("raw_fvs_nodes", "decycle", "graphs_decycle_nodelist.json")):
            path = (results / (folder + "_dataset_" + dataset) / filename).resolve()
            active[dataset][key] = _verified_canonical_graphs(_verified_read(path, inputs))
            if approved.get(str(path)) != inputs[str(path)]:
                raise ValueError("Active control endpoint is not publication-authorized: " + str(path))
    expected = {("synth", g, "Random-Decycle") for g in synth}
    expected.update(("real_all", g, method) for g in real for method in
                    ("Random-Decycle", "random_rein", "T2T_decycle", "T2T_direct"))
    records, fvs_count = {}, 0
    selected = manifest.get("records")
    if not isinstance(selected, list) or len(selected) != 248:
        raise ValueError("Single-run selected records must contain exactly 248 entries")
    for record in selected:
        identity = (record.get("dataset"), record.get("graph"), record.get("method"))
        if identity not in expected or identity in records or record.get("status") != "success":
            raise ValueError("Duplicate, out-of-domain or unsuccessful single-run record: " + str(identity))
        dataset, graph, method = identity
        count, edges = record.get("node_count"), record.get("edge_count")
        if type(count) is not int or count <= 0 or type(edges) is not int or edges < 0:
            raise ValueError("Invalid single-run graph dimensions")
        nodes = record.get("remove_nodes")
        if (not isinstance(nodes, list) or any(type(n) is not int for n in nodes)
                or len(nodes) != len(set(nodes)) or any(not 0 <= n < count for n in nodes)):
            raise ValueError("Invalid single-run removal set")
        _original_scalar_cell(record.get("cfn"), {})
        if not math.isclose(record["cfn"], len(nodes) / count, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Single-run CFN denominator does not match removal set")
        if (type(record.get("final_lcc")) is not int or type(record.get("lcc_threshold")) is not int
                or not 0 <= record["final_lcc"] <= record["lcc_threshold"] <= count):
            raise ValueError("Single-run final LCC violates its recorded threshold")
        for timing in ("time_model_seconds", "time_wall_seconds", "instrumentation_seconds"):
            value = record.get(timing)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Missing/nonfinite single-run timing: " + timing)
        for state in ("rng_before", "rng_after"):
            if not isinstance(record.get(state), dict) or not record[state]:
                raise ValueError("Missing recorded RNG state: " + state)
        fingerprint = record.get("fingerprint", {})
        if any(not fingerprint.get(key) for key in
               ("graph_sha256", "source_sha256", "native_sha256")):
            raise ValueError("Incomplete single-run source identity")
        if method.startswith("T2T_"):
            if (record.get("seed") != 123 or record.get("t2t_profile") not in ("P4S1", "P2S2")
                    or fingerprint.get("checkpoint_sha256") !=
                    "ff95893d15aba3a04fd014d55317dd9b113f08439524b5b904f4612dbdd9d5bf"
                    or not isinstance(record.get("effective_net_param"), dict)
                    or not record["effective_net_param"]):
                raise ValueError("T2T single-run configuration is not recorded")
        elif record.get("seed") is not None:
            raise ValueError("Random must follow the recorded original RNG state, not a new forced seed")
        fields = ["cfn", "remove_nodes", "time_model_seconds"]
        if method in ("Random-Decycle", "T2T_decycle"):
            raw_nodes = record.get("raw_fvs_nodes")
            if (not isinstance(raw_nodes, list) or any(type(n) is not int for n in raw_nodes)
                    or len(raw_nodes) != len(set(raw_nodes))
                    or any(not 0 <= n < count for n in raw_nodes)):
                raise ValueError("Missing/invalid same-call raw FVS")
            _original_scalar_cell(record.get("raw_fvs_ratio"), {})
            if not math.isclose(record["raw_fvs_ratio"], len(raw_nodes) / count, rel_tol=0, abs_tol=1e-12):
                raise ValueError("Raw FVS denominator mismatch")
            fields += ["raw_fvs_ratio", "raw_fvs_nodes"]
            fvs_count += 1
        for field in fields:
            if active[dataset][field].get(graph, {}).get(method) != record[field]:
                raise ValueError("Published active key disagrees with selected record: " + str((identity, field)))
        records[identity] = record
    if set(records) != expected or fvs_count != 194:
        raise ValueError("Incomplete single-run graph/method or raw-FVS coverage")

    def control_cell(dataset, graph, method, metric):
        record = records[(dataset, graph, method)]
        return _original_scalar_cell(record[metric], {"source_file": str(manifest_path),
             "dataset": dataset, "graph": graph, "source_method": method, "metric": metric,
             "seed": record["seed"], "fingerprint": record["fingerprint"],
             "t2t_profile": record.get("t2t_profile")}, single=True)

    if figure_number == 2:
        methods = ["MinSum", "BPD", "CoreHD", "CoreDQN", "DiffGDM", "Random-Decycle"]
        families = {label: [g for g in synth if g.startswith(prefix)] for label, prefix in
                    (("BA", "barabasi"), ("ER", "gnp"), ("WS", "ws"), ("PL", "static"))}
        if any(len(names) != 35 for names in families.values()) or set(sum(families.values(), [])) != set(synth):
            raise ValueError("Original synthetic family membership changed")
        panels = {}
        for panel, filename, metric in zip(("fig2a", "fig2b"), filenames[:2], ("raw_fvs_ratio", "cfn")):
            values = {g: {m: _original_scalar_cell(legacy[filename][g][m], {
                "source_file": filename, "source_method": m}) for m in methods[:-1]} for g in synth}
            for graph in synth:
                values[graph]["Random-Decycle"] = control_cell("synth", graph, "Random-Decycle", metric)
            panels[panel] = {"metric": metric, "graph_values": values,
                            "family_means": {m: {f: math.fsum(values[g][m]["mean"] for g in names) / len(names)
                                              for f, names in families.items()} for m in methods}}
        payload = {**common, "graphs": synth, "methods": methods, "family_graphs": families,
                   "panels": panels, "cell_counts": {"historical_scalar_cells": 1400, "single_run_cells": 280}}
    else:
        orders = {"a": ["MinSum", "BPD", "DiffGDM", "GFD", "Random-Decycle", "T2T-FVS"],
                  "b": ["DiffGDM_ens", "GFD", "MinSum", "BPD", "DiffGDM", "Random-Decycle", "T2T_decycle"],
                  "c": ["DiffGDM", "GDM", "FINDER", "CI", "DomiRank", "Random_direct", "T2T_direct"]}
        mapping = {"Random-Decycle": "Random-Decycle", "T2T-FVS": "T2T_decycle",
                   "T2T_decycle": "T2T_decycle", "Random_direct": "random_rein", "T2T_direct": "T2T_direct"}
        panels = {}
        for panel, filename in zip("abc", filenames[2:5]):
            metric = "raw_fvs_ratio" if panel == "a" else "cfn"
            values = {g: {m: (control_cell("real_all", g, mapping[m], metric) if m in mapping else
                              _original_scalar_cell(legacy[filename][g][m], {"source_file": filename,
                                  "source_method": m})) for m in orders[panel]} for g in real}
            panels[panel] = {"metric": metric, "methods": orders[panel], "graph_values": values,
                            "average_statistics": _original_average_statistics(real, orders[panel], values)}
        payload = {**common, "graphs": real, "panels": panels,
                   "cell_counts": {"historical_scalar_cells": 378, "single_run_cells": 162}}
    _verified_hash_guard(inputs)
    return payload


def original_controls_numeric_guard(fig, payload, figure_number):
    """Check every scalar label/bar; a single run must never acquire an SD."""
    figure_number = int(figure_number)
    if figure_number == 2:
        return verified_figure_numeric_guard(fig, payload, "2")
    if figure_number == 6:
        from matplotlib.colors import to_rgba
        artists = getattr(fig, "_verified_numeric_artists", ())
        if len(artists) != 364 or len(fig._verified_data_axes) != 2:
            raise ValueError("Historical Figure 6 requires 351 scalar and 13 explicit Average cells")
        identities, minima = set(), 0
        for item in artists:
            panel = payload["panels"][item["panel"]]
            values = (panel["average_statistics"] if item["graph"] == "Average"
                      else panel["graph_values"][item["graph"]])
            identity = (item["panel"], item["graph"], item["method"])
            if identity in identities:
                raise ValueError("Duplicate historical guidance heatmap cell")
            identities.add(identity)
            value = values[item["method"]]["mean"]
            lowest = value == min(cell["mean"] for cell in values.values())
            artist = item["artist"]
            row = 27 if item["graph"] == "Average" else payload["graphs"].index(item["graph"])
            col = panel["methods"].index(item["method"])
            if (artist.get_text() != f"{100 * value:.2f}"
                    or artist.get_position() != (col + .5, row + .5)
                    or (to_rgba(artist.get_color()) == to_rgba("red")) != lowest):
                raise ValueError("Historical Figure 6 scalar label, position or exact minimum mismatch")
            minima += lowest
        expected = {(key, graph, method) for key, panel in payload["panels"].items()
                    for graph in [*payload["graphs"], "Average"] for method in panel["methods"]}
        if identities != expected:
            raise ValueError("Historical guidance heatmap coverage changed")
        for key, axis in zip("ab", fig._verified_data_axes):
            if [label.get_text() for label in axis.get_xticklabels()] != [
                    method.removeprefix("DiffGDM_") for method in payload["panels"][key]["methods"]]:
                raise ValueError("Historical guidance method order changed")
        return {"network_cells": 351, "Average_cells": 13, "SD_labels": 0,
                "paired_publication": False, "all_scalar_labels_checked": True,
                "all_364_labels_checked": True, "Average_is_explicit_row": True,
                "all_unrounded_tied_minima_highlighted": True, "highlighted_cells": minima}
    artists = getattr(fig, "_verified_numeric_artists", ())
    if len(artists) != 560:
        raise ValueError("Original-plus-controls Figure 3 requires 540 cells and 20 Average cells")
    for item in artists:
        panel = payload["panels"][item["panel"]]
        cell = (panel["average_statistics"][item["method"]] if item["graph"] == "Average"
                else panel["graph_values"][item["graph"]][item["method"]])
        if item["artist"].get_text() != f"{100 * cell['mean']:.2f}":
            raise ValueError("Original-plus-controls Figure 3 scalar label mismatch")
    return {"network_cells": 540, "Average_cells": 20, "SD_labels": 0,
            "all_560_labels_checked": True}


def verified_figure3_payload(workspace, inputs, merge_audit):
    """Read accepted sources; never rerun inference, decoding, loading or merging."""
    imports = workspace / "server_result_import"
    model_root = workspace / "DiffGDM code/model_test"
    results = model_root / "result/DiffGDM"
    old = _verified_read(imports / "20261010_figures_3_qafix/Figure3_source_data.json", inputs)
    _verified_sha_input = imports / "20261010_figures_3_qafix/figure_source_data.csv"
    inputs[str(_verified_sha_input.resolve())] = _verified_sha(_verified_sha_input)
    control = _verified_read(imports / "20261011_mvc_run/supervision_control.json", inputs)
    accepted_t2t = control["t2t_resampling"]["delivery"]
    if (accepted_t2t["validation"]["status"] != "complete"
            or accepted_t2t["merge"]["post_merge_independent_audit"]["status"] != "passed"
            or inputs[str((imports / "20261011_mvc_run/t2t4/merge/merge_audit.json").resolve())]
            != accepted_t2t["validation"]["audit_SHA256"]):
        raise ValueError("T2T accepted merge identity or independent acceptance changed")
    order = control["phases"]["figures"]["figure3_method_order"]
    if tuple(len(order[p]) for p in "abc") != (6, 7, 6):
        raise ValueError("Author-approved Figure 3 order must be 6/7/6 methods")
    legacy = {}
    for panel, name in zip("abc", ("graphs_decycle_size.json", "graphs_coredismantle_size.json",
                                   "graphs_onepassdismantle_size.json")):
        path = results / "decycling_dismantling_analysis" / name
        legacy[panel] = _verified_canonical_graphs(_verified_read(path, inputs))
        if inputs[str(path.resolve())] != control["phases"]["figures"]["legacy_plot_input_SHA256"][name]:
            raise ValueError("Retained legacy plot input changed: " + name)

    mvc_audit = _verified_read(imports / "20261011_mvc_run/merge/merge_audit.json", inputs)
    mvc_source = imports / "20261011_mvc_run/download/mvc"
    t2t_source = imports / "20261011_mvc_run/t2t4/download/t2t4"
    records = {}
    for name, source, audit in (("mvc", mvc_source, mvc_audit), ("t2t4", t2t_source, merge_audit)):
        evidence = audit["shared_validation"]
        if audit["status"] != "complete" or evidence["valid_records"] != 540:
            raise ValueError("A source phase has not passed its original acceptance")
        for filename, expected_sha in evidence["source_metadata_sha256"].items():
            value = _verified_read(source / filename, inputs)
            if inputs[str((source / filename).resolve())] != expected_sha:
                raise ValueError("Source metadata differs from accepted audit: " + filename)
            if filename == "records.json":
                records[name] = value
    # Execution history is an independently accepted layer, not part of the frozen loader.
    for name in ("resume_gpu1_manifest.json", "resume_gpu1_before.tar.gz"):
        path = t2t_source / name
        inputs[str(path.resolve())] = _verified_sha(path)
        expected_sha = (accepted_t2t["independent_resume_history_acceptance"]["final_manifest_sha256"]
                        if name == "resume_gpu1_manifest.json" else
                        control["t2t_resampling"]["gpu1_formal_resume"]["before_archive_SHA256"])
        if inputs[str(path.resolve())] != expected_sha:
            raise ValueError("Independently accepted GPU resume history changed: " + name)
    node_counts = {}
    for record in records["mvc"]["records"].values():
        graph, count = record["graph"], record["node_count"]
        if record["status"] != "success" or type(count) is not int or count <= 0:
            raise ValueError("Invalid accepted MVC node count")
        if graph in node_counts and node_counts[graph] != count:
            raise ValueError("Inconsistent MVC node count: " + graph)
        node_counts[graph] = count
    if set(node_counts) != set(old["graphs"]) or len(old["graphs"]) != 27:
        raise ValueError("Figure 3 source graph domain mismatch")
    for record in records["t2t4"]["records"].values():
        if record["node_count"] != node_counts[record["graph"]]:
            raise ValueError("MVC/T2T graph preprocessing dimensions differ")

    data = {}
    for name, directory, audit in (("merged", results / "real_seed_variance", mvc_audit),
                                    ("t2t4", results / "real_seed_variance", merge_audit)):
        data[name] = {}
        approved = {entry["target"]: entry["new_sha256"] for entry in audit["transaction"]}
        approved = verified_result_hashes(directory, approved)
        for metric, filename in (("cfn", "seed_graphs_dismantle_size_target_1_True.json"),
                                 ("raw_fvs_ratio", "raw_fvs_seed_target_1_True.json")):
            path = (directory / filename).resolve()
            data[name][metric] = _verified_canonical_graphs(_verified_read(path, inputs))
            if inputs[str(path)] != approved[str(path)]:
                raise ValueError("Merged plotting data changed: " + filename)
    # The authenticated default ensemble is a derived sidecar, not a rewrite of
    # the protected historical tables or the paired Figure 6 publication.
    import sys
    if str(workspace.resolve()) not in sys.path:
        sys.path.insert(0, str(workspace.resolve()))
    from experiment_ops.seven_guidance_ensemble import current_ensemble_snapshot
    ensemble = current_ensemble_snapshot(results / "real_seed_variance")
    ensemble_directions = ["MVC", "LCC", "GCC", "ACC", "GE", "Density", "NC"]
    if ensemble is not None:
        expected_keys = {graph + "\0" + str(seed) for graph in old["graphs"] for seed in range(10)}
        if (ensemble.get("run_id") != "ens7_20261021"
                or len(ensemble.get("graphs", [])) != 27
                or set(ensemble["graphs"]) != set(old["graphs"])
                or set(ensemble.get("records", {})) != expected_keys):
            raise ValueError("Default ensemble requires the accepted 27-network ten-seed domain")
        for graph in old["graphs"]:
            for seed in range(10):
                record = ensemble["records"][graph + "\0" + str(seed)]
                if (record.get("graph") != graph or type(record.get("seed")) is not int
                        or record["seed"] != seed
                        or record.get("selected_direction") not in ensemble_directions):
                    raise ValueError("Default ensemble record identity or selected direction mismatch")
        if not isinstance(ensemble.get("input_SHA256"), dict) or not ensemble["input_SHA256"]:
            raise ValueError("Default ensemble lacks authenticated input hashes")
        for path, digest in ensemble["input_SHA256"].items():
            if (not isinstance(path, str) or not Path(path).is_absolute()
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)
                    or (path in inputs and inputs[path] != digest)):
                raise ValueError("Default ensemble input hash identity conflicts with plotting sources")
            inputs[path] = digest
    route = {"a": "decycle", "b": "decycle", "c": "direct"}
    panels = {}
    counts = {"ten_seed_cells": 0, "historical_scalar_cells": 0}
    for panel in "abc":
        metric = "raw_fvs_ratio" if panel == "a" else "cfn"
        graph_values = {}
        for graph in old["graphs"]:
            graph_values[graph] = {}
            for label in order[panel]:
                method, group = label, "merged"
                if label == "DiffGDM":
                    method = "DiffGDM_MVC_" + route[panel]
                elif label in {"T2T-FVS", "T2T_decycle", "T2T_direct"}:
                    method, group = "T2T_" + route[panel], "t2t4"
                if panel == "b" and label == "DiffGDM_ens" and ensemble is not None:
                    selected = [ensemble["records"][graph + "\0" + str(seed)] for seed in range(10)]
                    values = [record["cfn"] for record in selected]
                    trace = {"source_group": "default_ensemble_sidecar", "source_method": label,
                             "source": "experiment_ops.seven_guidance_ensemble.current_ensemble_snapshot",
                             "run_id": ensemble["run_id"], "metric": metric,
                             "seeds": list(range(10)), "node_count": node_counts[graph],
                             "selection": {"criterion": "minimum final removal count",
                                           "direction_order": ensemble_directions,
                                           "tie_rule": "first in canonical order; strict less-than update"},
                             "selected_directions": [record["selected_direction"] for record in selected]}
                    cell = _verified_ten_seed_cell(values, trace)
                    counts["ten_seed_cells"] += 1
                elif label in {"DiffGDM", "DiffGDM_ens", "Random-Decycle", "T2T-FVS", "T2T_decycle", "T2T_direct"}:
                    leaf = data[group][metric][graph]
                    values = [leaf[method + "_" + str(seed)] for seed in range(10)]
                    if metric == "raw_fvs_ratio":
                        if any(not isinstance(value, list) for value in values):
                            raise ValueError("Expected original raw FVS node lists")
                        values = [len(value) / node_counts[graph] for value in values]
                    trace = {"source_group": group, "source_method": method, "metric": metric,
                             "seeds": list(range(10)), "node_count": node_counts[graph]}
                    if group == "t2t4":
                        trace["profile"] = records["t2t4"]["network_modes"][graph]["profile"]
                    cell = _verified_ten_seed_cell(values, trace)
                    counts["ten_seed_cells"] += 1
                else:
                    source_label = "Domirank" if label == "DomiRank" and label not in legacy[panel][graph] else label
                    value = legacy[panel][graph][source_label]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("Invalid retained historical scalar")
                    cell = {"mean": value, "sd": None, "variance": None, "seed_num": None,
                            "sampling": "historical graph-level result; no inferred SD",
                            "source_trace": {"source_group": "legacy", "panel": panel,
                                             "source_method": source_label, "metric": metric}}
                    counts["historical_scalar_cells"] += 1
                graph_values[graph][label] = cell
        panels[panel] = {"methods": list(order[panel]), "metric": metric, "graph_values": graph_values,
                         "average_statistics": _verified_average_statistics(
                             old["graphs"], panel, order[panel], graph_values)}
    if counts != {"ten_seed_cells": 243, "historical_scalar_cells": 270}:
        raise ValueError("Figure 3 requires 513 cells: 243 ten-seed and 270 scalar cells")
    return {**old, "panels": panels, "cell_counts": counts,
            "new_cells": "mean +/- sample SD of ten inference seeds, ddof=1",
            "average_rule": "Equal network mean of per-network means; eligible columns use the mean of 27 within-network ddof=1 seed variances (fraction squared), with its square root displayed as RMS within-network SD (percentage points), not SEM or uncertainty of the grand mean"}


def verified_figure_numeric_guard(fig, payload, figure):
    """Bind actual rendered artists to every intended mean and sample SD."""
    axes = fig._verified_data_axes
    if figure == "2":
        total = 0
        for panel, ax in zip(("fig2a", "fig2b"), axes):
            expected = [payload["panels"][panel]["family_means"][method][family]
                        for method in payload["methods"] for family in payload["family_graphs"]]
            if len(ax.patches) != len(expected) or any(not math.isclose(bar.get_height(), value,
                    rel_tol=0, abs_tol=1e-14) for bar, value in zip(ax.patches, expected)):
                raise ValueError("Figure 2 rendered bars differ from unchanged source values")
            total += len(expected)
        return {"all_source_values_unchanged": True, "rendered_bars_checked": total}
    wanted = {}
    average_sd_count = 0
    for panel, details in payload["panels"].items():
        summaries = _verified_average_statistics(payload["graphs"], panel, details["methods"],
                                                  details["graph_values"])
        if details["average_statistics"] != summaries:
            raise ValueError("Recorded Average variances differ from the complete ten-seed sources")
        for method in details["methods"]:
            for graph in payload["graphs"]:
                cell = details["graph_values"][graph][method]
                text = "{:.2f}".format(100 * cell["mean"])
                if cell["sd"] is not None:
                    text += "\n±{:.2f}".format(100 * cell["sd"])
                wanted[(panel, graph, method)] = text
            summary = summaries[method]
            text = "{:.2f}".format(100 * summary["mean"])
            if summary["rms_within_network_sd"] is not None:
                text += "\n±{:.2f}".format(100 * summary["rms_within_network_sd"])
                average_sd_count += 1
            wanted[(panel, "Average", method)] = text
    actual = {}
    for item in fig._verified_numeric_artists:
        key = item["panel"], item["graph"], item["method"]
        if key in actual:
            raise ValueError("Duplicate numeric artist identity")
        actual[key] = item["artist"].get_text()
    if actual != wanted or len(actual) != 532 or average_sd_count != 9:
        raise ValueError("Rendered means/SD/Average do not match all 532 intended labels")
    return {"cells_checked": 513, "numeric_artists_checked": 532,
            "ten_seed_cells": 243, "historical_scalar_cells": 270,
            "average_labels_with_rms_sd": 9, "average_labels_without_sd": 10,
            "SD_labels": 252, "average_variance_ddof": 1}


def verified_figure_geometry_guard(fig):
    """Check the final renderer, without making a second preview or QA file."""
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer, canvas = fig.canvas.get_renderer(), fig.bbox
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < canvas.x0 - 1 or box.y0 < canvas.y0 - 1 or box.x1 > canvas.x1 + 1 or box.y1 > canvas.y1 + 1:
            raise ValueError("Visible text exceeds the figure canvas: " + text.get_text())
    boxes = []
    for item in getattr(fig, "_verified_numeric_artists", ()):
        artist = item["artist"]
        box, axis = artist.get_window_extent(renderer), artist.axes.bbox
        if box.x0 < axis.x0 or box.x1 > axis.x1 or box.y0 < axis.y0 or box.y1 > axis.y1:
            raise ValueError("A numeric mean/SD label exceeds its data panel")
        if any(ax is artist.axes and box.overlaps(other) for ax, other in boxes):
            raise ValueError("Numeric mean/SD labels overlap in the final renderer")
        boxes.append((artist.axes, box))
    return {"canvas_text_bounds_checked": True, "numeric_overlap_count": 0,
            "numeric_panel_bounds_checked": len(boxes), "dpi": fig.dpi,
            "figure_size_inches": list(fig.get_size_inches()), "visual_review_required": True}


def run_verified_figures(workspace_root, figure="all", output_dir=None, overwrite=False):
    """Reproduce verified figures, optionally into the isolated paper-figure outputs.

    With no output_dir, retain the existing PDF/preview and provenance revision
    behavior. The isolated branch returns its audits to the caller and never
    rewrites the accepted figure files or their provenance records.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    from analysis_visi.LCC_size_removed_nodes import (
        draw_verified_synthetic_comparison, draw_verified_route_comparison)
    workspace = Path(workspace_root).resolve()
    isolated_output = None
    if output_dir is not None:
        isolated_output = Path(output_dir).resolve()
        allowed_output = (workspace / "DiffGDM code/experiments_figure/outputs").resolve()
        if isolated_output != allowed_output and allowed_output not in isolated_output.parents:
            raise ValueError("Independent figure export is restricted to experiments_figure/outputs")
        if isolated_output.exists() and not isolated_output.is_dir():
            raise ValueError("The independent output directory is not a directory")
    imports = workspace / "server_result_import"
    inputs = {}
    audit_path = imports / "20261011_mvc_run/t2t4/merge/merge_audit.json"
    audit = _verified_read(audit_path, inputs)
    protected = dict(audit["protected_result_sha256"])
    if len(protected) != 5818 or len(audit["transaction"]) != 7:
        raise ValueError("Unexpected frozen result-protection manifest")
    for entry in audit["transaction"]:
        protected[entry["target"]] = entry["new_sha256"]
    protected = verified_result_hashes(
        workspace / "DiffGDM code/model_test/result/DiffGDM/real_seed_variance", protected)
    protected_before = _verified_hash_guard(protected)
    for path in (Path(__file__).resolve(), Path(__file__).parent / "analysis_visi/LCC_size_removed_nodes.py"):
        inputs[str(path.resolve())] = _verified_sha(path)
    jobs = []
    if figure in {"2", "all"}:
        directory = imports / "20261010_figures_2"
        payload = _verified_read(directory / "Figure2_source_data.json", inputs)
        path = directory / "figure_source_data.csv"
        inputs[str(path.resolve())] = _verified_sha(path)
        jobs.append(("2", directory, "Figure2_Random_Decycle", payload, draw_verified_synthetic_comparison))
    if figure in {"3", "all"}:
        payload = verified_figure3_payload(workspace, inputs, audit)
        jobs.append(("3", imports / "20261010_figures_3_qafix", "Figure3_Random_T2T",
                     payload, draw_verified_route_comparison))
    if not jobs:
        raise ValueError("Figure selection must be 2, 3 or all")
    if isolated_output is not None:
        for number, _, _, _, _ in jobs:
            for suffix in (".pdf", ".png"):
                destination = isolated_output / ("Figure{:02d}".format(int(number)) + suffix)
                if destination.exists() and not overwrite:
                    raise FileExistsError("Existing figure output requires overwrite=True: " + str(destination))
        isolated_output.mkdir(parents=True, exist_ok=True)
    _verified_hash_guard(inputs)
    revisions = []
    with matplotlib.rc_context({"font.family": "Arial", "font.size": 10,
                                "pdf.fonttype": 42, "text.usetex": False}):
        for number, directory, stem, payload, drawing_function in jobs:
            original_digest = _verified_digest(payload)
            fig = drawing_function(payload)
            try:
                fig.set_dpi(300)
                numeric = verified_figure_numeric_guard(fig, payload, number)
                geometry = verified_figure_geometry_guard(fig)
                if _verified_digest(payload) != original_digest:
                    raise ValueError("A plotting function mutated source values")
                if isolated_output is not None:
                    export_stem = "Figure{:02d}".format(int(number))
                    outputs = (isolated_output / (export_stem + ".pdf"),
                               isolated_output / (export_stem + ".png"))
                else:
                    outputs = (directory / (stem + ".pdf"), directory / (stem + "_preview.png"))
                if isolated_output is None and not all(path.is_file() for path in outputs):
                    raise ValueError("Only the existing current PDF and preview may be overwritten")
                fig.savefig(outputs[0], metadata={"Creator": "Original verified plotting entry",
                            "CreationDate": None, "ModDate": None})
                fig.savefig(outputs[1], dpi=300, metadata={"Software": "Original verified plotting entry"})
                revisions.append((directory, {"entry": "decycling_dismantling_analysis.py --verified-figures",
                    "drawing_function": drawing_function.__name__, "payload": payload,
                    "payload_sha256": original_digest, "numeric_QA": numeric, "renderer_QA": geometry,
                    "matplotlib_version": matplotlib.__version__,
                    "output_SHA256": {path.name: _verified_sha(path) for path in outputs},
                    "export_policy": ("Independent PDF and one 300-dpi PNG only; accepted figures and provenance untouched"
                                      if isolated_output is not None else
                                      "Existing PDF and one 300-dpi preview only; old SVG/high-dpi PNG untouched"),
                    "history_scope": "Initial GPU0 config identity retained. External GPU1 resume manifest/archive remain separately audited; frozen loader does not cover that history layer."}))
            finally:
                plt.close(fig)
    protected_after = _verified_hash_guard(protected)
    _verified_hash_guard(inputs)
    for directory, revision in revisions:
        revision.update(input_SHA256_before=dict(inputs), input_SHA256_after=dict(inputs),
            protected_result_guard={"existing_count": 5818, "new_transaction_count": 7,
                "before_manifest_digest": protected_before, "after_manifest_digest": protected_after,
                "all_SHA256_match": protected_before == protected_after,
                "complete_expected_manifest": str(audit_path.resolve()),
                "audit_SHA256": inputs[str(audit_path.resolve())]})
        if isolated_output is not None:
            continue
        path = directory / "figure_provenance.json"
        with path.open(encoding="utf-8") as stream:
            provenance = json.load(stream)
        provenance["revision_20261013"] = revision
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(provenance, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    result = {"figures": [job[0] for job in jobs], "result_guard_count": len(protected),
              "protected_results_unchanged": True, "source_inputs_unchanged": True}
    if isolated_output is not None:
        result["figure_audits"] = {job[0]: revision
                                  for job, (_, revision) in zip(jobs, revisions)}
    return result


CONTROL_MANIFEST_RELATIVE = "DiffGDM code/model_test/result/DiffGDM/dismantle_dataset_real_all/single_run_controls.json"
CONTROL_CHECKPOINT_SHA256 = "ff95893d15aba3a04fd014d55317dd9b113f08439524b5b904f4612dbdd9d5bf"


def _controls_atomic(path, value):
    """Checkpoint the one run record, never create per-trial result trees."""
    import tempfile
    path = Path(path)
    handle, temporary = tempfile.mkstemp(prefix=".single_controls_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _controls_rng_capture():
    """Observe the original random stream without drawing or reseeding it."""
    import random
    import numpy as np
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"algorithm": state[0], "keys": state[1].tolist(),
                      "position": state[2], "has_gauss": state[3],
                      "cached_gaussian": state[4]}}


def _controls_rng_restore(state):
    """Only explicit replay/resume uses restoration; normal runs never do."""
    import random
    import numpy as np
    def tuples(value):
        return tuple(tuples(item) for item in value) if isinstance(value, (list, tuple)) else value
    random.setstate(tuples(state["python"]))
    saved = state["numpy"]
    np.random.set_state((saved["algorithm"], np.asarray(saved["keys"], dtype=np.uint32),
                         saved["position"], saved["has_gauss"], saved["cached_gaussian"]))


def prepare_original_controls(workspace_root):
    """Prepare the approved domain; do not execute or change scalar results."""
    from datetime import datetime, timezone
    workspace = Path(workspace_root).resolve()
    manifest_path = workspace / CONTROL_MANIFEST_RELATIVE
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing.get("schema") != "original-controls-v1":
            raise ValueError("Conflicting single-run manifest; refusing replacement")
        return existing
    results = workspace / "DiffGDM code/model_test/result/DiffGDM"
    analysis = results / "decycling_dismantling_analysis"
    names = ("synth_graphs_decycle_size.json", "synth_graphs_dismantling_size.json",
             "graphs_decycle_size.json", "graphs_coredismantle_size.json",
             "graphs_onepassdismantle_size.json", "DiffGDM_graphs_decycle_size.json",
             "DiffGDM_graphs_decycledismantle_size_target.json")
    tables = {name: _verified_canonical_graphs(json.loads((analysis / name).read_text(encoding="utf-8")))
              for name in names}
    synth = sorted(tables[names[0]])
    real = sorted(tables[names[2]])
    if len(synth) != 140 or len(real) != 27 or set(tables[names[1]]) != set(synth):
        raise ValueError("The original paper graph domain must be 140 synthetic and 27 real graphs")
    if any(set(tables[name]) != set(real) for name in names[2:]):
        raise ValueError("Original real-network panels do not share their complete graph domain")
    expected = []
    for dataset, graphs, methods in (("synth", synth, ("Random-Decycle",)),
                                    ("real_all", real, ("Random-Decycle", "random_rein", "T2T_decycle", "T2T_direct"))):
        for graph in graphs:
            relative = "DiffGDM code/model_test/dataset/dataset_" + dataset + "/" + graph + ".graphml"
            if not (workspace / relative).is_file():
                raise ValueError("Missing original input graph: " + relative)
            graph_sha = _verified_sha(workspace / relative)
            for method in methods:
                expected.append({"dataset": dataset, "graph": graph, "method": method,
                                 "graph_relative": relative, "graph_sha256": graph_sha,
                                 "raw_fvs_required": method in ("Random-Decycle", "T2T_decycle")})
    value = {"schema": "original-controls-v1", "status": "PREPARED",
             "created_utc": datetime.now(timezone.utc).isoformat(),
             "expected_records": 248, "expected_raw_fvs": 194, "expected": expected,
             "input_files": {str((analysis / name).relative_to(workspace)).replace("\\", "/"):
                             _verified_sha(analysis / name) for name in names},
             "protected_result_sha256": {str(path.relative_to(workspace)).replace("\\", "/"):
                                         _verified_sha(path) for path in sorted(results.rglob("*"))
                                         if path.is_file() and path != manifest_path},
             "rng_policy": "original stream; no outer seed; model-internal default seed retained",
             "source_sync": {"verified": False}, "records": [], "allattempts": [],
             "network_modes": {}, "cursor": 0, "current": None,
             "checkpoint_sha256": CONTROL_CHECKPOINT_SHA256}
    _controls_atomic(manifest_path, value)
    return value


def _controls_asset_guard(workspace, value, models=None):
    """Check the actual shared module/assets, not only their recorded labels."""
    model_root = Path(workspace) / "DiffGDM code/model_test"
    receipt = value["source_sync"]
    if models is not None and Path(models.__file__).resolve() != (model_root / "method/modelGym/models.py").resolve():
        raise ValueError("Inference imported an unstaged or server-only model implementation")
    for relative, digest in receipt["asset_files"].items():
        if _verified_sha(model_root / relative) != digest:
            raise ValueError("An actual checkpoint/YAML/guidance asset changed: " + relative)
    actual_native = Path.cwd() / "method/noderein_linux/reverse-greedy"
    if _verified_sha(actual_native) != receipt["native_sha256"]:
        raise ValueError("The actually invoked Linux reinsertion binary differs")
    if _verified_sha(model_root / "method/modelGym/trained_models/T2T/T2T") != CONTROL_CHECKPOINT_SHA256:
        raise ValueError("The actual T2T checkpoint differs from the accepted weights")


def _controls_domain(workspace, value):
    """Reject a modified/expanded method or graph domain at every trust boundary."""
    analysis = Path(workspace) / "DiffGDM code/model_test/result/DiffGDM/decycling_dismantling_analysis"
    synth = _verified_canonical_graphs(json.loads((analysis / "synth_graphs_decycle_size.json").read_text(encoding="utf-8")))
    real = _verified_canonical_graphs(json.loads((analysis / "graphs_decycle_size.json").read_text(encoding="utf-8")))
    wanted = {("synth", graph, "Random-Decycle") for graph in synth}
    wanted.update(("real_all", graph, method) for graph in real
                  for method in ("Random-Decycle", "random_rein", "T2T_decycle", "T2T_direct"))
    expected = [(t.get("dataset"), t.get("graph"), t.get("method")) for t in value.get("expected", [])]
    if len(synth) != 140 or len(real) != 27 or len(expected) != 248 or len(set(expected)) != 248 or set(expected) != wanted:
        raise ValueError("Single-controls domain differs from the original 248 approved identities")
    for task in value["expected"]:
        relative = "DiffGDM code/model_test/dataset/dataset_" + task["dataset"] + "/" + task["graph"] + ".graphml"
        if (task.get("graph_relative") != relative
                or task.get("raw_fvs_required") is not (task["method"] in ("Random-Decycle", "T2T_decycle"))):
            raise ValueError("Invalid graph path or FVS route identity")
    return wanted


def _controls_call(models, graph, task, profile, helpers):
    """Observe the original functions, without replacing decoding/postprocessing."""
    import copy
    import time
    observation = {}
    cost = 0.0
    original_break, original_load = models.break_rein, models.load_diffusion_model
    def observe_break(g, nodes, target, rein):
        nonlocal cost
        start = time.monotonic()
        raw = helpers.validate_nodes(g, nodes, "same-call raw FVS")
        if "raw_fvs_nodes" in observation or not helpers.forest_after(g, raw):
            raise ValueError("Same-call raw FVS is missing, duplicated, or infeasible")
        observation.update(raw_fvs_nodes=raw, raw_fvs_ratio=len(raw) / g.number_of_nodes())
        cost += time.monotonic() - start
        return original_break(g, nodes, target, rein)
    def observe_load(*args, **kwargs):
        nonlocal cost
        path = Path(kwargs.get("path", args[0] if args else ""))
        if _verified_sha(path) != CONTROL_CHECKPOINT_SHA256:
            raise ValueError("The actually loaded T2T checkpoint differs")
        result = original_load(*args, **kwargs)
        start = time.monotonic()
        helpers.validate_t2t_effective_parameters(result[1], profile)
        observation["effective_net_param"] = helpers.jsonable(copy.deepcopy(result[1]))
        cost += time.monotonic() - start
        return result
    models.break_rein, models.load_diffusion_model = observe_break, observe_load
    start = time.monotonic()
    try:
        method = task["method"]
        if method == "Random-Decycle":
            remove, reported = models.Random_decycle_dismantle(graph, target=1, method_name=method)
        elif method == "random_rein":
            remove, reported = models.Random_dismantle(graph, target=1, method_name="random_rein")
        else:
            # Omit seed: the author's original model-internal default is retained.
            remove, reported = models.DiffGDM_dismantle_evaluation(
                graph, target=1, method_name="T2T", graph_name=task["graph"],
                models_REIN=True, iszero=False, infer_param=helpers.t2t_infer_param(profile),
                postprocess="decycle" if method == "T2T_decycle" else "direct")
            if "effective_net_param" not in observation:
                raise ValueError("T2T did not expose its actual loaded inference parameters")
        wall = time.monotonic() - start
    finally:
        models.break_rein, models.load_diffusion_model = original_break, original_load
    metrics = helpers.validate_candidate(graph, remove, 1)
    if task["raw_fvs_required"] != ("raw_fvs_nodes" in observation):
        raise ValueError("Same-call FVS evidence does not match the approved route")
    if not isinstance(reported, (int, float)) or not math.isfinite(reported) or reported < 0:
        raise ValueError("Original algorithm returned an invalid time")
    return {**metrics, **observation, "time_model_seconds": float(reported),
            "time_wall_seconds": wall, "instrumentation_seconds": cost}


def run_original_controls(workspace_root, manifest_path, *, resume=False):
    """An OS-owned lock prevents a second execution even after a lost reply."""
    import socket
    identity = hashlib.sha256(str(Path(manifest_path).resolve()).encode()).hexdigest()
    if os.name == "posix":
        lock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        address = "\0diffgdm-original-controls-" + identity
    else:
        lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        address = ("127.0.0.1", 48000 + int(identity[:8], 16) % 15000)
    try:
        lock.bind(address)
    except OSError:
        lock.close()
        raise RuntimeError("An original-control execution already owns this manifest; refusing duplicate start")
    try:
        return _run_original_controls_locked(workspace_root, manifest_path, resume=resume)
    finally:
        lock.close()


def _run_original_controls_locked(workspace_root, manifest_path, *, resume=False):
    """Run one original serial graph/method stream; no ten-seed controller."""
    import gc
    import sys
    import time
    import traceback
    import torch
    workspace, manifest_path = Path(workspace_root).resolve(), Path(manifest_path).resolve()
    sys.path.insert(0, str(workspace / "experiment_ops"))
    import baseline_seed_runner as helpers
    from method.modelGym import models
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("schema") != "original-controls-v1" or not value.get("source_sync", {}).get("verified"):
        raise ValueError("The local/server source and asset synchronization has not passed")
    if value.get("status") not in ("PREPARED", "READY", "LAUNCHING", "LAUNCHED") and not resume:
        raise ValueError("A run already exists; no implicit restart or duplicate inference")
    if resume and value.get("status") != "INTERRUPTED_BETWEEN_CALLS":
        raise ValueError("Resume requires an explicitly certified between-call interruption, not RUNNING or FAILED")
    _controls_domain(workspace, value)
    if len(value["expected"]) != 248 or sum(t["raw_fvs_required"] for t in value["expected"]) != 194:
        raise ValueError("Incorrect approved single-run domain")
    for relative, expected_sha in value["source_sync"]["python_files"].items():
        if _verified_sha(workspace / relative) != expected_sha:
            raise ValueError("Deployed Python source differs from local source: " + relative)
    for relative, expected_sha in value["input_files"].items():
        if _verified_sha(workspace / relative) != expected_sha:
            raise ValueError("Historical plotting input changed: " + relative)
    _controls_asset_guard(workspace, value, models)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or not torch.cuda.is_available():
        raise RuntimeError("Physical GPU1 is required; refusing another GPU or CPU fallback")
    value["runtime"] = {"pid": os.getpid(), "CUDA_VISIBLE_DEVICES": "1", "logical_device": "cuda:0",
                        "gpu_name": torch.cuda.get_device_name(0), "python": sys.version,
                        "torch": torch.__version__, "cuda": torch.version.cuda}
    if resume:
        if value.get("current") is not None or not value.get("rng_checkpoint"):
            raise ValueError("An interrupted in-flight call has no safe automatic resume")
        _controls_rng_restore(value["rng_checkpoint"])
    value["status"] = "RUNNING"
    _controls_atomic(manifest_path, value)
    while value["cursor"] < len(value["expected"]):
        task = value["expected"][value["cursor"]]
        path = workspace / task["graph_relative"]
        if _verified_sha(path) != task["graph_sha256"]:
            raise ValueError("Input graph changed since scheduling")
        graph = helpers.read_graph(path)
        _controls_asset_guard(workspace, value, models)
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 4 * graph.number_of_nodes() + 1000))
        profile = value["network_modes"].get(task["graph"], "P4S1")
        before = _controls_rng_capture()
        value["current"] = {**task, "t2t_profile": profile, "rng_before": before}
        _controls_atomic(manifest_path, value)
        record = {"dataset": task["dataset"], "graph": task["graph"], "method": task["method"],
                  "seed": 123 if task["method"].startswith("T2T_") else None,
                  "node_count": graph.number_of_nodes(), "edge_count": graph.number_of_edges(),
                  "node_order_sha256": _verified_digest(list(graph.nodes)),
                  "rng_before": before, "t2t_profile": profile if task["method"].startswith("T2T_") else None,
                  "fingerprint": {"graph_sha256": task["graph_sha256"],
                                  "source_sha256": value["source_sync"]["source_sha256"],
                                  "checkpoint_sha256": CONTROL_CHECKPOINT_SHA256 if task["method"].startswith("T2T_") else None,
                                  "native_sha256": value["source_sync"]["native_sha256"]},
                  "started_epoch": time.time(), "actual_gpu": 1 if task["method"].startswith("T2T_") else None}
        try:
            record.update(_controls_call(models, graph, task, profile, helpers), status="success")
        except Exception as error:
            record.update(status="failure", error_type=type(error).__name__, error=str(error),
                          traceback=traceback.format_exc(),
                          cuda_oom=helpers.explicit_cuda_oom(type(error).__name__, str(error)))
        record.update(rng_after=_controls_rng_capture(), ended_epoch=time.time())
        value["rng_checkpoint"] = record["rng_after"]
        value["allattempts"].append(record)
        value["current"] = None
        if record["status"] == "success":
            value["records"].append(record)
            value["cursor"] += 1
            _controls_atomic(manifest_path, value)
            print(json.dumps({"success": len(value["records"]), "total": 248,
                              "graph": task["graph"], "method": task["method"]}), flush=True)
            continue
        if (task["method"].startswith("T2T_") and profile == "P4S1" and record["cuda_oom"]):
            value["records"] = [r for r in value["records"]
                                if not (r["dataset"] == "real_all" and r["graph"] == task["graph"]
                                        and r["method"].startswith("T2T_"))]
            value["network_modes"][task["graph"]] = "P2S2"
            value["cursor"] = next(i for i, t in enumerate(value["expected"])
                                   if t["dataset"] == "real_all" and t["graph"] == task["graph"]
                                   and t["method"] == "T2T_decycle")
            gc.collect()
            torch.cuda.empty_cache()
            _controls_atomic(manifest_path, value)
            continue
        value["status"] = "FAILED"
        _controls_atomic(manifest_path, value)
        raise RuntimeError("Single-run inference stopped; see retained failure evidence")
    for relative, expected_sha in value["source_sync"]["python_files"].items():
        if _verified_sha(workspace / relative) != expected_sha:
            raise ValueError("Shared source changed during the run")
    _controls_asset_guard(workspace, value, models)
    value["status"] = "COMPLETE"
    value["completed_epoch"] = time.time()
    _controls_atomic(manifest_path, value)
    return {"status": "COMPLETE", "records": len(value["records"]),
            "raw_fvs": sum("raw_fvs_nodes" in r for r in value["records"])}


def _controls_products(value, before_snapshots):
    """Rebuild only the approved scalar method keys from accepted raw records."""
    products = {}
    allowed = {"synth": {"Random-Decycle"},
               "real_all": {"Random-Decycle", "random_rein", "T2T_decycle", "T2T_direct"}}
    seen = set()
    for record in value["records"]:
        key = (record["dataset"], record["graph"], record["method"])
        if (key in seen or record["dataset"] not in allowed or record["method"] not in allowed[record["dataset"]]
                or record.get("status") != "success"):
            raise ValueError("Products may update only unique approved successful control keys")
        seen.add(key)
    for dataset in ("synth", "real_all"):
        for kind, fields in (("dismantle", (("size", "cfn"), ("time", "time_model_seconds"), ("nodelist", "remove_nodes"))),
                             ("decycle", (("size", "raw_fvs_ratio"), ("nodelist", "raw_fvs_nodes")))):
            for suffix, field in fields:
                filename = ("graphs_dismantle_" + suffix + "_target_1_True.json" if kind == "dismantle"
                            else "graphs_decycle_" + suffix + ".json")
                relative = "DiffGDM code/model_test/result/DiffGDM/" + kind + "_dataset_" + dataset + "/" + filename
                original = json.loads(before_snapshots[relative])
                for record in value["records"]:
                    if record["dataset"] == dataset and field in record:
                        original.setdefault(record["graph"], {})[record["method"]] = record[field]
                products[relative] = (json.dumps(original, indent=4, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    return products


def validate_original_controls(workspace_root, value):
    """Independent structural acceptance, not merely worker success labels."""
    import sys
    workspace = Path(workspace_root).resolve()
    sys.path.insert(0, str(workspace / "experiment_ops"))
    import baseline_seed_runner as helpers
    if value.get("status") not in ("COMPLETE", "ACCEPTED", "PUBLISHED") or len(value.get("records", [])) != 248:
        raise ValueError("All 248 approved single-run results must be complete before publication")
    if not value.get("source_sync", {}).get("verified"):
        raise ValueError("Missing verified shared-source receipt")
    _controls_domain(workspace, value)
    if _verified_digest(value["source_sync"]["python_files"]) != value["source_sync"]["source_sha256"]:
        raise ValueError("Shared source receipt digest mismatch")
    for relative, expected_sha in value["source_sync"]["python_files"].items():
        if _verified_sha(workspace / relative) != expected_sha:
            raise ValueError("The local upload source changed after launch: " + relative)
    expected = {(t["dataset"], t["graph"], t["method"]): t for t in value["expected"]}
    if len(expected) != 248:
        raise ValueError("Duplicate or incorrect scheduled identities")
    seen, raw_count = set(), 0
    graphs = {}
    for record in value["records"]:
        key = (record["dataset"], record["graph"], record["method"])
        if key in seen or key not in expected or record.get("status") != "success":
            raise ValueError("Duplicate, unscheduled, or failed final record")
        seen.add(key)
        task = expected[key]
        path = workspace / task["graph_relative"]
        if _verified_sha(path) != task["graph_sha256"] or record["fingerprint"]["graph_sha256"] != task["graph_sha256"]:
            raise ValueError("Raw record graph fingerprint mismatch")
        graph_key = key[:2]
        graph = graphs.setdefault(graph_key, helpers.read_graph(path)) if graph_key not in graphs else graphs[graph_key]
        metrics = helpers.validate_candidate(graph, record["remove_nodes"], 1)
        if any(record[field] != metrics[field] for field in ("remove_nodes", "cfn", "final_lcc", "lcc_threshold")):
            raise ValueError("Raw deletion set, CFN denominator, or final LCC mismatch")
        if record["node_count"] != graph.number_of_nodes() or record["node_order_sha256"] != _verified_digest(list(graph.nodes)):
            raise ValueError("Preprocessed node domain differs from the original call")
        if task["raw_fvs_required"]:
            raw = helpers.validate_nodes(graph, record["raw_fvs_nodes"], "accepted raw FVS")
            if not helpers.forest_after(graph, raw) or record["raw_fvs_ratio"] != len(raw) / graph.number_of_nodes():
                raise ValueError("Raw FVS is infeasible or uses a different denominator")
            raw_count += 1
        elif any(field in record for field in ("raw_fvs_nodes", "raw_fvs_ratio")):
            raise ValueError("A direct route must not borrow an FVS")
        if record["method"].startswith("T2T_"):
            if record["seed"] != 123 or record["fingerprint"]["checkpoint_sha256"] != CONTROL_CHECKPOINT_SHA256 or record["actual_gpu"] != 1:
                raise ValueError("T2T default seed/checkpoint/device identity mismatch")
            helpers.validate_t2t_effective_parameters(record["effective_net_param"], record["t2t_profile"])
            required_profile = value.get("network_modes", {}).get(record["graph"], "P4S1")
            if record["t2t_profile"] != required_profile:
                raise ValueError("Both T2T routes must use the same accepted whole-network mode")
            if required_profile == "P2S2" and not any(
                    r.get("graph") == record["graph"] and r.get("dataset") == "real_all"
                    and r.get("method", "").startswith("T2T_") and r.get("t2t_profile") == "P4S1"
                    and r.get("status") == "failure" and r.get("cuda_oom") is True
                    and helpers.explicit_cuda_oom(r.get("error_type", ""), r.get("error", ""))
                    for r in value.get("allattempts", [])):
                raise ValueError("P2S2 was not triggered by a retained explicit P4 CUDA OOM")
        elif record["seed"] is not None:
            raise ValueError("Random was mislabeled as a seeded model run")
        if not record.get("rng_before") or not record.get("rng_after"):
            raise ValueError("Original RNG state evidence is missing")
        for field in ("time_model_seconds", "time_wall_seconds", "instrumentation_seconds"):
            if not isinstance(record[field], (int, float)) or not math.isfinite(record[field]) or record[field] < 0:
                raise ValueError("Invalid runtime evidence")
        if record["fingerprint"]["source_sha256"] != value["source_sync"]["source_sha256"]:
            raise ValueError("Raw record source fingerprint mismatch")
    if seen != set(expected) or raw_count != 194:
        raise ValueError("Incomplete single-run or same-call FVS coverage")
    return {"records": 248, "raw_fvs": 194, "structural_acceptance": True}


def publish_original_controls(workspace_root, value):
    """Transactional whitelist publication into the existing scalar directories."""
    import sys
    workspace = Path(workspace_root).resolve()
    manifest_path = workspace / CONTROL_MANIFEST_RELATIVE
    if value.get("status") == "PUBLISHED":
        single_controls_authorized_hashes(workspace / "DiffGDM code/model_test/result/DiffGDM", {})
        return value["acceptance"]
    acceptance = validate_original_controls(workspace, value)
    if value.get("status") == "ACCEPTED":
        publication = value["publication"]
        recovery = _controls_products(value, publication["before_snapshots"])
        if set(recovery) != set(publication["before_sha256"]) or len(recovery) != 10:
            raise ValueError("Invalid interrupted publication target set")
        for relative, content in recovery.items():
            old = publication["before_snapshots"][relative].encode("utf-8")
            old_sha, new_sha = hashlib.sha256(old).hexdigest(), hashlib.sha256(content).hexdigest()
            if (old_sha != publication["before_sha256"][relative]
                    or new_sha != publication["after_sha256"][relative]
                    or _verified_sha(workspace / relative) not in (old_sha, new_sha)):
                raise ValueError("Interrupted publication has a conflicting endpoint; refusing recovery")
            temporary = (workspace / relative).with_name(Path(relative).name + ".single_controls.tmp")
            if temporary.exists() and _verified_sha(temporary) != new_sha:
                raise ValueError("Interrupted publication temporary bytes conflict")
        for relative in recovery:
            target = workspace / relative
            temporary = target.with_name(target.name + ".single_controls.tmp")
            if temporary.exists():
                temporary.unlink()
            target.write_bytes(publication["before_snapshots"][relative].encode("utf-8"))
        value["publication_recovery"] = "Verified before/after endpoints restored before explicit republication"
        value["status"] = "COMPLETE"
        _controls_atomic(manifest_path, value)
    for relative, expected_sha in value["input_files"].items():
        if _verified_sha(workspace / relative) != expected_sha:
            raise ValueError("Original paper plotting inputs changed")
    for relative, expected_sha in value["protected_result_sha256"].items():
        if _verified_sha(workspace / relative) != expected_sha:
            raise ValueError("A protected pre-existing result changed: " + relative)
    # Snapshot existing endpoint bytes inside this one manifest, not a new backup tree.
    snapshots = {}
    targets = {
        "DiffGDM code/model_test/result/DiffGDM/" + kind + "_dataset_" + dataset + "/" + filename:
        None
        for dataset in ("synth", "real_all") for kind, filenames in (
            ("dismantle", ("graphs_dismantle_size_target_1_True.json", "graphs_dismantle_time_target_1_True.json", "graphs_dismantle_nodelist_target_1_True.json")),
            ("decycle", ("graphs_decycle_size.json", "graphs_decycle_nodelist.json"))) for filename in filenames}
    for relative in targets:
        snapshots[relative] = (workspace / relative).read_bytes().decode("utf-8")
    products = _controls_products(value, snapshots)
    before = {relative: _verified_sha(workspace / relative) for relative in products}
    after = {relative: hashlib.sha256(content).hexdigest() for relative, content in products.items()}
    value.update(status="ACCEPTED", acceptance=acceptance,
                 publication={"before_snapshots": snapshots, "before_sha256": before, "after_sha256": after,
                              "files": [{"relative": relative, "old_sha256": before[relative], "new_sha256": after[relative]}
                                        for relative in products]})
    _controls_atomic(manifest_path, value)
    # Keep exact before bytes for rollback; don't use a generic overwrite importer.
    changed = []
    try:
        for relative, content in products.items():
            target = workspace / relative
            if _verified_sha(target) != before[relative]:
                raise ValueError("Publication target changed during acceptance")
            temporary = target.with_name(target.name + ".single_controls.tmp")
            if temporary.exists():
                raise ValueError("Conflicting publication temporary file")
            with temporary.open("wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            changed.append(relative)
        value["status"] = "PUBLISHED"
        _controls_atomic(manifest_path, value)
    except Exception:
        for relative in reversed(changed):
            (workspace / relative).write_bytes(snapshots[relative].encode("utf-8"))
        raise
    single_controls_authorized_hashes(workspace / "DiffGDM code/model_test/result/DiffGDM", {})
    return acceptance


def single_controls_authorized_hashes(result_root, expected_hashes):
    """Authenticate this whitelist replacement without weakening older audits."""
    result_root = Path(result_root).resolve()
    workspace = result_root.parents[3]
    path = result_root / "dismantle_dataset_real_all/single_run_controls.json"
    expected = dict(expected_hashes)
    if not path.exists():
        return expected
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PUBLISHED":
        return expected
    _controls_domain(workspace, value)
    selected = value.get("records", [])
    if (len(selected) != 248 or sum("raw_fvs_nodes" in r for r in selected) != 194
            or {(r["dataset"], r["graph"], r["method"]) for r in selected} !=
            {(t["dataset"], t["graph"], t["method"]) for t in value["expected"]}):
        raise ValueError("Published controls lack their complete original domain")
    publication = value["publication"]
    products = _controls_products(value, publication["before_snapshots"])
    if set(products) != set(publication["before_sha256"]) or len(products) != 10:
        raise ValueError("Unexpected single-controls publication file set")
    for relative, content in products.items():
        target = workspace / relative
        before = publication["before_sha256"][relative]
        if hashlib.sha256(publication["before_snapshots"][relative].encode("utf-8")).hexdigest() != before:
            raise ValueError("Single-controls original endpoint bytes do not match their SHA")
        previous = expected.get(str(target))
        if previous is not None and previous != before:
            raise ValueError("Single-controls before image does not match the older protected audit")
        after = hashlib.sha256(content).hexdigest()
        if after != publication["after_sha256"][relative] or _verified_sha(target) != after:
            raise ValueError("Single-controls authorized after image differs")
        expected[str(target)] = after
    expected[str(path)] = _verified_sha(path)
    return expected


SCALE_ENSEMBLE_RELATIVE = "DiffGDM code/model_test/result/DiffGDM/dismantle_dataset_synth_size/scalability_ensemble.json"
SCALE_ENSEMBLE_DIRECTIONS = ["MVC", "LCC", "GCC", "ACC", "Density"]
SCALE_ENSEMBLE_WAVES = [["MVC"], ["LCC", "GCC"], ["ACC", "Density"]]
SCALE_ENSEMBLE_CONFIG = {
    "parallel_sampling": 1, "sequential_sampling": 2, "rewrite": True,
    "rewrite_steps": 6, "rewrite_ratio": .7, "inference_diffusion_steps": 20,
    "use_tag": True, "gradient_estimator_type": "cost_estimator",
    "target_weight": 1, "MVC_P": 1000, "diffusion_type": "discretFlow"}


def _scale_ensemble_domain(workspace, value=None):
    """Keep the new experiment on exactly the historical fifty graph instances."""
    import re
    workspace = Path(workspace)
    directory = (workspace / SCALE_ENSEMBLE_RELATIVE).parent
    size = json.loads((directory / "graphs_dismantle_size_target_1_True.json").read_text(encoding="utf-8"))
    times = json.loads((directory / "graphs_dismantle_time_target_1_True.json").read_text(encoding="utf-8"))
    if len(size) != 50 or set(size) != set(times):
        raise ValueError("Scale experiment requires the same fifty historical size/time graph keys")
    expected, groups = [], {}
    for graph in size:
        match = re.fullmatch(r"small-world_size_(20|200|2000|20000|200000)-([0-9]+)\.0_([0-9])", graph)
        if not match or int(match[2]) != int(match[1]) * 3 // 2:
            raise ValueError("Unexpected historical scale graph identity: " + graph)
        scale = int(match[1])
        groups.setdefault(scale, set()).add(int(match[3]))
        relative = "DiffGDM code/model_test/dataset/dataset_synth_size/" + graph + ".graphml"
        expected.append({"graph": graph, "graph_relative": relative,
                         "graph_sha256": _verified_sha(workspace / relative), "scale": scale})
    if set(groups) != {20, 200, 2000, 20000, 200000} or any(v != set(range(10)) for v in groups.values()):
        raise ValueError("Every scale must contain all ten distinct graph instances")
    expected.sort(key=lambda item: (item["scale"], int(item["graph"].rsplit("_", 1)[1])))
    if value is not None:
        if (value.get("schema") != "scale-ensemble-v1" or value.get("expected") != expected
                or value.get("directions") != SCALE_ENSEMBLE_DIRECTIONS
                or value.get("fixed_waves") != SCALE_ENSEMBLE_WAVES
                or value.get("effective_config") != SCALE_ENSEMBLE_CONFIG
                or value.get("expected_graphs") != 50 or value.get("expected_branches") != 250
                or value.get("no_outer_seed") is not True):
            raise ValueError("Five-guide scale experiment domain/configuration changed")
    return expected


def prepare_scale_ensemble_manifest(workspace_root):
    """Create only the scale sidecar; historical method JSON files remain immutable."""
    from datetime import datetime, timezone
    workspace = Path(workspace_root).resolve()
    path = workspace / SCALE_ENSEMBLE_RELATIVE
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        _scale_ensemble_domain(workspace, value)
        return value
    expected = _scale_ensemble_domain(workspace)
    protected_paths = list((workspace / "DiffGDM code/model_test/result").rglob("*"))
    protected_paths += list((workspace / "DiffGDM code/experiments_figure/outputs").glob("Figure*"))
    protected_paths += [workspace / "DiffGDM_Final_Theorem_Proof.tex"]
    protected = {p.relative_to(workspace).as_posix(): _verified_sha(p)
                 for p in sorted(protected_paths) if p.is_file() and p != path
                 and not p.name.startswith("Figure08")}
    value = {"schema": "scale-ensemble-v1", "state": "PREPARED",
             "created_utc": datetime.now(timezone.utc).isoformat(),
             "expected_graphs": 50, "expected_branches": 250, "expected": expected,
             "directions": SCALE_ENSEMBLE_DIRECTIONS, "fixed_waves": SCALE_ENSEMBLE_WAVES,
             "effective_config": SCALE_ENSEMBLE_CONFIG,
             "seed_policy": "Original internal default seed=123 at the original branch call; no outer reset",
             "no_outer_seed": True, "physical_gpu": 0, "max_workers": 2,
             "protected_files": protected, "records": [], "attempts": [], "current": None,
             "source_sync": {"verified": False}}
    _controls_atomic(path, value)
    return value


def _scale_ensemble_helpers(workspace):
    import sys
    sys.path.insert(0, str(Path(workspace) / "experiment_ops"))
    import baseline_seed_runner
    return baseline_seed_runner


def _scale_ensemble_source_guard(workspace, value, runtime=False):
    workspace = Path(workspace)
    receipt = value.get("source_sync", {})
    if not receipt.get("verified") or not receipt.get("python_files") or not receipt.get("asset_files"):
        raise ValueError("Scale experiment has no verified common-source/asset receipt")
    if _verified_digest(receipt["python_files"]) != receipt["source_sha256"]:
        raise ValueError("Scale common-source digest differs")
    model = workspace / "DiffGDM code/model_test"
    for relative, sha in receipt["python_files"].items():
        if _verified_sha(workspace / relative) != sha:
            raise ValueError("Scale experiment common Python source changed: " + relative)
    for relative, sha in receipt["asset_files"].items():
        if _verified_sha(model / relative) != sha:
            raise ValueError("Scale experiment model/guidance asset changed: " + relative)
        if runtime and _verified_sha(Path.cwd() / relative) != sha:
            raise ValueError("Scale runtime would load a different model/guidance asset: " + relative)
    if _verified_sha(model / "method/noderein_linux/reverse-greedy") != receipt["native_sha256"]:
        raise ValueError("Scale experiment native binary changed")
    if runtime and _verified_sha(Path.cwd() / "method/noderein_linux/reverse-greedy") != receipt["native_sha256"]:
        raise ValueError("Scale inference would invoke a different native binary")


def _scale_gpu_snapshot():
    """Read only GPU0 and distinguish owned subprocesses from unrelated compute."""
    import subprocess
    import time
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=15)
    rows = [[part.strip() for part in line.split(",")] for line in query.stdout.splitlines()]
    gpu = next((row for row in rows if row[0] == "0"), None)
    if gpu is None:
        raise RuntimeError("Physical GPU0 was not found")
    owned = {os.getpid()}
    parentage = {}
    for directory in Path("/proc").iterdir():
        if directory.name.isdigit():
            try:
                fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
                parentage[int(directory.name)] = int(fields[1])
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
    while True:
        expanded = owned | {pid for pid, parent in parentage.items() if parent in owned}
        if expanded == owned:
            break
        owned = expanded
    compute = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=15)
    applications = []
    for line in compute.stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) == 3 and values[0] == gpu[1]:
            applications.append({"pid": int(values[1]), "memory_mib": values[2], "owned": int(values[1]) in owned})
    return {"epoch": time.time(), "physical_index": 0, "uuid": gpu[1], "name": gpu[2],
            "total_mib": int(gpu[3]), "free_mib": int(gpu[4]), "utilization_percent": int(gpu[5]),
            "compute": applications, "foreign_pids": [a["pid"] for a in applications if not a["owned"]]}


def _scale_cuda_uuid():
    """Resolve the UUID of the actual PyTorch context, not an assumed index map."""
    import ctypes
    import uuid
    import torch
    torch.cuda.set_device(0)
    probe = torch.empty(1, device="cuda:0")
    driver = ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int()
    class DeviceUUID(ctypes.Structure):
        _fields_ = [("data", ctypes.c_ubyte * 16)]
    native_uuid = DeviceUUID()
    driver.cuCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    function = getattr(driver, "cuDeviceGetUuid_v2", driver.cuDeviceGetUuid)
    function.argtypes = [ctypes.POINTER(DeviceUUID), ctypes.c_int]
    if driver.cuCtxGetDevice(ctypes.byref(device)) != 0 or function(ctypes.byref(native_uuid), device) != 0:
        raise RuntimeError("Cannot certify PyTorch logical0 to physical GPU0 mapping")
    del probe
    return "GPU-" + str(uuid.UUID(bytes=bytes(native_uuid.data)))


def _scale_resume_history(workspace, value):
    """Authenticate an explicitly authorized continuation against immutable evidence."""
    import tarfile
    resume = value.get("resume")
    if not resume:
        return None
    if resume.get("authorization") != "2026-11-08 user requested restart after verified repair":
        raise ValueError("Missing explicit scale continuation authorization")
    archive = Path(workspace) / SCALE_ENSEMBLE_RELATIVE
    archive = archive.parent / "scalability_ensemble_failure_20261108.tar.gz"
    if _verified_sha(archive) != "80998a3e6da1f53e92f570032563ca6a813b3f9c02d5b7aba9a3e7f70e2327bf":
        raise ValueError("Original scale failure archive changed")
    with tarfile.open(archive, "r:gz") as bundle:
        raw = bundle.extractfile("scalability_ensemble.json").read()
    if hashlib.sha256(raw).hexdigest() != "feba4d5d0b3e80c613cae6bdf483355463f2582c67638ac78318419de99dad26":
        raise ValueError("Original scale manifest changed")
    prior = json.loads(raw)
    if (prior["state"] != "FAILED" or len(prior["records"]) != 40 or len(prior["attempts"]) != 41
            or _verified_digest(value["records"][:40]) != _verified_digest(prior["records"])
            or _verified_digest(value["attempts"][:41]) != _verified_digest(prior["attempts"])
            or resume.get("prior_source_sync") != prior["source_sync"]
            or resume.get("prior_launch") != prior["launch"]):
        raise ValueError("Scale continuation altered original successes/failure/provenance")
    if resume.get("budget_repair"):
        archive = archive.parent / "scalability_ensemble_budget_failure_20261108.tar.gz"
        if _verified_sha(archive) != "9238b9a1f8536fdf7e914fa047dba622adbc68bfc308f442b05d2438bd44518f":
            raise ValueError("Budget failure archive changed")
        with tarfile.open(archive, "r:gz") as bundle:
            raw = bundle.extractfile("scalability_ensemble.json").read()
        if hashlib.sha256(raw).hexdigest() != "9fae4cbb314fa9d6201821ea00a698ccea60ae3247c34f4ca1a3e6b5ac448b97":
            raise ValueError("Budget failure manifest changed")
        prior = json.loads(raw)
        if (prior["state"] != "FAILED" or len(prior["records"]) != 40 or len(prior["attempts"]) != 42
                or _verified_digest(value["records"][:40]) != _verified_digest(prior["records"])
                or _verified_digest(value["attempts"][:42]) != _verified_digest(prior["attempts"])
                or resume["budget_repair"].get("prior_source_sync") != prior["source_sync"]
                or resume["budget_repair"].get("prior_launch") != prior["launch"]):
            raise ValueError("Budget continuation altered preceding evidence")
    return prior


def _scale_ensemble_validate_record(workspace, value, task, record, helpers=None):
    """Recompute forest feasibility, final CFN and the measured wave intervals."""
    helpers = helpers or _scale_ensemble_helpers(workspace)
    source = value["source_sync"]["source_sha256"]
    if value.get("resume") and task in value["expected"][:40]:
        source = value["resume"]["prior_source_sync"]["source_sha256"]
    if (record.get("status") != "success" or record.get("graph") != task["graph"]
            or record.get("graph_sha256") != task["graph_sha256"] or record.get("scale") != task["scale"]
            or record.get("source_sha256") != source
            or record.get("actual_gpu") != 0 or record.get("seed") != 123):
        raise ValueError("Scale ensemble record identity/configuration mismatch")
    graph = helpers.read_graph(Path(workspace) / task["graph_relative"])
    if (record.get("node_count") != graph.number_of_nodes() or record.get("edge_count") != graph.number_of_edges()
            or record.get("node_order_sha256") != _verified_digest(list(graph.nodes))):
        raise ValueError("Scale ensemble preprocessing identity changed")
    audit = record["audit"]
    incoming = audit.get("branches", [])
    if (audit.get("status") != "COMPLETE" or audit.get("failures") or len(incoming) != 5
            or {b["direction"] for b in incoming} != set(SCALE_ENSEMBLE_DIRECTIONS)):
        raise ValueError("All five valid branches from one complete call are required")
    by_direction = {b["direction"]: b for b in incoming}
    branches = [by_direction[d] for d in SCALE_ENSEMBLE_DIRECTIONS]
    tolerance = {"rel_tol": 1e-9, "abs_tol": 1e-7}
    for branch in branches:
        raw = helpers.validate_nodes(graph, branch["raw_fvs_nodes"], "scale raw FVS")
        if len(raw) != branch["raw_fvs_count"] or not helpers.forest_after(graph, raw):
            raise ValueError("A scale branch has an invalid same-call raw FVS")
        metrics = helpers.validate_candidate(graph, branch["final_nodes"], 1)
        if (len(metrics["remove_nodes"]) != branch["final_count"]
                or branch.get("final_lcc") != metrics["final_lcc"]
                or branch.get("component_size_limit") != metrics["lcc_threshold"]):
            raise ValueError("Scale branch final removal count mismatch")
        params = branch["effective_config"]
        if (any(params.get(key) != expected for key, expected in SCALE_ENSEMBLE_CONFIG.items())
                or params.get("infer_direction") != branch["direction"]
                or branch.get("device") not in ("cuda:0", "cuda", 0)
                or branch.get("seed") != 123):
            raise ValueError("Scale branch actual effective parameters/device differ")
        for field in ("inference_seconds", "external_decode_seconds", "postprocess_seconds"):
            if not isinstance(branch.get(field), (int, float)) or not math.isfinite(branch[field]) or branch[field] < 0:
                raise ValueError("Incomplete/invalid scale timing: " + field)
        if not math.isclose(branch["inference_finished"] - branch["inference_started"], branch["inference_seconds"], **tolerance):
            raise ValueError("Branch inference clock interval differs")
    waves = audit.get("waves", [])
    if [w["directions"] for w in waves] != SCALE_ENSEMBLE_WAVES:
        raise ValueError("Scale ensemble parallel waves changed")
    last_end = None
    for wave in waves:
        selected = [b for b in branches if b["direction"] in wave["directions"]]
        begin = min(b["inference_started"] for b in selected)
        end = max(b["inference_finished"] for b in selected)
        if (not math.isclose(wave["start"], begin, **tolerance) or not math.isclose(wave["end"], end, **tolerance)
                or not math.isclose(wave["seconds"], end - begin, **tolerance)
                or (last_end is not None and begin < last_end)):
            raise ValueError("Scale ensemble wave interval mismatch")
        last_end = end
    parallel = sum(w["seconds"] for w in waves)
    for duration in (parallel, audit["wall_seconds"], record["wall_seconds"], record["legacy_return_seconds"]):
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            raise ValueError("Scale wall/inference times must be finite nonnegative durations")
    if (not math.isclose(audit["inference_parallel_seconds"], parallel, **tolerance)
            or not math.isclose(record["inference_seconds"], parallel, **tolerance)
            or not math.isclose(record["legacy_return_seconds"], audit["legacy_inference_sum_seconds"], **tolerance)
            or not math.isclose(audit["legacy_inference_sum_seconds"], sum(b["inference_seconds"] for b in branches), **tolerance)
            or audit["wall_seconds"] + 1e-6 < parallel or record["wall_seconds"] + 1e-6 < audit["wall_seconds"]):
        raise ValueError("Scale parallel inference time is not the sum of actual wave spans")
    winner = min(branches, key=lambda b: b["final_count"])
    metrics = helpers.validate_candidate(graph, record["remove_nodes"], 1)
    if (audit["winner"] != winner["direction"] or record["winner"] != winner["direction"]
            or record["remove_nodes"] != winner["final_nodes"]
            or any(record.get(key) != metrics[key] for key in ("cfn", "final_lcc", "lcc_threshold"))):
        raise ValueError("Scale ensemble did not select the original-order best feasible branch")
    resource = record.get("resource_observation", {})
    if (not resource.get("samples") or resource.get("errors") or resource.get("foreign_events")
            or resource.get("before", {}).get("physical_index") != 0 or resource.get("after", {}).get("physical_index") != 0
            or resource.get("before", {}).get("foreign_pids") or resource.get("after", {}).get("foreign_pids")
            or not resource.get("before", {}).get("uuid")
            or resource["before"]["uuid"] != resource["after"].get("uuid")
            or resource["before"]["uuid"] != record.get("actual_gpu_uuid")
            or record.get("actual_gpu_uuid") != value.get("runtime", {}).get("gpu_uuid")):
        raise ValueError("Scale inference time lacks clean GPU0 resource evidence")
    return {"graph": task["graph"], "branches": 5, "cfn": metrics["cfn"], "inference_seconds": parallel}


def run_scale_ensemble(workspace_root, manifest_path):
    """Run the fixed three waves once per graph in an isolated native directory."""
    import socket
    import sys
    import time
    import threading
    import traceback
    import torch
    workspace, path = Path(workspace_root).resolve(), Path(manifest_path).resolve()
    if Path(manifest_path).is_symlink() or path != workspace / SCALE_ENSEMBLE_RELATIVE:
        raise ValueError("Scale run must use its unique registered manifest path")
    identity = hashlib.sha256(str(path).encode()).hexdigest()
    if os.name != "posix" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Scale ensemble requires the isolated Linux session on physical GPU0")
    lock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    lock.bind("\0diffgdm-scale-ensemble-" + identity)
    value = None
    owned_execution = False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        prior = _scale_resume_history(workspace, value)
        allowed_history = prior is not None and len(value["attempts"]) == len(prior["attempts"]) and len(value["records"]) == 40
        if (value.get("state") not in ("PREPARED", "READY", "LAUNCHING", "LAUNCHED")
                or value.get("current") or (value.get("attempts") and not allowed_history)):
            raise ValueError("No implicit restart, retry, or reuse of an in-flight scale run")
        launch = value.get("launch", {})
        start_ticks = int((Path("/proc") / str(os.getpid()) / "stat").read_text().rsplit(") ", 1)[1].split()[19])
        if (launch.get("pid") != os.getpid() or launch.get("start_ticks") != start_ticks
                or not launch.get("token") or launch["token"] != os.environ.get("DIFFGDM_SCALE_ENSEMBLE_TOKEN")):
            raise ValueError("Scale run process does not own the registered PID/start-time/token")
        owned_execution = True
        _scale_ensemble_domain(workspace, value)
        _scale_ensemble_source_guard(workspace, value, runtime=True)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Physical GPU0 must be the only visible CUDA device")
        gpu_uuid = _scale_cuda_uuid()
        if gpu_uuid != _scale_gpu_snapshot()["uuid"] or gpu_uuid != value["source_sync"].get("actual_gpu", {}).get("uuid"):
            raise RuntimeError("PyTorch logical cuda:0 does not map to physical GPU0")
        from method.modelGym import models
        if Path(models.__file__).resolve() != (workspace / "DiffGDM code/model_test/method/modelGym/models.py").resolve():
            raise ValueError("Imported ensemble code is not the synchronized local source")
        helpers = _scale_ensemble_helpers(workspace)
        value["runtime"] = {"pid": os.getpid(), "CUDA_VISIBLE_DEVICES": "0", "logical_device": "cuda:0",
                            "gpu_uuid": gpu_uuid, "start_ticks": start_ticks, "launch_token": launch["token"],
                            "gpu_name": torch.cuda.get_device_name(0), "python": sys.version,
                            "torch": torch.__version__, "cuda": torch.version.cuda}
        value["state"] = "RUNNING"
        _controls_atomic(path, value)
        for task in value["expected"]:
            if any(r["graph"] == task["graph"] for r in value["records"]):
                _scale_ensemble_validate_record(workspace, value, task, next(r for r in value["records"] if r["graph"] == task["graph"]))
                continue
            _scale_ensemble_source_guard(workspace, value, runtime=True)
            if _verified_sha(workspace / task["graph_relative"]) != task["graph_sha256"]:
                raise ValueError("Scale input graph SHA changed")
            graph = helpers.read_graph(workspace / task["graph_relative"])
            sys.setrecursionlimit(max(sys.getrecursionlimit(), 4 * graph.number_of_nodes() + 1000))
            audit = {}
            observation = {"samples": 0, "errors": [], "foreign_events": []}
            record = {**task, "status": "running", "actual_gpu": 0, "actual_gpu_uuid": gpu_uuid, "seed": 123,
                      "node_count": graph.number_of_nodes(), "edge_count": graph.number_of_edges(),
                      "node_order_sha256": _verified_digest(list(graph.nodes)),
                      "source_sha256": value["source_sync"]["source_sha256"],
                      "rng_before": _controls_rng_capture(), "started_epoch": time.time()}
            value["current"] = {"graph": task["graph"], "scale": task["scale"], "method": "DiffGDM_ens"}
            _controls_atomic(path, value)
            stop = threading.Event()
            def observe():
                try:
                    sample = _scale_gpu_snapshot()
                    observation["samples"] += 1
                    observation.setdefault("before", sample)
                    observation["after"] = sample
                    observation["min_free_mib"] = min(observation.get("min_free_mib", sample["free_mib"]), sample["free_mib"])
                    if sample["foreign_pids"]:
                        observation["foreign_events"].append(sample)
                except Exception as error:
                    observation["errors"].append(str(error))
            def watch():
                while not stop.wait(2):
                    observe()
            watcher = None
            started = time.monotonic()
            try:
                observe()
                if observation["errors"] or observation["foreign_events"]:
                    raise RuntimeError("GPU0 is occupied or resource monitoring is unavailable; no inference started")
                watcher = threading.Thread(target=watch, daemon=True)
                watcher.start()
                parameters = dict(SCALE_ENSEMBLE_CONFIG, guided_model_path="method/modelGym/Diffusion_model/robustness_metrics_models")
                record["algorithm_call_started"] = True
                removed, legacy_time = models.DiffGDM_ensemble_evaluation(
                    graph, decode_target=1, graph_name=task["graph"], models_REIN=True,
                    infer_param=parameters, postprocess="decycle", ensemble_workers=2,
                    ensemble_directions=list(SCALE_ENSEMBLE_DIRECTIONS),
                    ensemble_fixed_waves=[list(w) for w in SCALE_ENSEMBLE_WAVES], ensemble_audit=audit)
                record.update(helpers.validate_candidate(graph, removed, 1))
                record.update(winner=audit["winner"], inference_seconds=audit["inference_parallel_seconds"],
                              legacy_return_seconds=float(legacy_time), status="success")
            except Exception as error:
                record.update(status="failure", error_type=type(error).__name__, error=str(error),
                              traceback=traceback.format_exc(),
                              cuda_oom=helpers.explicit_cuda_oom(type(error).__name__, str(error)))
            finally:
                record["wall_seconds"] = time.monotonic() - started
                stop.set()
                if watcher is not None:
                    watcher.join(timeout=35)
                    if watcher.is_alive():
                        observation["errors"].append("GPU resource sampler did not finish")
                observe()
                record.update(audit=helpers.jsonable(audit), resource_observation=observation,
                              rng_after=_controls_rng_capture(), ended_epoch=time.time())
            if record["status"] == "success":
                try:
                    _scale_ensemble_validate_record(workspace, value, task, record, helpers)
                except Exception as error:
                    record.update(status="failure", error_type=type(error).__name__, error=str(error),
                                  traceback=traceback.format_exc(), cuda_oom=False)
            value["attempts"].append(record)
            value["current"] = None
            if record["status"] != "success":
                value["state"] = "FAILED"
                _controls_atomic(path, value)
                raise RuntimeError("Scale ensemble stopped; complete failed-attempt evidence retained")
            value["records"].append(record)
            _controls_atomic(path, value)
            print(json.dumps({"state": "RUNNING", "successes": len(value["records"]), "expected": 50,
                              "branches": 5 * len(value["records"]), "graph": task["graph"],
                              "cfn": record["cfn"], "inference_seconds": record["inference_seconds"]}), flush=True)
        _scale_ensemble_source_guard(workspace, value, runtime=True)
        value.update(state="COMPLETE", completed_epoch=time.time())
        _controls_atomic(path, value)
        return {"state": "COMPLETE", "graphs": 50, "branches": 250}
    except BaseException as error:
        if owned_execution:
            value["state"] = "FAILED"
            value["execution_error"] = {"type": type(error).__name__, "error": str(error),
                                        "traceback": traceback.format_exc()}
            _controls_atomic(path, value)
        raise
    finally:
        lock.close()


def validate_scale_ensemble(workspace_root, value):
    workspace = Path(workspace_root).resolve()
    prior = _scale_resume_history(workspace, value)
    _scale_ensemble_domain(workspace, value)
    _scale_ensemble_source_guard(workspace, value)
    for relative, digest in value["protected_files"].items():
        if _verified_sha(workspace / relative) != digest:
            raise ValueError("A pre-existing result/figure changed: " + relative)
    if value.get("state") not in ("COMPLETE", "ACCEPTED") or len(value.get("records", [])) != 50:
        raise ValueError("All fifty scale ensemble graphs must be complete before acceptance")
    runtime = value.get("runtime", {})
    if (runtime.get("CUDA_VISIBLE_DEVICES") != "0" or runtime.get("logical_device") != "cuda:0"
            or not runtime.get("gpu_uuid") or runtime["gpu_uuid"] != value["source_sync"].get("actual_gpu", {}).get("uuid")):
        raise ValueError("Scale experiment did not run exclusively on physical GPU0")
    launch = value.get("launch", {})
    if (not launch.get("token") or runtime.get("launch_token") != launch["token"]
            or runtime.get("pid") != launch.get("pid") or runtime.get("start_ticks") != launch.get("start_ticks")):
        raise ValueError("Scale runtime does not match the registered launch identity")
    selected_attempts = value["attempts"] if prior is None else prior["records"] + value["attempts"][len(prior["attempts"]):]
    if len(selected_attempts) != 50 or any(r.get("status") != "success" for r in selected_attempts):
        raise ValueError("The scale run has failures, retries, or incomplete attempt history")
    if _verified_digest(value["records"]) != _verified_digest(selected_attempts):
        raise ValueError("Final records are not the complete single-attempt history")
    identities = [r["graph"] for r in value["records"]]
    if identities != [task["graph"] for task in value["expected"]]:
        raise ValueError("Scale ensemble graph coverage/order differs")
    helpers = _scale_ensemble_helpers(workspace)
    checks = [_scale_ensemble_validate_record(workspace, value, task, record, helpers)
              for task, record in zip(value["expected"], value["records"])]
    stats = {}
    for scale in (20, 200, 2000, 20000, 200000):
        group = [r for r in value["records"] if r["scale"] == scale]
        stats[str(scale)] = {"n_graphs": 10, "variance_ddof": 0}
        for field in ("cfn", "inference_seconds"):
            values = [r[field] for r in group]
            stats[str(scale)][field] = {"mean": statistics.mean(values),
                                       "variance": statistics.pvariance(values), "sd": statistics.pstdev(values)}
    return {"graphs": len(checks), "branches": 250, "raw_fvs": 250,
            "records_sha256": _verified_digest(value["records"]), "statistics": stats,
            "structural_acceptance": True, "historical_values_unchanged": True}


def _scale_ensemble_archive_guard(workspace, value):
    import tarfile
    archive = (Path(workspace) / SCALE_ENSEMBLE_RELATIVE).with_suffix(".tar.gz")
    transport = value.get("transport", {})
    if not archive.is_file() or _verified_sha(archive) != transport.get("archive_sha256"):
        raise ValueError("Scale ensemble final archive is absent or has changed")
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        wanted = transport.get("member_sha256", {})
        if {m.name for m in members} != set(wanted) or len(members) != len(wanted):
            raise ValueError("Scale ensemble archive member domain differs")
        for member in members:
            if not member.isfile() or hashlib.sha256(bundle.extractfile(member).read()).hexdigest() != wanted[member.name]:
                raise ValueError("Scale ensemble archive member hash mismatch")
        original = json.loads(bundle.extractfile("scalability_ensemble.json").read())
    if original.get("state") != "COMPLETE":
        raise ValueError("Scale ensemble archive is not a complete remote terminal snapshot")
    for field in ("records", "attempts", "expected", "source_sync", "runtime", "effective_config", "protected_files", "resume"):
        if _verified_digest(original.get(field)) != _verified_digest(value.get(field)):
            raise ValueError("Local scale ensemble differs from the archived terminal evidence: " + field)
    return archive


def accept_scale_ensemble(workspace_root):
    workspace = Path(workspace_root).resolve()
    path = workspace / SCALE_ENSEMBLE_RELATIVE
    value = json.loads(path.read_text(encoding="utf-8"))
    _scale_ensemble_archive_guard(workspace, value)
    acceptance = validate_scale_ensemble(workspace, value)
    if value.get("state") == "ACCEPTED":
        if value.get("acceptance") != acceptance:
            raise ValueError("Saved scale acceptance differs from independent verification")
        return acceptance
    value.update(state="ACCEPTED", acceptance=acceptance)
    _controls_atomic(path, value)
    return acceptance


def load_scale_ensemble_overlay(workspace_root):
    workspace = Path(workspace_root).resolve()
    path = workspace / SCALE_ENSEMBLE_RELATIVE
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("state") != "ACCEPTED":
        raise ValueError("Scale ensemble plotting is gated on complete acceptance")
    archive = _scale_ensemble_archive_guard(workspace, value)
    acceptance = validate_scale_ensemble(workspace, value)
    if acceptance != value.get("acceptance"):
        raise ValueError("Scale ensemble acceptance evidence changed")
    paths = [path, archive] + [workspace / t["graph_relative"] for t in value["expected"]]
    paths += [path.parent / name for name in ("graphs_dismantle_size_target_1_True.json", "graphs_dismantle_time_target_1_True.json")]
    return {"cfn": {r["graph"]: r["cfn"] for r in value["records"]},
            "inference_seconds": {r["graph"]: r["inference_seconds"] for r in value["records"]},
            "statistics": acceptance["statistics"],
            "provenance": {"input_SHA256": {str(p): _verified_sha(p) for p in paths},
                           "directions": SCALE_ENSEMBLE_DIRECTIONS, "effective_config": SCALE_ENSEMBLE_CONFIG,
                           "physical_gpu": 0, "waves": SCALE_ENSEMBLE_WAVES, "variance_ddof": 0,
                           "timing_scope": "synchronized decode_step wave spans; external postprocessing excluded"}}


def _verified_figure_cli():
    parser = argparse.ArgumentParser(description="Original analysis and verified figure reproduction")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verified-figures", action="store_true")
    mode.add_argument("--prepare-original-controls", action="store_true")
    mode.add_argument("--run-original-controls", action="store_true")
    mode.add_argument("--start-original-controls", action="store_true",
                      help="Prepare, verify and deploy this local source to GPU1 once")
    mode.add_argument("--status-original-controls", action="store_true",
                      help="One bounded read of the registered server execution")
    mode.add_argument("--fetch-original-controls", action="store_true",
                      help="Fetch completed records and whitelist-publish after independent acceptance")
    mode.add_argument("--publish-original-controls", action="store_true",
                      help="Accept already-fetched records; never execute inference")
    mode.add_argument("--prepare-scale-ensemble", action="store_true")
    mode.add_argument("--start-scale-ensemble", action="store_true")
    mode.add_argument("--run-scale-ensemble", action="store_true")
    mode.add_argument("--status-scale-ensemble", action="store_true")
    mode.add_argument("--fetch-scale-ensemble", action="store_true")
    mode.add_argument("--accept-scale-ensemble", action="store_true")
    parser.add_argument("--ensemble-manifest", type=Path, default=None)
    parser.add_argument("--controls-manifest", type=Path, default=None)
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--figure", choices=("2", "3", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.prepare_scale_ensemble or args.start_scale_ensemble or args.status_scale_ensemble or args.fetch_scale_ensemble:
        import sys
        sys.path.insert(0, str(args.workspace_root.resolve() / "experiment_ops"))
        import server_control
        if args.prepare_scale_ensemble or args.start_scale_ensemble:
            prepare_scale_ensemble_manifest(args.workspace_root)
            result = server_control.prepare_scale_ensemble(args.workspace_root)
            if args.start_scale_ensemble and not result.get("blocked"):
                result = server_control.start_scale_ensemble(args.workspace_root)
        elif args.status_scale_ensemble:
            result = server_control.status_scale_ensemble(args.workspace_root)
        else:
            result = server_control.fetch_scale_ensemble(args.workspace_root)
            result["acceptance"] = accept_scale_ensemble(args.workspace_root)
        print(json.dumps({k: result[k] for k in ("state", "started", "prepared", "blocked", "reason", "launch", "successes", "effective_failures", "oom_attempts", "current_trial", "entry", "acceptance") if k in result}, indent=2))
        return True
    if args.run_scale_ensemble:
        print(json.dumps(run_scale_ensemble(args.workspace_root, args.ensemble_manifest or args.workspace_root / SCALE_ENSEMBLE_RELATIVE), indent=2))
        return True
    if args.accept_scale_ensemble:
        print(json.dumps(accept_scale_ensemble(args.workspace_root), indent=2))
        return True
    if args.start_original_controls or args.status_original_controls or args.fetch_original_controls:
        import sys
        sys.path.insert(0, str(args.workspace_root.resolve() / "experiment_ops"))
        import server_control
        if args.start_original_controls:
            prepare_original_controls(args.workspace_root)
            result = server_control.start_original_controls(args.workspace_root)
            print(json.dumps({key: result[key] for key in ("started", "blocked", "reason", "launch") if key in result}, indent=2))
        elif args.status_original_controls:
            result = server_control.status_original_controls(args.workspace_root)
            print(json.dumps({key: result.get(key) for key in ("state", "successes", "expected_calls", "effective_failures", "oom_attempts", "current_trial", "entry")}, indent=2))
        else:
            server_control.fetch_original_controls(args.workspace_root)
            value = json.loads((args.workspace_root / CONTROL_MANIFEST_RELATIVE).read_text(encoding="utf-8"))
            print(json.dumps(publish_original_controls(args.workspace_root, value), indent=2))
        return True
    if args.publish_original_controls:
        value = json.loads((args.workspace_root / CONTROL_MANIFEST_RELATIVE).read_text(encoding="utf-8"))
        print(json.dumps(publish_original_controls(args.workspace_root, value), indent=2))
        return True
    if args.prepare_original_controls:
        value = prepare_original_controls(args.workspace_root)
        print(json.dumps({"status": value["status"], "expected_records": 248, "expected_raw_fvs": 194}))
        return True
    if args.run_original_controls:
        manifest = args.controls_manifest or args.workspace_root / CONTROL_MANIFEST_RELATIVE
        print(json.dumps(run_original_controls(args.workspace_root, manifest), indent=2))
        return True
    if args.verified_figures:
        print(json.dumps(run_verified_figures(args.workspace_root, args.figure,
                                             output_dir=args.output_dir,
                                             overwrite=args.overwrite), indent=2))
    return args.verified_figures


if __name__ == "__main__" and not _verified_figure_cli():
    decycling_dismantling_file_path="result/DiffGDM/decycle_dataset_real_all"
    dismantling_file_path="result/DiffGDM/dismantle_dataset_real_all"
    plt_decycling_dismantling_analysis(dismantling_file_path, decycling_dismantling_file_path)
   # plt_DiffGDM_direction_analysis(dismantling_file_path,decycling_dismantling_file_path)
    #plt_graph_property_analysis()
    #plt_graph_auc_analysis()
   # plt_synth_dismantling()
