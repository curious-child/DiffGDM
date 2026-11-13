import copy
from matplotlib.colors import LinearSegmentedColormap
import igraph
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib as mpl
# plt.rcParams['font.family'] = 'serif'  # 设置主字体族为衬线字体
# plt.rcParams['font.serif'] = ['Times New Roman']
line_color_list=["#8ECFC9","#DAEDF1", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#E7DAD2","r"]
line_style_list=["-","--","-.",":"]
marker_style_list=[".","v",'^','>','<','1','2','3','4','s','p','*','h','H','+','x','D','d',"o",]
def draw_robustness_curves(ax,remove_nodes_dic,network,title=False,data_name=None):
    color_number=1
    marker_number=0
    for method_name,remove_nodes in remove_nodes_dic.items():
        remove_nodes_per, lcc_size_list,_=get_dismantl_curves_data(remove_nodes, network)
        #print("{}:{}".format(method_name,lcc_size))

        colors = ["#B50707", "#FFE4B5"]  # 深红色到橙红色
        n_bins = len(remove_nodes_per)
        cmap = LinearSegmentedColormap.from_list("RedOrange", colors, N=n_bins)

        # 为每个数据点生成对应的颜色
        point_colors = cmap(np.linspace(0, 1, n_bins))

        ax.plot(remove_nodes_per, lcc_size_list,
                label=method_name,
               # color=line_color_list[color_number],
                color="r",
                linestyle="-",
                marker=marker_style_list[marker_number]
                )
        for i, (xi, yi) in enumerate(zip(remove_nodes_per, lcc_size_list)):
            plt.scatter(xi, yi, color=point_colors[i], s=100, zorder=2)  # s控制点的大小
        color_number+=1
        if color_number>=len(line_color_list):
            marker_number+=1
            color_number=0
 #   ax.legend(loc="upper right",framealpha=0.1,prop={"size":7})
    ax.set_xlabel("percentage of removed nodes(%)")
    ax.set_ylabel("LCC size")
    if title==True :
     # ax.set_title("{}:{} nodes,{} edges".format(data_name,network.number_of_nodes(),network.number_of_edges()))
     ax.set_title("{}".format(data_name))
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.grid(linestyle="--")

def get_dismantl_curves_data(remove_nodes,network):
    graph_node_number = network.number_of_nodes()
    graph = copy.deepcopy(network)
    remove_nodes_per = list()
    lcc_size_list = list()
    slcc_size_list = list()
    number_remove = 0.0

    connected_component = list(nx.connected_components(graph))
    lcc = max(connected_component, key=len)
    lcc_size = len(lcc)
    connected_component.remove(lcc)
    if not connected_component:
        slcc_size = 0
    else:
        slcc = max(connected_component, key=len)
        slcc_size = len(slcc)
    remove_nodes_per.append(number_remove / graph_node_number)
    lcc_size_list.append(lcc_size / graph_node_number)
    slcc_size_list.append(slcc_size / graph_node_number)
    # print("method_name:{}".format(method_name))
    for remove_node in remove_nodes:
        # print("remove_node:{}".format(remove_node))
        graph.remove_node(remove_node)
        connected_component = list(nx.connected_components(graph))
        lcc = max(connected_component, key=len)
        lcc_size = len(lcc)
        connected_component.remove(lcc)
        if not connected_component:
            slcc_size = 0
        else:
            slcc = max(connected_component, key=len)
            slcc_size = len(slcc)
        number_remove += 1.
        remove_nodes_per.append(number_remove / graph_node_number)
        lcc_size_list.append(lcc_size / graph_node_number)
        slcc_size_list.append(slcc_size / graph_node_number)
    return remove_nodes_per, lcc_size_list, slcc_size_list
def draw_dismantle_statistics_synth(ax,graphs_dismantle_size,node_size=1000):
    colormap = ["#8ECFC9","#DAEDF1", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#E7DAD2","r"]
    synth_graph_names=["BA","ER","WS",'PL']
    colors_num = len(colormap)

    graphs_name = list(graphs_dismantle_size.keys())
    methods = list(list(graphs_dismantle_size.values())[0].keys())
    methods_num=len(methods)
    graphs_dismantle_size_specific={}

    for method in methods:
        new_method_name = method.replace("OD_", "").replace("_D", "").replace("_no_tag", "").replace(
            "_cost_estimator_", "")
        new_method_name = new_method_name.replace("global_CC", "GCC").replace("average_CC",
                                                                              "ACC").replace(
            "global_efficiency", "GE")
        graphs_dismantle_size_specific[new_method_name] = {}
        for graph_type in synth_graph_names:
            graphs_dismantle_size_specific[new_method_name][graph_type] = []
    for graph_type in synth_graph_names:
        for i,graph_name in enumerate(graphs_name):
            print(graph_name)
            if graph_type=="BA":
                # if "barabasi_n_{}".format(node_size) in graph_name:
                #     for method in methods:
                #         graphs_dismantle_size_specific[method][graph_type].append(graphs_dismantle_size[graph_name][method])
                if "barabasi" in graph_name:
                    for method in methods:
                        #print(method)
                        new_method_name = method.replace("OD_", "").replace("_D", "").replace("_no_tag", "").replace(
                            "_cost_estimator_", "")
                        new_method_name = new_method_name.replace("global_CC", "GCC").replace("average_CC",
                                                                                              "ACC").replace(
                            "global_efficiency", "GE")
                      #  new_methods.append(new_method_name)
                        graphs_dismantle_size_specific[new_method_name][graph_type].append(graphs_dismantle_size[graph_name][method])
            elif graph_type=="ER":
                # if "random_graph_n_{}".format(node_size) in graph_name:
                #     for method in methods:
                #         graphs_dismantle_size_specific[method][graph_type].append(graphs_dismantle_size[graph_name][method])
                if "gnp" in graph_name:
                    for method in methods:
                        new_method_name = method.replace("OD_", "").replace("_D", "").replace("_no_tag", "").replace(
                            "_cost_estimator_", "")
                        new_method_name = new_method_name.replace("global_CC", "GCC").replace("average_CC",
                                                                                              "ACC").replace(
                            "global_efficiency", "GE")

                        graphs_dismantle_size_specific[new_method_name][graph_type].append(
                             graphs_dismantle_size[graph_name][method])
            elif graph_type=="WS":
                if "small" in graph_name:
                    for method in methods:
                        new_method_name = method.replace("OD_", "").replace("_D", "").replace("_no_tag", "").replace(
                            "_cost_estimator_", "")
                        new_method_name = new_method_name.replace("global_CC", "GCC").replace("average_CC",
                                                                                              "ACC").replace(
                            "global_efficiency", "GE")
                        graphs_dismantle_size_specific[new_method_name][graph_type].append(graphs_dismantle_size[graph_name][method])
                # for method in methods:
                #      graphs_dismantle_size_specific[method][graph_type].append(
                #          graphs_dismantle_size[graph_name][method])
            elif graph_type=="PL":
                if "power_law" in graph_name:
                    for method in methods:
                        new_method_name = method.replace("OD_", "").replace("_D", "").replace("_no_tag", "").replace(
                            "_cost_estimator_", "")
                        new_method_name = new_method_name.replace("global_CC", "GCC").replace("average_CC",
                                                                                              "ACC").replace(
                            "global_efficiency", "GE")
                        graphs_dismantle_size_specific[new_method_name][graph_type].append(
                            graphs_dismantle_size[graph_name][method])
                # for method in methods:
                #      graphs_dismantle_size_specific[method][graph_type].append(
                #          graphs_dismantle_size[graph_name][method])

    methods_mean_graphtype={}
    methods_std_graphtype={}
    new_methods=graphs_dismantle_size_specific.keys()
    for method in new_methods:
        methods_mean_graphtype[method]=[]
        methods_std_graphtype[method]=[]
        for graph_type in synth_graph_names:
            methods_mean_graphtype[method].append(np.mean(graphs_dismantle_size_specific[method][graph_type]))
            methods_std_graphtype[method].append(np.std(graphs_dismantle_size_specific[method][graph_type],ddof=1))
    #带误差棒的多数据并列柱图绘图
    x=np.arange(len(synth_graph_names))
    error_attri=dict(elinewidth=1,ecolor="black",capsize=2)
    bar_width=0.1
    max=0
    for i,method in enumerate(new_methods):
        if max<np.max(methods_mean_graphtype[method]):
            max=np.max(methods_mean_graphtype[method])
        ax.bar(x+i*bar_width,methods_mean_graphtype[method],
                bar_width,
                color=colormap[i%colors_num],
                align="center",
                # yerr=methods_std_graphtype[method],
                # error_kw=error_attri,
                label=method
                )
    ax.set_xlabel("Types of synthetic networks")
    ax.set_ylabel("Relative size of solutions")
  #  plt.ylim(0,max+0.05)
    ax.set_xticks(x+methods_num*bar_width/2,synth_graph_names)
   # plt.title("node size {}".format(node_size))
 #   plt.title("node size {}".format(node_size))

    ax.grid(True,axis="y",ls=":",color="gray",alpha=0.2)
   # plt.legend(prop={"size":8})#调整图例大小
    ax.legend(loc="upper right",prop={"size":10},ncol=2)



def draw_dismantle_statistics_real(ax,graphs_dismantle_size, *, label_fontsize=None,
                                  apply_tight_layout=True):
    #colormap = ["b", "g", "r", "c", "m", "y", "k"]
    colormap = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#E7DAD2"]
    colors_num = len(colormap)

    graphs_name = list(graphs_dismantle_size.keys())
    methods=list(list(graphs_dismantle_size.values())[0].keys())
    new_methods = []
    for method in methods:
        new_methods.append(method.replace("_","\n_"))


    methods_all_dismantle_size = np.zeros(len(new_methods))
    bottom_setup = np.zeros(len(new_methods))
    bars_gap = 0.000 * np.ones(len(new_methods))

    draw_x = list(range(len(new_methods)))
    bar_artists, total_artists = [], []
    # top_k=len(graphs_name)
    for i, graph_name in enumerate(graphs_name):
        # ##渐变色 堆积柱状图 alpha调透明度，colar可利用RGB 三元组调色(归一化)
        # red_val = i * (1 / top_k)
        # green_val = 0.4 - i * (0.3 / top_k)
        #
        # ax.bar(x=range(len(graphs_dismantle_size[graph_name].values())),
        #        height=list(graphs_dismantle_size[graph_name].values()),
        #        align="center", bottom=methods_all_dismantle_size, color=(red_val,green_val,0.9),alpha=1,
        #        tick_label=methods, label=graph_name
        #        )

        ## 非渐变色设计
        height = list(1000 * value for value in graphs_dismantle_size[graph_name].values())
        bars = ax.bar(x=draw_x,
               height=height,
               align="center", bottom=bottom_setup, color=colormap[i % colors_num], alpha=1,
               tick_label=new_methods, label=graph_name
               )
        bar_artists.extend({"graph": graph_name, "method": method, "artist": bar}
                           for method, bar in zip(methods, bars))
        bottom_setup += np.array(height) + bars_gap

        methods_all_dismantle_size += 100*np.array(list(graphs_dismantle_size[graph_name].values()))
    # 柱状图标注文本数据
    for x, y,bottom, method in zip(draw_x, methods_all_dismantle_size,bottom_setup, methods):
        artist = ax.text(x, bottom, '%.3f' % y, ha='center', va='bottom',
                         fontsize=label_fontsize)
        total_artists.append({"method": method, "artist": artist})
    # 设置 图像轴是否显示
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # 设置网格线
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', which='both', labelleft=False, left=False)
    ax.tick_params(axis='x', which='both', bottom=False)
    ax.grid(axis='y', linestyle="-", alpha=0.5)
    # plt.axis("off")#
    # 设置标签在图像外侧

   # plt.legend(loc="upper right", framealpha=0.1, prop={"size": 8})


    ax._auc_bar_artists = bar_artists
    ax._auc_total_artists = total_artists
    if apply_tight_layout:
        plt.tight_layout()
def draw_dismantle_methods_graphs_synth(ax, graphs_dismantle_size):
    pass
def draw_dismantle_methods_graphs_real(ax,graphs_dismantle_size,method_list=None,yticklabels=True,cbar=True,
                                     *, verified_cells=None, average=False, graph_order=None,
                                     average_statistics=None):
    if verified_cells is not None:
        methods = (list(next(iter(verified_cells.values())))
                   if method_list is None else list(method_list))
        graphs = list(verified_cells) if graph_order is None else list(graph_order)
        means = np.array([[100 * verified_cells[graph][method]["mean"] for method in methods]
                          for graph in graphs], dtype=float)
        average_values = means.mean(axis=0) if average else None
        matrix = np.vstack((means, average_values)) if average else means
        mesh = ax.pcolormesh(matrix, cmap="Blues", vmin=0, vmax=100,
                             edgecolors="white", linewidth=.4, rasterized=False)
        ax.set_ylim(len(matrix), 0)
        ax.set_xticks(np.arange(len(methods)) + .5, methods, fontsize=7)
        ax.set_yticks(np.arange(len(matrix)) + .5,
                      [*graphs, "Average"] if average else graphs, fontsize=7)
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False,
                       labelbottom=False, length=0, pad=4)
        ax.tick_params(axis="y", length=0, pad=4, labelleft=yticklabels)
        for spine in ax.spines.values():
            spine.set_visible(False)
        numeric_artists = []
        for i, graph in enumerate(graphs):
            lowest = means[i].min()
            for j, method in enumerate(methods):
                item = verified_cells[graph][method]
                text = f"{means[i, j]:.2f}"
                if item.get("seed_num") == 10:
                    text += f"\n±{100 * item['sd']:.2f}"
                is_lowest = np.isclose(means[i, j], lowest, rtol=0, atol=1e-10)
                artist = ax.text(j + .5, i + .5, text, ha="center", va="center",
                                 fontsize=7, linespacing=1.05,
                                 fontweight="bold" if is_lowest else "normal",
                                 color="red" if is_lowest else
                                 ("white" if means[i, j] > 62 else "black"))
                if is_lowest and means[i, j] > 62:
                    artist.set_bbox({"boxstyle": "square,pad=0.05",
                                     "facecolor": "white", "alpha": .8,
                                     "edgecolor": "none"})
                numeric_artists.append({"graph": graph, "method": method, "artist": artist})
        if average:
            ax.axhline(len(graphs), color="#555555", linewidth=.8)
            if average_statistics is not None and set(average_statistics) != set(methods):
                raise ValueError("Average statistics do not match the displayed methods")
            for j, mean in enumerate(average_values):
                text = f"{mean:.2f}"
                if average_statistics is not None:
                    summary = average_statistics[methods[j]]
                    if not np.isclose(mean, 100 * summary["mean"], rtol=0, atol=1e-10):
                        raise ValueError("Average mean differs from the unchanged network means")
                    if summary["rms_within_network_sd"] is not None:
                        text += f"\n±{100 * summary['rms_within_network_sd']:.2f}"
                artist = ax.text(j + .5, len(graphs) + .5, text,
                                 ha="center", va="center", fontsize=7,
                                 linespacing=1.05,
                                 color="white" if mean > 62 else "black")
                numeric_artists.append({"graph": "Average", "method": methods[j],
                                        "artist": artist})
        if cbar:
            plt.colorbar(mesh, ax=ax)
        mesh._verified_numeric_artists = numeric_artists
        return mesh

    values_scale=100
    if method_list is None:
        method_list=list(list(graphs_dismantle_size.values())[0].keys())
    graphs_list=list(graphs_dismantle_size.keys())
    re_graphs_dismantle_size={}
    average_graphs_dismantle_size=[]
    for method in method_list:
        re_graphs_dismantle_size[method]={}
        dismant_size_all=0
        #new_method=method
        for graph in graphs_list:
            re_graphs_dismantle_size[method][graph]=values_scale*graphs_dismantle_size[graph][method]
            dismant_size_all+=values_scale*graphs_dismantle_size[graph][method]

        average_graphs_dismantle_size.append(method.replace("_","\n_")
                                             + "\n" + str(round(dismant_size_all / len(graphs_list), 2)))

    df = pd.DataFrame(re_graphs_dismantle_size)
    print(df)
    row_mins = df.idxmin(axis=1)
    ax1 = sns.heatmap(df, ax=ax, annot=False,yticklabels=yticklabels,vmin=0, vmax=100,cbar=cbar,
                      cbar_kws={
        "shrink": 0.8,      # 长度缩放
        "aspect": 20,       # 长宽比（越大越细）
        "fraction": 0.046,  # 占用主图宽度的比例
        "pad": 0.04         # 与主图的距离
    },
                      square=False, cmap="Blues", fmt='.4g',annot_kws={"color":"black"})
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            value = df.iloc[i, j]
            # 检查当前单元格是否是当前行的最小值所在列
            if df.columns[j] == row_mins[i]:
                # 是行最小值，标红
                text_color = "red"
                font_weight = "bold"
            else:
                # 不是行最小值，标黑
                text_color = "black"
                font_weight = "normal"
            # 在单元格中心添加文本
            ax.text(j + 0.5, i + 0.5, f'{value:.2f}',
                    ha="center", va="center",
                    color=text_color, fontweight=font_weight,
                    #fontsize=10
                    )


    #print(new_xticks)
    #print(ax1.get_yticklabels())
    if yticklabels:
        new_xticks = [0] + ax1.get_xticks().tolist()
        new_xticklabels=["Average                "]+average_graphs_dismantle_size
    else:
        new_xticks = ax1.get_xticks().tolist()
        new_xticklabels=average_graphs_dismantle_size
    ax1.set_xticks(new_xticks)
    ax1.set_xticklabels(new_xticklabels)

    #对自己方法进行特殊标记 醒目 可略去
    # index = int(method_list.index("Stable-AGDM")) + 1  # 因为 xticklabels 相比于methodlist 增加“Average”标签
    # ax1.get_xticklabels()[index].set_color("red")

    plt.setp(ax1.get_xticklabels(), rotation="horizontal")  # "vertical"竖立  horizontal 横放
def draw_statistic_graphs_real(ax,graphs_dismantle_size,method_list=None,yticklabels=True,cbar=True):
    values_scale=1
    if method_list is None:
        method_list=list(list(graphs_dismantle_size.values())[0].keys())
    graphs_list=list(graphs_dismantle_size.keys())
    re_graphs_dismantle_size={}
    average_graphs_dismantle_size=[]
    for method in method_list:
        re_graphs_dismantle_size[method]={}
        dismant_size_all=0
        for graph in graphs_list:
            re_graphs_dismantle_size[method][graph]=values_scale*graphs_dismantle_size[graph][method]
            dismant_size_all+=values_scale*graphs_dismantle_size[graph][method]

        average_graphs_dismantle_size.append(method )
    df = pd.DataFrame(re_graphs_dismantle_size)
    df_norm = df.apply(lambda col:
                       100 * (col - col.min()) / (col.max() - col.min()) if col.max() > col.min() else 0
                       )
    print(df)
    row_mins = df.idxmin(axis=1)
    ax1 = sns.heatmap(df_norm, ax=ax, annot=False,yticklabels=yticklabels,vmin=0, vmax=150,cbar=cbar, square=False,
                      cmap="Blues", fmt='.2e',
                      annot_kws={"color":"black"}
                      )
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            value = df.iloc[i, j]
            # 检查当前单元格是否是当前行的最小值所在列
            # if df.columns[j] == row_mins[i]:
            #     # 是行最小值，标红
            #     text_color = "red"
            #     font_weight = "bold"
            # else:
            #     # 不是行最小值，标黑
            #     text_color = "black"
            #     font_weight = "normal"
            # text_color = "black"
            # font_weight = "normal"
            # 在单元格中心添加文本
            ax.text(j + 0.5, i + 0.5, f'{value:.2e}',
                    ha="center", va="center",
                  #  color=text_color, fontweight=font_weight,
                   # fontsize=10
                    )


   # plt.rc('font', family='Arial', size=15)

    #print(new_xticks)
    #print(ax1.get_yticklabels())

    new_xticks = ax1.get_xticks().tolist()
    new_xticklabels=average_graphs_dismantle_size
    ax1.set_xticks(new_xticks)
    ax1.set_xticklabels(new_xticklabels)

    #对自己方法进行特殊标记 醒目 可略去
    # index = int(method_list.index("Stable-AGDM")) + 1  # 因为 xticklabels 相比于methodlist 增加“Average”标签
    # ax1.get_xticklabels()[index].set_color("red")

    plt.setp(ax1.get_xticklabels(), rotation="horizontal")  # "vertical"竖立  horizontal 横放

def draw_dismantle_methods_synth_curve(ax, graphs_dismantle_size,graph_type="WS"):

    colormap = ["#8ECFC9" ,"#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#E7DAD2","r"]
    colors_num = len(colormap)
    node_sizes = [20, 200, 2000, 20000,
                  200000
                  ]
    #node_sizes = [25, 50, 100, 1000,  ]

    graphs_name = list(graphs_dismantle_size.keys())
    methods = list(list(graphs_dismantle_size.values())[0].keys())
    methods_num = len(methods)
    graphs_dismantle_size_specific = {}
    for method in methods:
        graphs_dismantle_size_specific[method] = {}
        for node_size in node_sizes:
            graphs_dismantle_size_specific[method][node_size] = []

    for node_size in node_sizes:
        for i, graph_name in enumerate(graphs_name):

            # if graph_type == "BA":
            #     if "barabasi_albert_size_{}".format(node_size) in graph_name:
            #         for method in methods:
            #             graphs_dismantle_size_specific[method][node_size].append(
            #                 graphs_dismantle_size[graph_name][method])
            # elif graph_type == "ER":
            #     if "random_graph_n_{}".format(node_size) in graph_name:
            #         for method in methods:
            #             graphs_dismantle_size_specific[method][node_size].append(
            #                 graphs_dismantle_size[graph_name][method])
            # elif graph_type == "WS":
            #     if "small-world_size_{}".format(node_size) in graph_name:
            #         for method in methods:
            #             graphs_dismantle_size_specific[method][node_size].append(
            #                 graphs_dismantle_size[graph_name][method])
            # elif graph_type == "PL":
            #     if "static_power_law_size_{}".format(node_size) in graph_name:
            #         for method in methods:
            #             graphs_dismantle_size_specific[method][node_size].append(
            #                 graphs_dismantle_size[graph_name][method])
            if "_{}-".format(node_size) in graph_name:
                for method in methods:
                    graphs_dismantle_size_specific[method][node_size].append(
                        graphs_dismantle_size[graph_name][method])
    methods_mean_nodesize = {}
    methods_std_nodesize = {}
    for method in methods:
        methods_mean_nodesize[method] = []
        methods_std_nodesize[method] = []
        for node_size in node_sizes:
            methods_mean_nodesize[method].append(100*np.mean(graphs_dismantle_size_specific[method][node_size]))
            methods_std_nodesize[method].append(np.std(graphs_dismantle_size_specific[method][node_size], ddof=1))

    for i, method in enumerate(methods):
        ax.errorbar(node_sizes, methods_mean_nodesize[method],
                     fmt=":.",
                     ms=5,#数据点大小
                    mfc=colormap[i % colors_num],#数据点颜色
                    # mec=colormap[i % colors_num],#数据点的标记边缘颜色
                    color=colormap[i % colors_num],
                   # yerr=methods_std_nodesize[method],
                  #  ecolor=colormap[i % colors_num],elinewidth=1,capsize=2,
                    label=method
                )
    ax.set_ylabel("Relative size of dismantling set (%)")
    ax.set_ylim(0,100)
    # ax.set_xlabel("Network size")
    # ax.set_xscale("symlog")
    # x_ticks = ["20-30", "200-300", "2k-3k", "20k-30k", "200k-300k"]
    # ax.set_xticks(node_sizes, x_ticks)
    # ax.set_title("{}".format(graph_type))
    # ax.grid(True, axis="y", ls=":", color="gray", alpha=0.2)
    # ax.legend(prop={"size": 8},loc="lower left",shadow=False,framealpha=0.5)  # 调整图例大小

def draw_runtime_methods_synth_curve(ax, graphs_dismantle_size):

    colormap = ["#8ECFC9" ,"#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#E7DAD2","r"]
    colors_num = len(colormap)
    node_sizes = [20, 200, 2000, 20000,
                  200000
                  ]

    graphs_name = list(graphs_dismantle_size.keys())
    methods = list(list(graphs_dismantle_size.values())[0].keys())
    methods_num = len(methods)
    graphs_dismantle_size_specific = {}
    for method in methods:
        graphs_dismantle_size_specific[method] = {}
        for node_size in node_sizes:
            graphs_dismantle_size_specific[method][node_size] = []

    for node_size in node_sizes:
        for i, graph_name in enumerate(graphs_name):

            if "_{}-".format(node_size) in graph_name:
                for method in methods:
                    graphs_dismantle_size_specific[method][node_size].append(
                        graphs_dismantle_size[graph_name][method])
    methods_mean_nodesize = {}
    methods_std_nodesize = {}

    for method in methods:
        methods_mean_nodesize[method] = []

        runtime_mean_min = np.mean(graphs_dismantle_size_specific[method][20])
        for node_size in node_sizes:
            methods_mean_nodesize[method].append(np.mean(graphs_dismantle_size_specific[method][node_size])/runtime_mean_min)
           # methods_std_nodesize[method].append(np.std(graphs_dismantle_size_specific[method][node_size], ddof=1))


    for i, method in enumerate(methods):
        ax.errorbar(node_sizes, methods_mean_nodesize[method],
                    fmt=":o",
                    ms=5,  # 数据点大小
                    mfc=colormap[i % colors_num],  # 数据点颜色
                    # mec=colormap[i % colors_num],#数据点的标记边缘颜色
                    color=colormap[i % colors_num],
                    # yerr=methods_std_nodesize[method],
                    #  ecolor=colormap[i % colors_num],elinewidth=1,capsize=2,
                    label=method
                    )
    ax.set_ylabel("Relative increase of runtime")
    ax.set_xlabel("Network size")


    ax.set_xscale("symlog")
    x_ticks=["20-30","200-300","2k-3k","20k-30k","200k-300k"]
    ax.set_xticks(node_sizes,x_ticks)
   # ax.set_title("{}".format(graph_type))
    ax.grid(True, axis="y", ls=":", color="gray", alpha=0.2)
    ax.legend(prop={"size": 8}, loc="best", shadow=False, framealpha=0.5,ncol=2)  # 调整图例大小


def draw_memory_methods_synth_curve(ax, graphs_dismantle_size):
    colormap = ["#8ECFC9", "#DAEDF1", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#E7DAD2", "r"]
    colors_num = len(colormap)
    node_sizes = [20, 200, 2000, 20000,
                  200000
                  ]

    graphs_name = list(graphs_dismantle_size.keys())
    methods = list(list(graphs_dismantle_size.values())[0].keys())
    methods_num = len(methods)
    graphs_dismantle_size_specific = {}
    for method in methods:
        graphs_dismantle_size_specific[method] = {}
        for node_size in node_sizes:
            graphs_dismantle_size_specific[method][node_size] = []

    for node_size in node_sizes:
        for i, graph_name in enumerate(graphs_name):

            if "_{}-".format(node_size) in graph_name:
                for method in methods:
                    graphs_dismantle_size_specific[method][node_size].append(
                        graphs_dismantle_size[graph_name][method])
    methods_mean_nodesize = {}
    methods_std_nodesize = {}

    for method in methods:
        methods_mean_nodesize[method] = []
        methods_std_nodesize[method] = []

        for node_size in node_sizes:
            methods_mean_nodesize[method].append(
                np.mean(graphs_dismantle_size_specific[method][node_size]) )
            methods_std_nodesize[method].append(np.std(graphs_dismantle_size_specific[method][node_size], ddof=1))

    for i, method in enumerate(methods):
        ax.errorbar(node_sizes, methods_mean_nodesize[method],
                    fmt="-o",
                    ms=5,  # 数据点大小
                    mfc=colormap[i % colors_num],  # 数据点颜色
                     mec=colormap[i % colors_num],#数据点的标记边缘颜色
                    color=colormap[i % colors_num],
                    #yerr=methods_std_nodesize[method],
                     # ecolor=colormap[i % colors_num],elinewidth=1,capsize=2,
                    label=method
                    )
    ax.set_xlabel("Network size")
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_xscale("symlog")
    x_ticks = ["20-30", "200-300", "2k-3k", "20k-30k", "200k-300k"]
    ax.set_xticks(node_sizes,x_ticks)
    # ax.set_title("{}".format(graph_type))
    ax.grid(True, axis="y", ls=":", color="gray", alpha=0.2)
    ax.legend(prop={"size": 8}, loc="best", shadow=False, framealpha=0.5)  # 调整图例大小
def draw_paper_resource_comparison(runtime_records, dismantling_records,
                                  ensemble_overlay=None, cfn_limits=None):
    """Assemble the original runtime and CFN curves without changing records.

    Runtime retains each method's smallest-scale normalization. CFN is the
    mean across the ten graph instances at each scale. The stored CoreDiffGDM
    key is mapped only in this in-memory plotting input to the paper label.
    """
    methods = ("CoreHD", "BPD", "DiffGDM", "MinSum")
    runtime = {graph: {method: values[method] for method in methods}
               for graph, values in runtime_records.items()}
    dismantling = {graph: {method: values["CoreDiffGDM" if method == "DiffGDM" else method]
                           for method in methods}
                   for graph, values in dismantling_records.items()}
    if ensemble_overlay is not None:
        if (set(ensemble_overlay["cfn"]) != set(dismantling)
                or set(ensemble_overlay["inference_seconds"]) != set(runtime)):
            raise ValueError("Ensemble overlay must cover the same original graphs")
        methods += ("DiffGDM_ens",)
        for graph in runtime:
            runtime[graph]["DiffGDM_ens"] = ensemble_overlay["inference_seconds"][graph]
            dismantling[graph]["DiffGDM_ens"] = ensemble_overlay["cfn"][graph]
    markers = ("o", "s", "^", "D", "v")
    line_styles = (":", "--", "-", "-.", (0, (3, 1, 1, 1)))
    with plt.rc_context({'font.family': 'Arial', 'font.size': 10,
                         'pdf.fonttype': 42, 'text.usetex': False}):
        fig, axes = plt.subplots(2, 1, figsize=(4, 4.9))
        draw_runtime_methods_synth_curve(axes[0], runtime)
        axes[0].get_legend().remove()
        axes[0].set_xlabel("")
        draw_dismantle_methods_synth_curve(axes[1], dismantling)
        axes[1].set_ylabel("CFN (%)")
        if ensemble_overlay is None:
            axes[1].set_ylim(25, 80)
            axes[1].set_yticks(range(30, 81, 10))
        else:
            if cfn_limits is None:
                means = [value for line in axes[1].lines for value in line.get_ydata()]
                padding = max(1., .05 * (max(means) - min(means)))
                cfn_limits = (max(0., 5 * np.floor((min(means) - padding) / 5)),
                              min(100., 5 * np.ceil((max(means) + padding) / 5)))
            axes[1].set_ylim(*cfn_limits)
            step = 10 if cfn_limits[1] - cfn_limits[0] > 35 else 5
            axes[1].set_yticks(np.arange(step * np.ceil(cfn_limits[0] / step),
                                        cfn_limits[1] + .01, step))
        axes[1].set_xlabel("Network size")
        axes[1].set_xscale("symlog")
        axes[1].set_xlim(axes[0].get_xlim())
        node_sizes = (20, 200, 2000, 20000, 200000)
        tick_labels = ("20–30", "200–300", "2K–3K", "20K–30K", "200K–300K")
        for panel, axis in zip("ab", axes):
            axis.set_xticks(node_sizes, tick_labels)
            axis.tick_params(axis="both", labelsize=9.5, length=3, pad=3)
            axis.grid(True, axis="y", ls=":", color="gray", alpha=.2)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            for index, line in enumerate(axis.lines):
                line.set_marker(markers[index])
                line.set_linestyle(line_styles[index])
                line.set_markersize(6 if index != 2 else 5)
                line.set_markerfacecolor("none")
                line.set_markeredgewidth(1)
                line.set_linewidth(1.15)
                line.set_zorder(5 if index == 2 else 3)
            axis.text(0, 1.03, panel, transform=axis.transAxes,
                      fontsize=12, fontweight="bold", va="bottom", ha="left")
        fig.legend([container.lines[0] for container in axes[0].containers], methods,
                   loc="upper center", bbox_to_anchor=(.54, .997),
                   ncol=4 if ensemble_overlay is None else 3,
                   fontsize=9.5, frameon=False, handlelength=1.6,
                   handletextpad=.4, columnspacing=.8)
        fig.subplots_adjust(left=.20, right=.92, bottom=.105,
                            top=.87 if ensemble_overlay is None else .82, hspace=.42)
        fig._verified_data_axes = tuple(axes)
        fig._exact_export_size = True
        _publication_font_floor(fig, 9.5, 78.2)
    return fig



def draw_relative_resource_comparison(payload):
    """Redraw the accepted scale comparison, including the original inset.

    The recorded stage-time means supply the runtime series. All individual
    graph CFNs, the random baseline and the five-scale ratios remain available
    in the accompanying source data.
    """
    from matplotlib.patches import ConnectionPatch, Rectangle
    from matplotlib.ticker import FixedLocator, MultipleLocator, StrMethodFormatter
    import math
    statistics = payload["statistics"]
    methods, scales = payload["methods"], payload["scales"]
    normalized = statistics["normalized_cfn"]
    matrix = [normalized[m] for m in methods] if isinstance(normalized, dict) else normalized
    prior_log_manifest = {"marker_size_pt": payload["marker_size_pt"]}
    def _same(a, b):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    fig = draw_paper_resource_comparison(payload["runtime_records"],
        payload["cfn_records"], ensemble_overlay=payload["ensemble"])
    for line, method in zip(fig._verified_data_axes[0].lines, methods):
        line.set_ydata(statistics["runtime_relative"][method])
    a, b = fig._verified_data_axes
    a.set_yscale('log', base=10)
    a.set_ylim(.72, 1.45 * max(max(values) for values in statistics['runtime_relative'].values()))
    a.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000, 10000]))
    a.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    a.tick_params(axis='y', which='minor', left=False)
    a.set_ylabel('Relative runtime (×)', fontsize=10)
    for line in a.lines:
        line.set_markersize(prior_log_manifest['marker_size_pt'])
    if (len(a.lines) != 5 or any(list(line.get_xdata(orig=False)) != scales
           or any(not _same(y, expected) or not a.get_ylim()[0] <= y <= a.get_ylim()[1]
                  for y, expected in zip(line.get_ydata(orig=False), statistics['runtime_relative'][m]))
           for line, m in zip(a.lines, methods))):
        raise ValueError('The log-axis preview must retain all 25 original runtime points')

    b.clear()
    b.set_position([a.get_position().x0, .131, a.get_position().width, .291])
    b.set_xscale(a.get_xscale())
    b.set_xlim(a.get_xlim())
    scale_labels = [label.get_text() for label in a.get_xticklabels()]
    scale_labels[-2:] = [label.replace('–', '–\n') for label in scale_labels[-2:]]
    b.set_xticks(scales, scale_labels)
    values = [v for row in matrix for v in row if v is not None]
    b.set_ylim(min(.86, min(values, default=1) - .008),
               max(1.02, max(values, default=1) + .008))
    b.yaxis.set_major_locator(MultipleLocator(.04))
    b.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
    b.tick_params(axis='both', labelsize=9.5, length=3, pad=3)
    b.grid(True, axis='y', ls=':', color='gray', alpha=.22)
    b.spines[['top', 'right']].set_visible(False)
    b.set_xlabel('Network size', fontsize=10, labelpad=3)
    b.set_ylabel('Relative CFN', fontsize=10)
    b.text(0, 1.03, 'b', transform=b.transAxes, fontsize=12, fontweight='bold', va='bottom')
    b.axhline(1, color='#69737c', lw=.85, ls='--', zorder=2)
    b.annotate('Random-Decycle', (.98, 1), xycoords=b.get_yaxis_transform(),
               xytext=(0, 3), textcoords='offset points', ha='right', va='bottom',
               color='#4d5863', fontsize=9.5)

    line_styles = (':', '--', '-', '-.', (0, (3, 1, 1, 1)))
    points_checked = 0
    final_lines = {}
    for method, row, source, linestyle in zip(methods, matrix, a.lines, line_styles):
        # Break the line at an undefined ratio; never move or fill in points.
        present = np.flatnonzero([v is not None for v in row])
        actual = []
        for run in np.split(present, np.flatnonzero(np.diff(present) > 1) + 1):
            if not len(run):
                continue
            line, = b.plot([scales[j] for j in run], [row[j] for j in run],
                           color=source.get_color(), ls=linestyle, lw=1.12,
                           marker=source.get_marker(), ms=3.25, mfc='none', mew=.9,
                           label=method, zorder=source.get_zorder())
            actual.extend(zip(line.get_xdata(orig=False), line.get_ydata(orig=False)))
            if run[-1] == len(scales) - 1:
                final_lines[method] = (line, linestyle)
        if actual != [(scale, v) for scale, v in zip(scales, row) if v is not None]:
            raise ValueError('Figure 8b moved or altered an accepted ratio for ' + method)
        points_checked += len(actual)

    inset = None
    inset_data = None
    if len(final_lines) == len(methods):
        # Give the lower panel more room, keeping the runtime panel fixed from the top.
        width, height = fig.get_size_inches()
        extra_height = .95
        new_height = height + extra_height
        top_position = a.get_position().frozen()
        lower_position = b.get_position().frozen()
        legend = fig.legends[0]
        anchor = legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        fig.set_size_inches(width, new_height)
        a.set_position([top_position.x0, (top_position.y0 * height + extra_height) / new_height,
                        top_position.width, top_position.height * height / new_height])
        legend.set_bbox_to_anchor((anchor.x0, (anchor.y0 * height + extra_height) / new_height),
                                  transform=fig.transFigure)
        b.set_position([lower_position.x0, lower_position.y0 * height / new_height,
                        lower_position.width,
                        (lower_position.height * height + extra_height) / new_height])

        inset = b.inset_axes([.415, .045, .330, .289], zorder=9)
        inset.set_xscale(b.get_xscale())
        last = scales[-1]
        x0, x1 = last * .82, last * 1.16
        endpoints = {method: row[-1] for method, row in zip(methods, matrix)}
        # CoreHD and MinSum are already visually separated at the last scale.
        zoom_methods = ('BPD', 'DiffGDM', 'DiffGDM_ens')
        final_values = {method: endpoints[method] for method in zoom_methods}
        y0, y1 = min(final_values.values()) - .001, max(final_values.values()) + .0035
        if any(y0 <= value <= y1 for method, value in endpoints.items() if method not in zoom_methods):
            raise ValueError('The largest-scale zoom would omit a point inside the selected range')
        inset.set_xlim(x0, x1)
        inset.set_ylim(y0, y1)
        inset.set_xticks([])
        inset.xaxis.set_minor_locator(FixedLocator([]))
        inset.yaxis.set_major_locator(MultipleLocator(.002))
        inset.yaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
        inset.yaxis.tick_right()
        inset.tick_params(axis='y', length=2, pad=3, labelsize=9.5, colors='#4d5863')
        inset.grid(True, axis='y', ls=':', color='#6c7880', alpha=.27)
        inset.text(.96, .97, '200K–300K', transform=inset.transAxes,
                   ha='right', va='top', fontsize=9.5, color='#29333b',
                   bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .7})
        for spine in inset.spines.values():
            spine.set_linewidth(.7)
            spine.set_color('#77828a')
        for method in zoom_methods:
            source, linestyle = final_lines[method]
            line, = inset.plot(source.get_xdata(orig=False), source.get_ydata(orig=False),
                               color=source.get_color(), ls=linestyle, lw=.92,
                               marker=source.get_marker(), ms=2.6, mfc='none', mew=.82,
                               zorder=source.get_zorder())
            if (list(line.get_xdata(orig=False)) != list(source.get_xdata(orig=False))
                    or list(line.get_ydata(orig=False)) != list(source.get_ydata(orig=False))
                    or line.get_xdata(orig=False)[-1] != last
                    or line.get_ydata(orig=False)[-1] != final_values[method]):
                raise ValueError('The enlargement must clip the original lines without moving any point')
        b.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                              fill=False, lw=.7, ec='#77828a', ls='--', zorder=2.5))
        b.add_artist(ConnectionPatch((x0, y0), (1, 1), coordsA=b.transData,
                                     coordsB=inset.transAxes, color='#77828a', lw=.7,
                                     ls='--', zorder=2.5))
        inset_data = {'scale': last, 'methods': list(zoom_methods),
                      'x_limits': [x0, x1], 'y_limits': [y0, y1],
                      'final_values': final_values, 'source': 'clipped original curve segments'}
    return fig


def threshold_dismantlesize_curve(ax,graphs_dismantle_size,methods_list,graph_property=None):
    color_number = 0
    marker_number = 0
    for method_name in methods_list:
        threshold_list=list(graphs_dismantle_size[method_name].keys())
        dismantle_list=list(graphs_dismantle_size[method_name].values())

        ax.plot(threshold_list,dismantle_list,
                label=method_name,
                color=line_color_list[color_number],
                linestyle="-",
                marker=marker_style_list[marker_number]
                )
        color_number += 1
        if color_number >= len(line_color_list):
            marker_number += 1
            color_number = 0
    ax.legend(loc="upper right", framealpha=0.1,prop={"size":7})
    ax.set_xlabel("dismantling threshold(%)")
    if graph_property is not None:
        ax.set_title("Relative size of {}".format(graph_property))
    else:
        ax.set_ylabel("Relative size of NDS")


def draw_verified_synthetic_comparison(payload):
    """Return the accepted synthetic comparison, without its in-figure footer.

    The caller supplies validated values and owns all saving and source-data
    bookkeeping. No input values or existing plotting defaults are changed.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
    colors = ("#8ECFC9", "#DAEDF1", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC")
    labels = list(payload["family_graphs"])
    methods = payload["methods"]
    width = .12
    x = np.arange(len(labels))
    for panel, ax in zip(("fig2a", "fig2b"), axes):
        for i, method in enumerate(methods):
            values = [payload["panels"][panel]["family_means"][method][f]
                      for f in labels]
            ax.bar(x + (i - (len(methods) - 1) / 2) * width, values, width,
                   label=method, color=colors[i], edgecolor="white", linewidth=.4)
        ax.text(-.08, 1.02, panel[-1], transform=ax.transAxes,
                fontsize=14, fontweight="bold")
        ax.set_ylabel("Relative FVS size" if panel == "fig2a" else "CFN")
        ax.grid(axis="y", linestyle=":", alpha=.25)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        highest = max(v for values in payload["panels"][panel]["family_means"].values()
                      for v in values.values())
        ax.set_ylim(0, min(1.1, max(.05, highest * 1.23)))
    axes[0].legend(ncol=3, fontsize=9, frameon=False, loc="upper left")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("Types of synthetic networks")
    fig.tight_layout(rect=(0, .075, 1, 1))
    fig._verified_data_axes = tuple(axes)
    return fig


def draw_verified_route_comparison(payload):
    """Return horizontal route heatmaps from validated cells in payload order.

    Ten-seed cells display their supplied sample SD. Eligible Average cells
    display the RMS within-network SD; historical scalars receive no invented SD.
    This function never imports or executes model or postprocessing code.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    panel_keys = ("a", "b", "c")
    names = list(payload["graphs"])
    panels = payload["panels"]
    counts = tuple(len(panels[key]["methods"]) for key in panel_keys)
    expected_counts = ((6, 7, 7) if payload.get("profile") == "original-plus-controls"
                       else (6, 7, 6))
    if len(names) != 27 or counts != expected_counts:
        raise ValueError("Route comparison network/method coverage differs from its explicit profile")

    canvas_width, canvas_height = 280.0, 195.0
    fig = plt.figure(figsize=(canvas_width / 25.4, canvas_height / 25.4))
    left_mm, bottom_mm, height_mm = 36.0, 11.0, 165.0
    gap_mm, panel_span_mm = 5.0, 219.0
    column_mm = (panel_span_mm - 2 * gap_mm) / sum(counts)
    axes = []
    cursor_mm = left_mm
    for count in counts:
        width_mm = count * column_mm
        ax = fig.add_axes((cursor_mm / canvas_width, bottom_mm / canvas_height,
                           width_mm / canvas_width, height_mm / canvas_height),
                          sharey=axes[0] if axes else None)
        axes.append(ax)
        cursor_mm += width_mm + gap_mm

    labels = {"DiffGDM_ens": "DiffGDM\nens", "Random-Decycle": "Random-\nDecycle",
              "T2T-FVS": "T2T-FVS", "T2T_decycle": "T2T-\ndecycle",
              "T2T_direct": "T2T-\ndirect", "Random_direct": "Random\n_direct"}
    titles = {"a": "FVS", "b": "Decycling-to-dismantling CFN",
              "c": "Direct-removal CFN"}
    display_names = payload.get("display_names", {})

    def wrap_network_label(name):
        label = display_names.get(name, name).replace("\\n", "")
        label = label.replace("\n", "").replace("\r", "")
        if len(label) <= 18:
            return label
        cuts = [i + 1 for i, character in enumerate(label[:-1])
                if character in "-_ "]
        balanced = [cut for cut in cuts if max(cut, len(label) - cut) <= 18]
        candidates = balanced or cuts
        cut = (min(candidates, key=lambda position: abs(position - len(label) / 2))
               if candidates else len(label) // 2)
        return label[:cut] + "\n" + label[cut:]

    network_labels = [wrap_network_label(name) for name in names]
    last_mesh = None
    numeric_artists = []
    for index, (panel, ax) in enumerate(zip(panel_keys, axes)):
        methods = panels[panel]["methods"]
        values = panels[panel]["graph_values"]
        last_mesh = draw_dismantle_methods_graphs_real(
            ax, values, method_list=methods, yticklabels=index == 0, cbar=False,
            verified_cells=values, average=True, graph_order=names,
            average_statistics=panels[panel]["average_statistics"])
        numeric_artists.extend({"panel": panel, **item}
                               for item in last_mesh._verified_numeric_artists)
        ax.set_xticks(np.arange(len(methods)) + .5,
                      [labels.get(method, method) for method in methods], fontsize=7)
        ax.set_yticks(np.arange(len(names) + 1) + .5,
                      [*network_labels, "Average"], fontsize=7)
        if index == 0:
            for label in ax.get_yticklabels():
                label.set_linespacing(1.0)
            ax.get_yticklabels()[-1].set_fontweight("bold")
        ax.text(0, 1.068, panel, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom", ha="left")
        ax.text(.09, 1.068, titles[panel], transform=ax.transAxes,
                fontsize=8.5, va="bottom", ha="left")

    color_axis = fig.add_axes((263.0 / canvas_width, bottom_mm / canvas_height,
                               3.0 / canvas_width, height_mm / canvas_height))
    colorbar = fig.colorbar(last_mesh, cax=color_axis)
    colorbar.set_ticks((0, 20, 40, 60, 80, 100))
    colorbar.ax.tick_params(labelsize=7, length=2, pad=2)
    colorbar.set_label("Relative size (%)", fontsize=7, labelpad=3)
    fig._verified_data_axes = tuple(axes)
    fig._verified_numeric_artists = numeric_artists
    return fig


def _publication_font_floor(fig, minimum_pt, embedded_width_mm):
    """Keep text readable after the declared CAS-column scaling."""
    from matplotlib.text import Text
    fig.canvas.draw()
    for text in fig.findobj(Text):
        if text.get_fontsize() < minimum_pt:
            text.set_fontsize(minimum_pt)
    fig._minimum_export_font = minimum_pt
    fig._embedded_width_mm = embedded_width_mm


def draw_verified_guidance_comparison(payload):
    """Draw the historical 27-network, 6/7-guide comparison at final print size."""
    names = list(payload["graphs"])
    panels = payload["panels"]
    if (payload.get("profile") != "original-plus-controls" or len(names) != 27
            or tuple(len(panels[key]["methods"]) for key in "ab") != (6, 7)):
        raise ValueError("Historical guidance heatmaps require 27 networks and 6/7 methods")

    def wrapped(name):
        text = name.replace("\\n", "").replace("\n", "").replace("\r", "")
        if len(text) <= 18:
            return text
        cuts = [index + 1 for index, char in enumerate(text[:-1]) if char in "-_ "]
        candidates = [cut for cut in cuts if max(cut, len(text) - cut) <= 18] or cuts
        cut = min(candidates, key=lambda item: abs(item - len(text) / 2)) if candidates else len(text) // 2
        return text[:cut] + "\n" + text[cut:]

    with mpl.rc_context({"font.family": "Arial", "font.size": 7,
                         "pdf.fonttype": 42, "text.usetex": False}):
        fig = plt.figure(figsize=(180 / 25.4, 180 / 25.4))
        column_mm = 123 / 13
        axes, artists = [], []
        left = 33.0
        for index, key in enumerate("ab"):
            panel = panels[key]
            width = len(panel["methods"]) * column_mm
            axis = fig.add_axes((left / 180, 8 / 180, width / 180, 157 / 180))
            axes.append(axis)
            mesh = draw_dismantle_methods_graphs_real(
                axis, panel["graph_values"], method_list=panel["methods"],
                yticklabels=index == 0, cbar=False, verified_cells=panel["graph_values"],
                average=True, graph_order=names, average_statistics=panel["average_statistics"])
            axis.set_xticks(np.arange(len(panel["methods"])) + .5,
                            [name.removeprefix("DiffGDM_") for name in panel["methods"]], fontsize=7)
            axis.set_yticks(np.arange(28) + .5, [*[wrapped(name) for name in names], "Average"],
                            fontsize=7)
            axis.tick_params(axis="y", labelleft=index == 0, pad=3)
            for label in axis.get_yticklabels():
                label.set_linespacing(1.0)
            if index == 0:
                axis.get_yticklabels()[-1].set_fontweight("bold")
            axis.text(0, 1.055, key, transform=axis.transAxes, fontsize=9,
                      fontweight="bold", ha="left", va="bottom")
            axis.text(.1, 1.055, "FVS (%)" if key == "a" else "CFN (%)",
                      transform=axis.transAxes, fontsize=8.5, ha="left", va="bottom")
            # Compare unrounded values, not equal-looking printed labels.
            for item in mesh._verified_numeric_artists:
                values = (panel["average_statistics"] if item["graph"] == "Average"
                          else panel["graph_values"][item["graph"]])
                value = values[item["method"]]["mean"]
                is_lowest = value == min(cell["mean"] for cell in values.values())
                item["artist"].set_color("red" if is_lowest else "white" if 100 * value > 62 else "black")
                item["artist"].set_fontweight("bold" if is_lowest else "normal")
                item["artist"].set_bbox({"boxstyle": "square,pad=0.05", "facecolor": "white",
                                         "alpha": .8, "edgecolor": "none"}
                                        if is_lowest and 100 * value > 62 else None)
                artists.append({"panel": key, **item})
            left += width + 4
        color_axis = fig.add_axes((165 / 180, 8 / 180, 2.5 / 180, 157 / 180))
        colorbar = fig.colorbar(mesh, cax=color_axis)
        colorbar.set_ticks((0, 20, 40, 60, 80, 100))
        colorbar.ax.tick_params(labelsize=7, length=2, pad=2)
        colorbar.set_label("Relative size (%)", fontsize=7, labelpad=3)
        fig._verified_data_axes = tuple(axes)
        fig._verified_numeric_artists = artists
        fig._exact_export_size = True
        _publication_font_floor(fig, 7.4, 173.8)
        return fig


def draw_guidance_benefit_analysis(payload):
    """Return paired main-text and supplementary figures from one cached source.

    Category offsets only separate overlapping marks; all CFN values and lower
    scatter coordinates remain exact. Historical case members stay in the source.
    """
    from collections import Counter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator

    methods, graphs = list(payload["methods"]), list(payload["graphs"])
    if methods != ["NC", "Density", "GE"] or len(graphs) != 27 or len(set(graphs)) != 27:
        raise ValueError("The analysis requires NC, Density, GE and 27 distinct networks")
    outcome_keys = ("improved", "tied", "worse")
    direction_keys = ("decreased", "tied", "increased")
    metric_directions = ("decreased", "unchanged", "increased")

    def outcome_of(point):
        change = point["final_nodes"] - point["base_final_nodes"]
        return "improved" if change < 0 else "worse" if change > 0 else "tied"

    all_points = []
    for method in methods:
        points = payload["points"][method]
        if len(points) != 27 or {point["graph"] for point in points} != set(graphs):
            raise ValueError("Network coverage differs: " + method)
        counts = {row: {column: 0 for column in direction_keys} for row in outcome_keys}
        for point in points:
            x, y = point["delta_trees"], point["delta_cfn_pp"]
            if not np.isfinite([x, y]).all() or float(x) != int(x):
                raise ValueError("Scatter coordinates must be finite, with integer tree counts")
            if x != point["oversized_trees"] - point["base_oversized_trees"]:
                raise ValueError("Tree-count change differs from its paired endpoints")
            outcome = outcome_of(point)
            if ((outcome == "improved" and y >= 0) or
                    (outcome == "worse" and y <= 0) or (outcome == "tied" and y != 0)):
                raise ValueError("CFN direction differs from final deletion counts")
            direction = "decreased" if x < 0 else "increased" if x > 0 else "tied"
            counts[outcome][direction] += 1
        if counts != payload["direction_counts"][method]:
            raise ValueError("Direction counts differ from scatter data: " + method)
        if {row: sum(counts[row].values()) for row in outcome_keys} != payload["outcomes"][method]:
            raise ValueError("Outcome totals differ from scatter data: " + method)
        all_points.extend(points)

    if payload.get("schema_version") != 4:
        raise ValueError("The expanded analysis requires the verified schema-4 metric cache")
    metric_cache = payload["metric_decrease_summary"]
    fields = metric_cache["row_fields"]
    rows = {method: [dict(zip(fields, raw)) for raw in metric_cache["rows"][method]]
            for method in methods}
    for method in methods:
        if [row["graph"] for row in rows[method]] != graphs:
            raise ValueError("Metric rows must retain every network in the original order")

    width_mm, height_mm = 88.0, 140.0
    metric_width_mm, metric_canvas_height = 180.0, 70.0
    axes_width, metric_height, count_height = 48.0, 50.0, 30.0
    minimum_spacing_pt = 5.0
    outcomes = (("improved", "#2877A5", "o"), ("tied", "#858B91", "s"),
                ("worse", "#D88735", "^"))
    x_values = [int(point["delta_trees"]) for point in all_points]
    y_values = [float(point["delta_cfn_pp"]) for point in all_points]
    x_min, x_max = min(0, min(x_values)), max(0, max(x_values))
    x_padding = max((x_max - x_min) * .09, .8)
    y_min, y_max = min(0.0, min(y_values)), max(0.0, max(y_values))
    y_padding = max((y_max - y_min) * .09, .06)
    ylim = (y_min - y_padding, y_max + y_padding)

    def category_offsets(selected):
        """Deterministic, physical-distance placement without any RNG calls."""
        x_scale = axes_width / 25.4 * 72 / 3.1
        y_scale = metric_height / 25.4 * 72 / (ylim[1] - ylim[0])
        order = {graph: index for index, graph in enumerate(graphs)}
        placed, result = [], {}
        for row in sorted(selected, key=lambda item: (item["delta_cfn_pp"], order[item["graph"]])):
            y = row["delta_cfn_pp"] * y_scale
            candidates = [0.0]
            for px, py in placed:
                dy = abs(y - py)
                if dy < minimum_spacing_pt:
                    dx = np.sqrt(max(0.0, minimum_spacing_pt ** 2 - dy ** 2))
                    candidates.extend((px - dx, px + dx))
            available = [x for x in candidates if abs(x / x_scale) <= .34
                         and all((x - px) ** 2 + (y - py) ** 2 >=
                                 minimum_spacing_pt ** 2 - 1e-8 for px, py in placed)]
            if not available:
                raise ValueError("Category marks cannot fit without overlap or exceeding the allowed offset")
            x = min(available, key=lambda value: (abs(value), value))
            placed.append((x, y))
            result[row["graph"]] = x / x_scale
        return result

    with mpl.rc_context({"font.family": "Arial", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "axes.linewidth": .65,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
        metric_fig = plt.figure(figsize=(metric_width_mm / 25.4, metric_canvas_height / 25.4))

        def add_axes_mm(figure, left, bottom, width, height):
            canvas_width, canvas_height = (width_mm, height_mm) if figure is fig else (
                metric_width_mm, metric_canvas_height)
            return figure.add_axes((left / canvas_width, bottom / canvas_height,
                                     width / canvas_width, height / canvas_height))

        outcome_labels = {"improved": "Improved", "tied": "Unchanged", "worse": "Worse"}
        handles = [Line2D([], [], color=color, marker=marker, linestyle="none",
                          markersize=4.8, label=outcome_labels[outcome])
                   for outcome, color, marker in outcomes]
        fig.legend(handles=handles, loc="center", ncol=3, frameon=False,
                   bbox_to_anchor=(.53, 137 / height_mm), fontsize=7.5,
                   columnspacing=.75, handletextpad=.4, handlelength=1.0)
        metric_fig.legend(handles=handles, loc="center", ncol=3, frameon=False,
                          bbox_to_anchor=(.55, 67 / metric_canvas_height), fontsize=8,
                          columnspacing=1.7, handletextpad=.5, handlelength=1.1)
        fig.text(4 / width_mm, .53, "Change in CFN (percentage points)",
                 fontsize=8, rotation=90, ha="center", va="center")
        metric_fig.text(3 / metric_width_mm, 38 / metric_canvas_height,
                        "Change in CFN (percentage points)", fontsize=8,
                        rotation=90, ha="center", va="center")

        scatter_artists, overlap_labels, quadrant_artists = [], [], []
        metric_artists, metric_groups = [], []
        scatter_axes, metric_axes = {}, {}
        for index, method in enumerate(methods):
            upper = add_axes_mm(metric_fig, 20 + 54 * index, 9, axes_width, metric_height)
            metric_axes[method] = upper
            upper.spines[["top", "right"]].set_visible(False)
            upper.set_xlim(-.55, 2.55)
            upper.set_ylim(*ylim)
            upper.tick_params(axis="x", length=0, pad=4, labelsize=7)
            upper.tick_params(axis="y", length=2.5, width=.6, pad=2, labelleft=index == 0)
            upper.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            upper.axhline(0, color="#AEB4B9", linewidth=.65, zorder=0)
            upper.set_title(method + "  (n=27)", fontsize=8.5, fontweight="bold", pad=5)
            upper.text(-.15, 1.045, "abc"[index], transform=upper.transAxes,
                       fontsize=9, fontweight="bold", ha="left", va="bottom")
            positioned, category_labels = {}, []
            for category, direction in enumerate(metric_directions):
                selected = [row for row in rows[method] if row["metric_direction"] == direction]
                wanted = sum(metric_cache["direction_counts"][method][direction].values())
                if len(selected) != wanted:
                    raise ValueError("Metric category totals differ from the audited cache")
                offsets = category_offsets(selected)
                for row in selected:
                    positioned[row["graph"]] = (category + offsets[row["graph"]], category)
                category_labels.append(direction.capitalize() + "\n(n=" + str(len(selected)) + ")")
                metric_groups.append({"method": method, "direction": direction,
                                      "category": category, "count": len(selected), "axis": upper})
            upper.set_xticks([0, 1, 2], category_labels)
            for outcome, color, marker in outcomes:
                selected = [row for row in rows[method]
                            if ("tied" if row["outcome"] == "unchanged" else row["outcome"]) == outcome]
                artist = upper.scatter([positioned[row["graph"]][0] for row in selected],
                                       [row["delta_cfn_pp"] for row in selected],
                                       s=18, marker=marker, color=color, alpha=.90,
                                       edgecolors="white", linewidths=.35, zorder=3)
                metric_artists.append({"method": method, "outcome": outcome, "artist": artist,
                                       "graphs": [row["graph"] for row in selected],
                                       "categories": [positioned[row["graph"]][1] for row in selected]})

            ax = add_axes_mm(fig, 19, 100 - 40 * index, 64, count_height)
            scatter_axes[method] = ax
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=2.5, width=.6, pad=2, labelleft=True)
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(*ylim)
            left, right = ax.get_xlim()
            bottom, top = ax.get_ylim()
            for direction, x0, x1, text_x, align, tree_label in (
                    ("decreased", left, 0, .04, "left", "Fewer trees"),
                    ("increased", 0, right, .985, "right", "More trees")):
                for outcome, y0, y1, text_y, valign, background, text_color in (
                        ("worse", 0, top, .96, "top", "#FCF4EA", "#8B653E"),
                        ("improved", bottom, 0, 0, "bottom", "#F0F6FA", "#356780")):
                    region = None
                    if ((direction == "decreased" and outcome == "worse")
                            or (direction == "increased" and outcome == "improved")):
                        region = Rectangle((x0, y0), x1 - x0, y1 - y0,
                                           facecolor=background, edgecolor="none", zorder=-2)
                        ax.add_patch(region)
                    count = payload["direction_counts"][method][outcome][direction]
                    # One line keeps the lower counts clear of points in the compact panels.
                    separator = "; " if outcome == "improved" else "\n"
                    label_x = .015 if outcome == "improved" and direction == "decreased" else text_x
                    label = ax.text(label_x, text_y,
                                    tree_label + separator + outcome_labels[outcome] + ": " + str(count),
                                    transform=ax.transAxes, ha=align, va=valign,
                                    fontsize=7.5, linespacing=1.0, color=text_color, zorder=1)
                    quadrant_artists.append({"method": method, "row": outcome,
                                             "column": direction, "patch": region, "artist": label})
            ax.axhline(0, color="#AEB4B9", linewidth=.65, zorder=0)
            ax.axvline(0, color="#AEB4B9", linewidth=.65, zorder=0)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, min_n_ticks=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            if index == 2:
                ax.set_xlabel("Change in number of trees\nexceeding the threshold, " + r"$\Delta K$",
                              labelpad=4, linespacing=1.1, fontsize=7.3)
            ax.set_title(method + "  (n=27)", fontsize=8.5, fontweight="bold", pad=4)
            ax.text(-.15, 1.045, "abc"[index], transform=ax.transAxes,
                    fontsize=9, fontweight="bold", ha="left", va="bottom")
            points = payload["points"][method]
            for outcome, color, marker in outcomes:
                selected = [point for point in points if outcome_of(point) == outcome]
                artist = ax.scatter([point["delta_trees"] for point in selected],
                                    [point["delta_cfn_pp"] for point in selected],
                                    s=22, marker=marker, color=color, alpha=.90,
                                    edgecolors="white", linewidths=.35, zorder=3)
                scatter_artists.append({"method": method, "outcome": outcome, "artist": artist,
                                        "graphs": [point["graph"] for point in selected]})
            multiplicities = Counter((int(point["delta_trees"]), float(point["delta_cfn_pp"]))
                                     for point in points)
            for coordinates, count in sorted(multiplicities.items()):
                if count <= 1:
                    continue
                offset_x = -9 if coordinates[0] >= x_max else 9
                offset_y = -12 if coordinates[1] > 0 else 12
                annotation = ax.annotate(
                    "n=" + str(count), xy=coordinates,
                    xytext=(offset_x, offset_y), textcoords="offset points",
                    fontsize=7.5, color="#50575D",
                    ha="right" if offset_x < 0 else "left",
                    va="top" if offset_y < 0 else "bottom",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": .4},
                    arrowprops={"arrowstyle": "-", "color": "#767D83",
                                "linewidth": .55, "shrinkA": 2, "shrinkB": 2}, zorder=5)
                overlap_labels.append({"method": method, "coordinates": coordinates,
                                       "count": count, "artist": annotation})

        fig._guidance_layout = "forest-count-vertical"
        fig._guidance_scatter_artists = scatter_artists
        fig._guidance_overlap_labels = overlap_labels
        fig._guidance_quadrants = quadrant_artists
        fig._guidance_axes = {"scatter": scatter_axes}
        metric_fig._guidance_layout = "metric-direction-horizontal"
        metric_fig._guidance_metric_artists = metric_artists
        metric_fig._guidance_metric_groups = metric_groups
        metric_fig._guidance_metric_spacing_points = minimum_spacing_pt
        metric_fig._guidance_axes = {"metric": metric_axes}
        fig._exact_export_size = metric_fig._exact_export_size = True
        _publication_font_floor(fig, 7.5, 83.74)
        _publication_font_floor(metric_fig, 8, 164.6)
        return fig, metric_fig


if __name__ == "__main__":

    remove_nodes_dic={"test1":np.arange(0,12,1),

                      #"test3":[1,2,4,6,7,8,13],
                      }
    g = igraph.Graph.Read_GraphML("../dataset/dataset_synth/barabasi_n_25_m_2_0.graphml")
    net_g = g.to_networkx()
   # print(remove_nodes_dic["test2"])



   # graph=nx.read_graphml("../dataset/barabasi_0_0.graphml",node_type=int)#暂不可用  graph格式节点标号不对
    #nx.draw(graph,with_labels=True)
   # plt.show()
    fig = plt.figure(num="robustness_curves 测试")
   # ax = fig.add_subplot(122)
    ax=fig.gca()
    draw_robustness_curves(ax=ax,remove_nodes_dic=remove_nodes_dic,network=net_g,title=False,data_name="test graph")
    plt.show()



