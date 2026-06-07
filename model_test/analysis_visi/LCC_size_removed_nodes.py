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



def draw_dismantle_statistics_real(ax,graphs_dismantle_size):
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
        ax.bar(x=draw_x,
               height=height,
               align="center", bottom=bottom_setup, color=colormap[i % colors_num], alpha=1,
               tick_label=new_methods, label=graph_name
               )
        bottom_setup += np.array(height) + bars_gap

        methods_all_dismantle_size += 100*np.array(list(graphs_dismantle_size[graph_name].values()))
    # 柱状图标注文本数据
    for x, y,bottom in zip(draw_x, methods_all_dismantle_size,bottom_setup):
        ax.text(x, bottom, '%.3f' % y, ha='center', va='bottom', #fontsize=10
                 )
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


    plt.tight_layout()
def draw_dismantle_methods_graphs_synth(ax, graphs_dismantle_size):
    pass
def draw_dismantle_methods_graphs_real(ax,graphs_dismantle_size,method_list=None,yticklabels=True,cbar=True):
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



