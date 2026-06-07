import igraph
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from .draw_utils import list_of_groups
line_color_list=["b","g","r","c","m","y","k"]
line_style_list=["-","--","-.",":"]
marker_style_list=[".",",","o","v",'^','>','<','1','2','3','4','s','p','*','h','H','+','x','D','d']
def draw_robustness_curves(ax,remove_nodes_dic,network,title=False,data_name=None):
    color_number=0
    linestyle_number=0
    for method_name,remove_nodes in remove_nodes_dic.items():
        graph_node_number=network.number_of_nodes()
        graph=network.copy()
        remove_nodes_per=list()
        lcc_size_list=list()
        number_remove=0.0
        lcc_size = len(max(nx.connected_components(graph), key=len))
        remove_nodes_per.append(number_remove/graph_node_number)
        lcc_size_list.append(lcc_size / graph_node_number)
        #print("method_name:{}".format(method_name))
        for remove_node in remove_nodes:
           # print("remove_node:{}".format(remove_node))
            graph.remove_node(remove_node)
            lcc_size = len(max(nx.connected_components(graph), key=len))
            number_remove+=1.
            remove_nodes_per.append(number_remove/graph_node_number)
            lcc_size_list.append(lcc_size/graph_node_number)
        #print("{}:{}".format(method_name,lcc_size))
        ax.plot(remove_nodes_per, lcc_size_list,
                label=method_name,
                color=line_color_list[color_number],
                linestyle=line_style_list[linestyle_number]
                )
        color_number+=1
        if color_number>=len(line_color_list):
            linestyle_number+=1
            color_number=0
    ax.legend(loc="best",framealpha=0.2,frameon=False)
    ax.set_xlabel("percentage of removed nodes")
    ax.set_ylabel("LCC size")
    if title==True :
      ax.set_title("{}:{} nodes,{} edges".format(data_name,network.number_of_nodes(),network.number_of_edges()))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(linestyle="--")

def draw_remove_statistics(fig,graphs_remove_size):
    # colormap = ["b", "g", "r", "c", "m", "y", "k"]
    colormap = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#E7DAD2"]
    colors_num = len(colormap)

    graphs_name = list(graphs_remove_size.keys())
    methods_list = list(list(graphs_remove_size.values())[0].keys())
    children_list_len=32
    methods_split=list_of_groups(methods_list,children_list_len=children_list_len)
    subplot_numRows=len(methods_split)
    subplot_iter = 1
    for methods in methods_split:

        ax = fig.add_subplot(subplot_numRows, 1, subplot_iter)
        methods_all_dismantle_size = np.zeros(len(methods))
        bottom_setup = np.zeros(len(methods))
        bars_gap = 0.008 * np.ones(len(methods))

        draw_x = list(range(len(methods)))
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
            methods_dismantle_size=[graphs_remove_size[graph_name].get(method) for method in methods]
            ax.bar(x=draw_x,
                   height=methods_dismantle_size,
                   align="center", bottom=bottom_setup, color=colormap[i % colors_num], alpha=1,
                   tick_label=methods, label=graph_name
                   )
            bottom_setup += np.array(methods_dismantle_size) + bars_gap
            methods_all_dismantle_size += np.array(methods_dismantle_size)
            # 柱状图标注文本数据
        for x, y, bottom in zip(draw_x, methods_all_dismantle_size, bottom_setup):
            plt.text(x, bottom, '%.3f' % y, ha='center', va='bottom', fontsize=5)
            # 设置 图像轴是否显示
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # 设置网格线
        ax.set_axisbelow(True)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_yticklines(), visible=False)
        plt.setp(ax.get_xticklines(), visible=False)
        plt.grid(linestyle="-", axis="y", alpha=0.5)
        # plt.axis("off")#
        if subplot_iter==1:
        # 设置标签在图像外侧
            plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0, frameon=False)

        subplot_iter += 1




if __name__ == "__main__":

    remove_nodes_dic={"test1":np.arange(0,12,1),
                      "test2":np.arange(24,0,-1),
                      #"test3":[1,2,4,6,7,8,13],
                      }
    g = igraph.Graph.Read_GraphML("../dataset/barabasi_0_0.graphml")
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



