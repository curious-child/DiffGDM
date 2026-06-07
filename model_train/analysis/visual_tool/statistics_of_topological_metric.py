from glob import glob

import igraph
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
line_color_list=["b","g","r","c","m","y","k","w"]
line_style_list=["-","--","-.",":"]
marker_style_list=[".",",","o","v",'^','>','<','1','2','3','4','s','p','*','h','H','+','x','D','d']
def draw_statistics_topological_metric(fig,subplot_numCols,network_list,topogical_metrics_list):
    statistics_metrics={}

    subplot_nums=len(topogical_metrics_list)
    subplot_numRows = int(np.ceil(subplot_nums / subplot_numCols))

    subplot_iter=1
    for topogical_metric in topogical_metrics_list:
        statistics_metrics[topogical_metric]=list()

        for graph in network_list:

            if topogical_metric == "edges density":
                edge_density=nx.density(graph)
                statistics_metrics[topogical_metric].append(edge_density)

            if topogical_metric == "global efficiency":
                global_efficiency=nx.global_efficiency(graph)
                statistics_metrics[topogical_metric].append(global_efficiency)

            if topogical_metric == "modularity":
                graph_communities=nx.community.greedy_modularity_communities(graph)
                modularity=nx.community.modularity(graph,communities=graph_communities)
                statistics_metrics[topogical_metric].append(modularity)

            if topogical_metric == "assortativity coefficient":
                global_efficiency=nx.assortativity.degree_pearson_correlation_coefficient(graph)
                statistics_metrics[topogical_metric].append(global_efficiency)

            if topogical_metric == "average clustering coefficient":
                clustering_coefficient=nx.average_clustering(graph)
                statistics_metrics[topogical_metric].append(clustering_coefficient)

            if topogical_metric == "transitivity":#在社会学上常用来描述网络的聚类性质 指标 橫截性
                transitivity=nx.transitivity(graph)
                statistics_metrics[topogical_metric].append(transitivity)

            if topogical_metric == "diameter":
                diameter=nx.diameter(graph)
                statistics_metrics[topogical_metric].append(diameter)

            if topogical_metric == "average shortest path length":
                average_shortest_path_length = nx.average_shortest_path_length(graph)
                statistics_metrics[topogical_metric].append(average_shortest_path_length)

            if topogical_metric == "average node connectivity":
                '''
                Local node connectivity for two non adjacent nodes s and t is the minimum number of nodes that must be removed
                (along with their incident edges) to disconnect them.
                The average node connectivity of a graph G is the average of local node connectivity over all pairs of nodes of G
                '''
                average_node_connectivity = nx.average_node_connectivity(graph)
                statistics_metrics[topogical_metric].append(average_node_connectivity)

            if topogical_metric == "small-world coefficient":
                '''
                The small-world coefficient of a graph G is:
                omega = Lr/L - C/Cl
                The small-world coefficient (omega) ranges between -1 and 1. 
                Values close to 0 means the G features small-world characteristics. 
                Values close to -1 means G has a lattice shape whereas values close to 1 means G is a random graph
                '''
                small_world_coefficient = nx.omega(graph)
                statistics_metrics[topogical_metric].append(small_world_coefficient)

        ax=fig.add_subplot(subplot_numRows,subplot_numCols,subplot_iter)
        ax.hist(statistics_metrics[topogical_metric],)
        ax.set_title(topogical_metric)
        subplot_iter+=1

def draw_topological_metric_curves(ax,remove_nodes_dic,network,topogical_metric,title=False,data_name=None):

    color_number = 0
    linestyle_number = 0

    #选择拓扑特征度量函数
    if topogical_metric == "edges density":
        metric_function = nx.density
    if topogical_metric == "global efficiency":
        metric_function = nx.global_efficiency
    if topogical_metric == "modularity":
        metric_function = nx.community.modularity
    if topogical_metric == "assortativity coefficient":
        metric_function = nx.assortativity.degree_pearson_correlation_coefficient
    if topogical_metric == "average clustering coefficient":
        metric_function = nx.average_clustering
    if topogical_metric == "transitivity":  # 在社会学上常用来描述网络的聚类性质 指标 橫截性
        metric_function = nx.transitivity
    if topogical_metric == "diameter":
        metric_function = nx.diameter
    if topogical_metric == "average shortest path length":
        metric_function = nx.average_shortest_path_length
    if topogical_metric == "average node connectivity":
        metric_function = nx.average_node_connectivity
    if topogical_metric == "small-world coefficient":
        metric_function = nx.omega

    # 最大连通子图的拓扑度量值-删除节点 曲线绘制
    for method_name,remove_nodes in remove_nodes_dic.items():
        graph_node_number=network.number_of_nodes()
        graph=network.copy()
        remove_nodes_per=list()
        topogical_metric_value = list()
        number_remove=0.0
        LCC_node_set= max(nx.connected_components(graph), key=len)
        LCC=graph.subgraph(LCC_node_set)
        if topogical_metric =="modularity":
            metric_value=metric_function(LCC,communities=nx.community.greedy_modularity_communities(LCC))
        else:
            metric_value = metric_function(LCC)
        remove_nodes_per.append(number_remove/graph_node_number)
        topogical_metric_value.append(metric_value )
        #print("method_name:{}".format(method_name))
        for remove_node in remove_nodes:
           # print("remove_node:{}".format(remove_node))
            graph.remove_node(remove_node)
            LCC_node_set = max(nx.connected_components(graph), key=len)
            LCC = graph.subgraph(LCC_node_set)
            if metric_function == "modularity":
                metric_value = metric_function(LCC, communities=nx.community.greedy_modularity_communities(LCC))
            else:
                metric_value = metric_function(LCC)
            number_remove += 1.
            remove_nodes_per.append(number_remove / graph_node_number)
            topogical_metric_value.append(metric_value)

        ax.plot(remove_nodes_per, topogical_metric_value,
                label=method_name,
                color=line_color_list[color_number],
                linestyle=line_style_list[linestyle_number]
                )
        color_number+=1
        if color_number>=8:
            linestyle_number+=1
            color_number=0
    ax.legend(loc="best",framealpha=0.2,frameon=False)
    ax.set_xlabel("percentage of removed nodes")
    ax.set_ylabel("{}".format(topogical_metric))
    if title==True :
      ax.set_title("{}:{} nodes,{} edges".format(data_name,network.number_of_nodes(),network.number_of_edges()))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(linestyle="--")

def pre_DataSet(dataSet, file_path,filter="*"):

    dataSet.clear()
    for file in sorted(glob(file_path+'/'+filter+".graphml")):
        g=igraph.Graph.Read_GraphML(file).to_networkx()
        dataSet.append(g)

if __name__ == "__main__":




    topogical_metrics_list=[
                        "edges density",
                        "global efficiency",
                       # "modularity",
                        "assortativity coefficient",
                        "average clustering coefficient",
                        #"transitivity","diameter",
                        "average shortest path length",
                         "average node connectivity",
                         #"small-world coefficient"
                            ]

    graphset = []
    pre_DataSet(dataSet=graphset, file_path="../dataset/train_dataset", filter="barabasi_*")
    fig = plt.figure(num="statistics_topological_metric 测试")
    draw_statistics_topological_metric(fig,subplot_numCols=3,network_list=graphset,topogical_metrics_list=topogical_metrics_list)

    # remove_nodes_dic = {"test1": np.arange(0, 18, 1),
    #                     "test2": np.arange(18, 0, -1),
    #                     # "test3":[1,2,4,6,7,8,13],
    #                     }
    # g = igraph.Graph.Read_GraphML("../dataset/barabasi_0_0.graphml")
    # net_g = g.to_networkx()
    # fig = plt.figure(num="topogical_metric_curves 测试")
    # ax = fig.add_subplot(121)
    # draw_topological_metric_curves(ax=ax,remove_nodes_dic=remove_nodes_dic,network=net_g,topogical_metric=topogical_metrics_list[1])
    # ax = fig.add_subplot(122)
    # draw_topological_metric_curves(ax=ax, remove_nodes_dic=remove_nodes_dic, network=net_g,
    #                                topogical_metric=topogical_metrics_list[0])

    plt.show()








