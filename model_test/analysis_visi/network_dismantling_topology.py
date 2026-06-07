from glob import glob
import igraph
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from utils.utils import seed_torch


def draw_topology(fig,network,key_graph_nodes=[]):

    target_values=list(nx.get_node_attributes(network,"t_0.18-diversity").values())
    key_nodes=[k for k,v in nx.get_node_attributes(network,"t_0.18-diversity").items() if v==1.0]
    key_edges=network.edges(key_nodes)
  #  key_graph=network.subgraph(key_graph_nodes)
    key_graph= network.edge_subgraph(key_edges)
    subgraph_values=list(nx.get_node_attributes(key_graph, "t_0.18-diversity").values())

    ax = fig.add_subplot(121)
    ax.set_title("the original network and key nodes ")
    pos = nx.kamada_kawai_layout(network)
    nx.draw_networkx(network, pos=pos,node_color=target_values,with_labels=True,cmap="OrRd",vmin=0.1,vmax=0.4)

    ax = fig.add_subplot(122)
    ax.set_title("the subnetwork induced on key nodes ")
    nx.draw_networkx(key_graph, pos=pos,node_color=subgraph_values,with_labels=True,cmap="OrRd",vmin=0.1,vmax=0.4)
def draw_3d(G,ax,pos,color_value):

    x, y, z = zip(*pos.values())

    for n, m in G.edges():
        zline = (z[n], z[m])
        xline = (x[n], x[m])
        yline = (y[n], y[m])
        ax.plot3D(xline, yline, zline,'k',linewidth=1)
    return ax.scatter3D(x, y, z, c=color_value, marker='o',alpha=1, s=100)
def draw_dimantle_nodes(ax,network,remove_nodes,title=False,data_name=None,method_name=None,visual_dim=2):
    nodes_remove_record={}
    remove_nodes=list(map(int, remove_nodes))
    # 不考虑拆解顺序的
    for node in network.nodes():
        if node in set(remove_nodes):
            nodes_remove_record[node]=1
        else:
            nodes_remove_record[node]=0
    seed_torch(123)
  #  print(remove_nodes)
    #考虑拆解顺序的
    # for node in network.nodes():#不考虑拆解顺序的
    #     if node in set(remove_nodes):
    #         nodes_remove_record[node]=len(remove_nodes)-int(remove_nodes.index(node))
    #     else:
    #         nodes_remove_record[node]=0

    if visual_dim==2:

        #pos = nx.spring_layout(network,)
        pos = nx.kamada_kawai_layout(network, )
        # core_num = nx.core_number(network)  # 每个节点的 core number
        # # 按 core_number 分组 → 喂给 shell_layout 的 nlist
        # levels = sorted(set(core_num.values()), reverse=True)
        # nlist = [[n for n in network if core_num[n] == lv] for lv in levels]
        # pos = nx.shell_layout(network, nlist=nlist)
       # pos=centrality_radial_layout(network, cent=nx.degree_centrality(network), n_rings=3, center_fraction=0.1)

        if title == True:
            ax.set_title("{} \ndismantle method:{}".format(data_name,method_name))
        nx.draw_networkx_nodes(network,node_size=50, pos=pos,linewidths=0.1,ax=ax, node_color=list(nodes_remove_record.values()),edgecolors='k',cmap="OrRd")
        nx.draw_networkx_edges(network, pos=pos,ax=ax, width=1,alpha=0.4,edge_color='grey')
    elif visual_dim==3:
        pos = nx.kamada_kawai_layout(network, dim=3)
        if title == True:
            ax.set_title("{} \ndismantle method:{}".format(data_name,method_name))
        draw_3d(G=network,pos=pos,ax=ax,color_value=list(nodes_remove_record.values()))

    ax.axis('off')
def centrality_radial_layout(G, cent=None, n_rings=3, center_fraction=0.3):
    """
    cent: 已有的中心性字典；为 None 时用度中心性
    n_rings: 把非核心节点分几圈
    center_fraction: 最核心节点允许占据的半径比例
    """
    if cent is None:
        cent = nx.degree_centrality(G)

    nodes_sorted = sorted(G.nodes(), key=lambda n: cent[n], reverse=True)

    # 核心节点：最重要的那个/几个直接放 (0,0) 附近
    n_core = max(1, int(len(G) * center_fraction))
    core_nodes = nodes_sorted[:n_core]
    rest_nodes = nodes_sorted[n_core:]

    pos = {}

    # 核心节点：在小圆半径内散开，避免完全重合
    r_core = 0.15
    for i, n in enumerate(core_nodes):
        angle = 2 * np.pi * i / max(len(core_nodes), 1)
        pos[n] = (r_core * np.cos(angle), r_core * np.sin(angle))

    # 其余节点按重要性分到 rings
    ring_groups = np.array_split(rest_nodes, min(n_rings, len(rest_nodes)))
    for ring_idx, group in enumerate(ring_groups, start=1):
        r = 0.4 + 0.6 * (ring_idx / n_rings)   # 半径递增
        for i, n in enumerate(group):
            angle = 2 * np.pi * i / len(group)
            pos[n] = (r * np.cos(angle), r * np.sin(angle))

    return pos


def gerenate_key_subgraph(diversity_target_networks):
    initial_network=diversity_target_networks[0]
    keygraph_nodes_set=set(initial_network)
    for network in  diversity_target_networks:
        key_nodes = [k for k, v in nx.get_node_attributes(network, "t_0.18-diversity").items() if v == 1.0]
        key_edges = network.edges(key_nodes)
        subgraph = network.edge_subgraph(key_edges)
        keygraph_nodes_set=keygraph_nodes_set&set(subgraph.nodes) #并集
   # keygraph=initial_network.subgraph(keygraph_nodes_set)

    return keygraph_nodes_set







def draw_networks_topogy(file_path="../dataset/", filter="gnp_*",save_path="../out/subgraph_keynodes2/"):
    diversity_target_networks=[]
    diversity_target_networks_name=None
    keynodes_list=[]

    # for file in sorted(glob(file_path + filter + ".graphml")):
    #     g = igraph.Graph.Read_GraphML(file)
    #     G = g.to_networkx()
    #     if file[len(file_path):-9] !=diversity_target_networks_name:
    #        # print(file[len(file_path):-9])
    #         if diversity_target_networks_name!=None:
    #             keygraph_nodes=gerenate_key_subgraph(diversity_target_networks)
    #             keynodes_list.append(keygraph_nodes)
    #         diversity_target_networks_name=file[len(file_path):-9]
    #         diversity_target_networks=[]
    #     diversity_target_networks.append(G)

   # diversity_target_networks_name=None
  #  keygraph_nodes_iter=iter(keynodes_list)
  #  keygraph_nodes=next(keygraph_nodes_iter)
    for file in sorted(glob(file_path + filter + ".graphml")):
        g = igraph.Graph.Read_GraphML(file)
        G = g.to_networkx()
        fig = plt.figure(figsize=(15, 10))
        # if file[len(file_path):-9] !=diversity_target_networks_name:
        #     if diversity_target_networks_name!=None:
        #         keygraph_nodes=next(keygraph_nodes_iter)
        #     diversity_target_networks_name = file[len(file_path):-9]

     #   draw_topology(fig, network=G, key_graph_nodes=keygraph_nodes)
        draw_topology(fig, network=G,)

        plt.savefig(save_path + file[len(file_path):-8].replace(".", "_"))
        plt.close()

     #   plt.show()

def draw_topology_graph(file_path="../dataset/train_dataset_diversity/",filter="gnp_random_graph_n25_p0.1_4_*"):
    diversity_target_networks = []
    keynodes_list = []

    for file in sorted(glob(file_path + filter + ".graphml")):
        g = igraph.Graph.Read_GraphML(file)
        G = g.to_networkx()
        print(file[len(file_path):-8])
        diversity_target_networks.append(G)
    keygraph_nodes = gerenate_key_subgraph(diversity_target_networks)

    fig = plt.figure()
    subplot_numCols = 2
    subplot_nums = len(diversity_target_networks)
    print(subplot_nums)
    subplot_numRows = int(np.ceil(subplot_nums / subplot_numCols))

    subplot_iter = 1
    for file in sorted(glob(file_path + filter + ".graphml")):
        g = igraph.Graph.Read_GraphML(file)
        G = g.to_networkx()
        key_graph = G.subgraph(keygraph_nodes)
        subgraph_values = list(nx.get_node_attributes(key_graph, "t_0.18-diversity").values())
        pos = nx.kamada_kawai_layout(G)
        ax = fig.add_subplot(subplot_numRows, subplot_numCols, subplot_iter)
        nx.draw_networkx(key_graph, node_size=300, pos=pos, node_color=subgraph_values, edgecolors="k",
                         with_labels=True, cmap="OrRd", vmin=0.1,
                         vmax=0.4)

        subplot_iter += 1
    plt.show()
if __name__ == "__main__":

    file_path="../dataset/train_dataset_diversity/"
    filter="barabasi_14_*"
    draw_topology_graph(file_path=file_path,filter=filter)
    # file="../dataset/train_dataset_diversity/gnp_random_graph_n25_p0.1_4_1.graphml"
    # g = igraph.Graph.Read_GraphML(file)
    # G = g.to_networkx()
    # node_color=np.zeros(25)
    # pos = nx.kamada_kawai_layout(G)
    # nx.draw_networkx(G, pos=pos, node_color=node_color, with_labels=True, cmap="OrRd", vmin=0.1, vmax=0.4)
    #
    # plt.show()
