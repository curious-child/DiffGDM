from glob import glob
import igraph
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

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

def gerenate_key_subgraph(diversity_target_networks,):
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



if __name__ == "__main__":

     draw_networks_topogy()

