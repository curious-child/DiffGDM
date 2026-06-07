import os
import igraph
import torch_geometric as tg
import numpy as np
import networkx as nx
from torch_geometric.data import Data, Batch
import torch
import numpy as np
from igraph import *
import matplotlib.pyplot as plt
from glob import glob
from sklearn.manifold import TSNE
import yaml

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def gen_graph(g_type, num_min, num_max):
    max_n = num_max
    min_n = num_min
    cur_n = np.random.randint(max_n - min_n + 1) + min_n
    cur_n=40
    if g_type == 'erdos_renyi':
        g = Graph.Erdos_Renyi(n=cur_n,p=0.15,directed=False)
        g.vs['node_feature'] = np.ones((cur_n,2),dtype=float)
    elif g_type == 'small-world':
        g = Graph.Watts_Strogatz(dim=1,size=cur_n,nei=8,p=0.1)
        g.vs['node_feature'] = np.ones((cur_n, 2), dtype=float)
    elif g_type == 'barabasi_albert':
        g = Graph.Barabasi(n=cur_n, m=4,directed=False)
        g.vs['node_feature'] = np.ones((cur_n, 2), dtype=float)
    return g

def to_gnngraph(g,graph_name,features):
    g=nx.convert_node_labels_to_integers(g)
    g = g.to_directed() if not nx.is_directed(g) else g
    if isinstance(g, (nx.MultiGraph, nx.MultiDiGraph)):
        edges = list(g.edges(keys=False))
    else:
        edges = list(g.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().view(2, -1)

    if "None" in features:
        x = np.ones((g.number_of_nodes(), 1))

    elif "one-hot" in features:
        x=np.identity(g.number_of_nodes())
    else:
        x = np.column_stack(
            tuple(
                list(nx.get_node_attributes(g,feature).values()) for feature in features
            )
        )

    x = torch.from_numpy(x).to(torch.float)

    return Data(x=x,edge_index=edge_index,graph_name=graph_name)

def load_model(path: str,model,device) :
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :return: The loaded model,loaded network parameters.
    """

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)

    loaded_net_param = state["net_param"]

    loaded_state_dict = state['state_dict']

    if torch.equal(loaded_state_dict["encoder_conv_layers.0.lin_src.weight"],loaded_state_dict["encoder_conv_layers.0.lin_dst.weight"]):
        loaded_state_dict["encoder_conv_layers.0.lin.weight"]=loaded_state_dict["encoder_conv_layers.0.lin_src.weight"]
    if torch.equal(loaded_state_dict["encoder_conv_layers.1.lin_src.weight"],loaded_state_dict["encoder_conv_layers.1.lin_dst.weight"]):
        loaded_state_dict["encoder_conv_layers.1.lin.weight"]=loaded_state_dict["encoder_conv_layers.1.lin_src.weight"]
    loaded_net_param["device"] = device
    model = model(gnn_name=loaded_net_param["gnn_name"], net_param=loaded_net_param).to(
        loaded_net_param["device"])
    model.load_state_dict(loaded_state_dict,strict=False)


    return model,loaded_net_param
def load_graph_model(path: str, device, infer_para=None) :
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :return: The loaded model,loaded network parameters.
    """

    # Load model and args
    from ..models import graph_models
    state = torch.load(path, map_location=lambda storage, loc: storage)

    loaded_net_param = state["net_param"]
    loaded_state_dict = state['state_dict']
    if infer_para is not None:
        loaded_net_param.update(infer_para)


    model = graph_models(gnn_name=loaded_net_param["gnn_name"], net_param=loaded_net_param).to(
        device)
    model.load_state_dict(loaded_state_dict,strict=False)
    model = model.to(device)

    return model,loaded_net_param
def del_tensor_ele(arr, index):
    """
    arr: 输入tensor
    index: 需要删除位置的索引
    n: 从index开始，需要删除的行数
    """
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)

def extract_list_del(l1,end,start=0,):
    l2=l1[start:end]
    del l1[start:end]
    return l2,l1

def igraph_to_gnngraph(g,features,targets=None):

    if "None" in features:
        x = np.ones((g.vcount(), 1))
    elif "one-hot" in features:
        x=np.identity(g.vcount())
    else:
        x = np.column_stack(
            tuple(
                g.vs[feature] for feature in features
            )
        )
    x = torch.from_numpy(x).to(torch.float)
    source_nodes1 = [edge.source for edge in g.es]
    target_nodes1 = [edge.target for edge in g.es]
    source_nodes = source_nodes1 + target_nodes1
    target_nodes = target_nodes1 + source_nodes1

    if targets !=None:
        y = g.vs[targets]
        y = torch.tensor(y).to(torch.float)
        return Data(x=x, y=y, edge_index=torch.tensor([source_nodes, target_nodes], dtype=torch.long))
    else:
        return Data(x=x, edge_index=torch.tensor([source_nodes, target_nodes], dtype=torch.long))




def pre_DataSet(file_path, features,targets,filter="*"):

    dataSet=dict()
    for file in sorted(glob(file_path+'/'+filter+".graphml")):

        g=igraph.Graph.Read_GraphML(file)
        graph_name=file[len(file_path) + 1:].replace(".graphml", "")
        g=igraph_to_gnngraph(g,features=features,targets=targets)
        dataSet[graph_name]=g


    return dataSet