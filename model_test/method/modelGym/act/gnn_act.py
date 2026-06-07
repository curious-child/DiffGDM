import torch_geometric.nn as gnn
import torch
def gnn_act(act_name):
    if act_name=="ELU":
        return torch.nn.ELU()
    elif act_name=="ReLU":
        return torch.nn.ReLU()
    elif act_name=="Tanh":
        return torch.nn.Tanh()
    elif act_name=="Leakyrelu":
        return torch.nn.LeakyReLU()
    else:
        raise ValueError("the definition  don't exit\n"
                         "\tyou can define it before using it")