from .metrics_model.vanilla_models import ND_GNN_vanilla, robustenss_metircs_model, diffusion_CEP_model


def graph_models(gnn_name,net_param):
    if gnn_name=="ND_GNN_vanilla":
        return ND_GNN_vanilla(net_param=net_param)
    elif gnn_name=="robustenss_metircs_model":
        return robustenss_metircs_model(net_param=net_param)
    elif gnn_name=="diffusion_CEP_model":
        return diffusion_CEP_model(net_param=net_param)
    else:
        raise ValueError("the definition  don't exit\n"
                         "\tyou can define it before using it")

def diffusion_models(task_model,net_param):
    from .Diffusion_model.diffusion_model import FVSModel
    from .Diffusion_model.diffusion_gnn_encoder import T2TGNNEncoder, NDGNNEncoder

    if net_param["GNNEncoder_set"] == "T2TGNNEncoder":
        GNNEnconder = T2TGNNEncoder
    elif net_param["GNNEncoder_set"] == "NDGNNEncoder":
        GNNEnconder = NDGNNEncoder
    else:
        raise ValueError("the definition  don't exit\n"
                         "\tyou can define it before using it")

    if task_model=="FVSModel":
        return FVSModel(net_param=net_param,GNNEncoder=GNNEnconder)
    else:
        raise ValueError("the definition  don't exit\n"
                         "\tyou can define it before using it")


