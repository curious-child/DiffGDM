import json
import os
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from tqdm import tqdm
import networkx as nx
from glob import glob
import yaml
from utils.utils import record_scores_analysis, to_gnngraph_nx, load_graph_model
from torch_geometric.loader import  DataLoader
from utils.utils import to_gnngraph_nx, save_checkpoint
def evaluate_dataset_properties(dataset_path, graph_property):
    print("evaluate_dataset_properties",graph_property)
    dataSet_ptoperties = []
    for file in tqdm(sorted(glob(dataset_path + '/' + "*" + ".graphml"))):
        try:
            nx_g = nx.read_graphml(file)
            nx_g = nx.convert_node_labels_to_integers(nx_g)
        except Exception:

            raise Exception("There are errors in {}".format(file))
        # g=igraph.Graph.Read_GraphML(file)
        # if g.vcount()<25:
        #     print("error node number:{}".format(g.vcount()))
        #     continue
        # if "random" in params["features"]:
        #    # for _ in range(10):
        #     g=to_gnngraph_nx(nx_g,features=params["features"],targets=targets,SL=params["SL"])
        #     dataSet.append(g)
        # else:
        g = to_gnngraph_nx(nx_g, features="random", targets=graph_property, SL=True)
        dataSet_ptoperties.append(g.y)
    dataSet_ptoperties=np.array(dataSet_ptoperties)
    population_var =np.var(dataSet_ptoperties)
    mean = np.mean(dataSet_ptoperties)
    sst = np.sum((dataSet_ptoperties - mean) ** 2)
    print("population_var:{}".format(population_var))
    print("dataset size:{}".format(len(dataSet_ptoperties)))
    print("sst:{}".format(sst))
def evaluate_models_predictions(models_path, dataset_path, graph_property,ckpt_model=False):
    dataSet = []
    dataSet_ptoperties = []
    record_method_eval = dict()
    for file in tqdm(sorted(glob(dataset_path + '/' + "*" + ".graphml"))):
        try:
            nx_g = nx.read_graphml(file)
            nx_g = nx.convert_node_labels_to_integers(nx_g)
        except Exception:

            raise Exception("There are errors in {}".format(file))
        # g=igraph.Graph.Read_GraphML(file)
        # if g.vcount()<25:
        #     print("error node number:{}".format(g.vcount()))
        #     continue
        # if "random" in params["features"]:
        #    # for _ in range(10):
        #     g=to_gnngraph_nx(nx_g,features=params["features"],targets=targets,SL=params["SL"])
        #     dataSet.append(g)
        # else:
        g = to_gnngraph_nx(nx_g, features="random", targets=graph_property, SL=True)
        dataSet.append(g)
        dataSet_ptoperties.append(g.y)
    dataSet_ptoperties = np.array(dataSet_ptoperties)
    mean = np.mean(dataSet_ptoperties)
    sst = np.sum((dataSet_ptoperties - mean) ** 2)

    evaluate_loader = DataLoader(dataSet, batch_size=128, shuffle=False, drop_last=False)
    filelist = os.listdir(models_path)

    if "configs_record_scores.json" in filelist:
        filelist.remove("configs_record_scores.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("------------graph_property:{}------------".format(graph_property))
    for method in filelist:
        if ".json" in method:
            continue
        if graph_property not in method:
            continue
        method_path = os.path.join(models_path, method)


        if not ckpt_model :
            method_model_path = method_path + "/hold_out/trained_model/{}_metrics_model_trained".format(graph_property)
            model_eval,_=load_graph_model(method_model_path,device)
            model_eval.eval()
            ssr_all=0
            err_pred_all=[]
            record_method_eval[method] = dict()
            for n, data in enumerate(evaluate_loader):
                data = data.to(device)
                out = model_eval(data).squeeze()
                err_pred_batch=torch.abs(out-data.y)
                err_pred_all.append(err_pred_batch.detach())
                ssr_batch=torch.sum((out-data.y)**2)
                ssr_all+=ssr_batch.item()

            err_pred_all=torch.cat(err_pred_all,dim=0)
            record_method_eval[method]["ssr"] = ssr_all
            record_method_eval[method]["mean_err"] = torch.mean(err_pred_all).item()
            record_method_eval[method]["var_err"] = torch.var(err_pred_all,unbiased=False).item()
            record_method_eval[method]["max_err"] = torch.max(err_pred_all).item()
            record_method_eval[method]["min_err"] = torch.min(err_pred_all).item()
            record_method_eval[method]["median_err"] = torch.median(err_pred_all).item()
            record_method_eval[method]["r2"] = 1-ssr_all/sst.item()
        else:
            method_ckpt_path = method_path + "/hold_out/ckpt"
            filelist = os.listdir(method_ckpt_path)
            for ckpt_file in filelist:
                ckpt_model_path = os.path.join(method_ckpt_path, ckpt_file)
                model_eval, _ = load_graph_model(ckpt_model_path, device)
                model_eval.eval()
                ckpt_method=method+"_"+ckpt_file
                ssr_all = 0
                err_pred_all = []
                record_method_eval[ckpt_method] = dict()
                for n, data in enumerate(evaluate_loader):
                    data = data.to(device)
                    out = model_eval(data).squeeze()
                    err_pred_batch = torch.abs(out - data.y)
                    err_pred_all.append(err_pred_batch.detach())
                    ssr_batch = torch.sum((out - data.y) ** 2)
                    ssr_all += ssr_batch.item()

                err_pred_all = torch.cat(err_pred_all, dim=0)
                record_method_eval[ckpt_method]["ssr"] = ssr_all
                record_method_eval[ckpt_method]["mean_err"] = torch.mean(err_pred_all).item()
                record_method_eval[ckpt_method]["var_err"] = torch.var(err_pred_all, unbiased=False).item()
                record_method_eval[ckpt_method]["max_err"] = torch.max(err_pred_all).item()
                record_method_eval[ckpt_method]["min_err"] = torch.min(err_pred_all).item()
                record_method_eval[ckpt_method]["median_err"] = torch.median(err_pred_all).item()
                record_method_eval[ckpt_method]["r2"] = 1 - ssr_all / sst.item()
    method_best,method_best_score=max(record_method_eval.items(),key=lambda x:x[1]["r2"])
    print("best method:{} best score:{}".format(method_best,method_best_score["r2"]))
    # eval_metrics=["ssr","r2","mean_err","var_err","max_err","min_err","median_err"]
    # colors_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    # for n,metric in enumerate(eval_metrics):
    #     record_metric_method=[]
    #     plt.figure(figsize=(8, 5))  # 设置图像大小
    #     for method in filelist:
    #         if ".json" in method:
    #             continue
    #         record_metric_method.append(record_method_eval[method][metric])
    #     x = np.arange(len(record_metric_method))
    #     plt.plot(x, record_metric_method, label=metric, color=colors_list[n], linestyle=':')  # 点线
    #     plt.title('eval_metrics_{}\n dataset_{}_sst {}'.format(metric,graph_property, sst.item()))
    # plt.show()# 显示图例





if __name__ == '__main__':
    models_path = "results/robustness_metrics_models_natural_connectivity/grid_search"
    dataset_path = "dataset/train_dataset_new"
    # graph_properties=["global_CC","average_CC","natural_connectivity",
    #                   "global_efficiency"]
    graph_properties = ["natural_connectivity"]
    # filelist = os.listdir(models_path)
    # configs_record_scores=dict()
    # for method in filelist:
    #     if ".json" in method:
    #         continue
    #     method_path = os.path.join(models_path, method)
    #     scores_path = os.path.join(method_path, "hold_out/train_trace/record_scores.json")
    #     with open(scores_path, "r") as f:
    #         scores = json.load(f)
    #     configs_record_scores[method] = scores
    # record_scores_analysis(configs_record_scores,metrics="r2",train_validation="val_scores",score_thershold=0.8)# 0.23340462389640354
    #
    # graph_property="density"
    # evaluate_models_predictions(models_path, dataset_path, graph_property)
   #  for graph_property in graph_properties:
   #      evaluate_dataset_properties(dataset_path, graph_property)# no normalization:0.0192 normalzation:0.04
    for graph_property in graph_properties:
        evaluate_models_predictions(models_path, dataset_path, graph_property,ckpt_model=False)