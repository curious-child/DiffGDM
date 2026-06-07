import json
import multiprocessing

import torch_geometric
import yaml
from scipy.fftpack import ifftn

from configs.configs_model_metrics import parse_args
import os
import numpy as np
from utils.utils import pre_DataSet, save_config, grid_parameters_supervised_learning
from sklearn.model_selection import train_test_split,KFold


import itertools as it
from train.train_vanilla import run_training
#from train.train_diffusion import run_training
import matplotlib.pyplot as plt
from utils.data_visualization import model_evaluation_metrics_curves, model_evaluation_metrics_curves_vanilla
import random
import torch

def seed_torch(seed=1029):
    random.seed(seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)   # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True   # 选择确定性算法


# 留出法 模型性能评估
def hold_out_score(dataset,train_param,net_param,loss_param,optimizer_param,records_path,configs_counts=0,vision_show=True):
    save_data_path = os.path.join(records_path,"hold_out")
    if os.path.exists(save_data_path):
        print("waring:hold out文件夹目录已存在")
    else:
        os.mkdir(save_data_path)

    trainset,validationset = train_test_split(dataset,train_size=train_param["traindata_size"])

    record_scores=run_training(trainset=trainset,
                             validationset=validationset,
                             train_param=train_param,
                             net_param=net_param,
                             loss_param=loss_param,
                             optimizer_param=optimizer_param,
                             records_path=save_data_path)
    # record_scores_path=os.path.join(save_data_path,"train_trace")
    # with open(record_scores_path + '/record_scores.json', 'r') as f:
    #     record_scores=json.load(f)
    #可视化 基于不同模型性能指标的训练轨迹 此处可拓展 也可略去
    if vision_show==True:
       fig = plt.figure("metric—curves_configs_{}".format(configs_counts))
       model_evaluation_metrics_curves_vanilla(fig=fig,score_metrics=train_param['score_metrics'],record_scores=record_scores)
       fig.savefig(save_data_path+"/metric—curves_configs_{}".format(configs_counts))
       plt.close(fig)
    return record_scores

# 交叉验证 模型性能评估
#未修改 暂略
def cross_val_score(dataset,train_param,net_param,loss_param,optimizer_param,records_path,configs_counts=0,vision_show=True):
    path = os.path.join(records_path ,"cross_val")
    if os.path.exists(path):
        print("waring:cross validation文件夹目录已存在")
    else:
        os.mkdir(path)
    data_induce = np.arange(0, len(dataset))
    m_fold = KFold(n_splits=train_param["n_splits"])
    average_scores = {
        "epoch": list(),
    }
    for score_metric in train_param["score_metrics"]:
        average_scores[score_metric] = dict()

    for n, (train_index, val_index) in enumerate(m_fold.split(dataset)):
        save_data_path = path + "/random_{}".format(n)
        if os.path.exists(save_data_path):
            print("waring:cross validation文件夹目录已存在")
        else:
            os.mkdir(save_data_path)
        train_index=train_index.tolist()
        val_index = val_index.tolist()

        trainset = [dataset[i] for i in train_index]
        validationset=[dataset[i] for i in val_index]


        record_scores = run_training(trainset=trainset,
                                     validationset=validationset,
                                     train_param=train_param,
                                     net_param=net_param,
                                     loss_param=loss_param,
                                     optimizer_param=optimizer_param,
                                     records_path=save_data_path)
        #交叉验证 模型性能指标平均值
        if n==0:
            average_scores['epoch']=average_scores['epoch']
            for score_metric in train_param['score_metrics']:
                average_scores[score_metric]["train_scores"]=np.array(record_scores[score_metric]["train_scores"])
                average_scores[score_metric]['val_scores']  =np.array(record_scores[score_metric]['val_scores'])
        else:
            for score_metric in train_param['score_metrics']:
                average_scores[score_metric]["train_scores"]=(np.array(record_scores[score_metric]["train_scores"])+n*average_scores[score_metric]["train_scores"])/(n+1)
                average_scores[score_metric]['val_scores']  =(np.array(record_scores[score_metric]['val_scores'])+n*average_scores[score_metric]['val_scores'])/(n+1)
    for score_metric in train_param["score_metrics"]:
        if type(average_scores[score_metric]['val_scores']) is np.ndarray or type(average_scores[score_metric]['train_scores']) is np.ndarray:
            average_scores[score_metric]['train_scores'] = average_scores[score_metric]['train_scores'].tolist()
            average_scores[score_metric]['val_scores'] = average_scores[score_metric]['val_scores'].tolist()
    #保存
    with open(path + '/average_scores.json', 'w') as f:
        json.dump(average_scores, f, indent=4, separators=(',',':'))

    # 可视化 基于不同模型性能指标的训练轨迹 （此处可拓展 也可略去）
    if vision_show == True:
        fig = plt.figure("metric—curves_configs_{}".format(configs_counts))
        model_evaluation_metrics_curves(fig=fig,
                                        t_score_metrics=train_param['t_score_metrics'],
                                        v_score_metrics=train_param['v_score_metrics'],
                                        t_v_metrics_same=train_param["t_v_metrics_same"],
                                        record_scores=average_scores)
        plt.show()
    return average_scores

def grid_search(dataset_params,train_params,net_params,loss_params,optimizer_params,records_path):
    grid_search_path = records_path + "/grid_search"
    if os.path.exists(grid_search_path):
        print("waring:grid_search文件夹目录已存在")
    else:
        os.mkdir(grid_search_path)

    configs_record_scores=dict()

    for params_values in it.product(  *dataset_params.values() ):
        dataset_param=dict(zip(dataset_params.keys(),params_values))
        dataset = pre_DataSet(**dataset_param)
        configs_count = 0
        parameters_list=grid_parameters_supervised_learning(train_params,net_params,loss_params,optimizer_params)

        for train_param,net_param,loss_param,optimizer_param in parameters_list:

            seed_torch(123)
            save_config_path = os.path.join(grid_search_path , "{}_metrics_{}".format(dataset_param["targets"],configs_count))
            if os.path.exists(save_config_path):
                print("waring:config文件夹目录已存在")
            else:
                os.mkdir(save_config_path)
            # 保存每一个训练模型参数,并检查是否已训练
            net_param["num_features"] = dataset[0].num_node_features
            Not_train_flag=save_config(path=save_config_path, configs_name="{}_metrics_{}.yaml".format(dataset_param["targets"],configs_count),
                        dataset_param=dataset_param, train_param=train_param,
                        net_param=net_param, loss_param=loss_param, optimizer_param=optimizer_param)
            if Not_train_flag:
                if train_param["model_evaluation"]=="hold_out":
                    record_scores= hold_out_score(dataset=dataset,train_param= train_param, net_param=net_param,loss_param= loss_param, optimizer_param=optimizer_param,
                                   records_path=save_config_path,configs_counts=configs_count)
                elif train_param["model_evaluation"]=="cross_val":
                    record_scores= cross_val_score(dataset=dataset,train_param= train_param, net_param=net_param,loss_param= loss_param,  optimizer_param=optimizer_param,
                                    records_path=save_config_path, configs_counts=configs_count)
                else:
                    raise ValueError("the definition of model_evaluation don't exit\n"
                                     "\tyou can define it before using it")
                configs_record_scores[configs_count]=record_scores
            #
            configs_count+=1
            #各模型性能评价指标记录，用于分析模型超参数选择




    #

    with open(grid_search_path + '/configs_record_scores.json', 'w') as f:
        json.dump(configs_record_scores, f, indent=4, separators=(',', ':'))


def train_wrap(dataset,dataset_param,train_param,net_param,loss_param,optimizer_param,grid_search_path,configs_count=0):
    seed_torch(123)
    save_config_path = os.path.join(grid_search_path, "config_{}".format(configs_count))
    if os.path.exists(save_config_path):
        print("waring:config文件夹目录已存在")
    else:
        os.mkdir(save_config_path)

    if train_param["model_evaluation"] == "hold_out":
        record_scores = hold_out_score(dataset=dataset, train_param=train_param, net_param=net_param,
                                       loss_param=loss_param, optimizer_param=optimizer_param,
                                       records_path=save_config_path, configs_counts=configs_count)
    elif train_param["model_evaluation"] == "cross_val":
        record_scores = cross_val_score(dataset=dataset, train_param=train_param, net_param=net_param,
                                        loss_param=loss_param, optimizer_param=optimizer_param,
                                        records_path=save_config_path, configs_counts=configs_count)
    else:
        raise ValueError("the definition of model_evaluation don't exit\n"
                         "\tyou can define it before using it")
    # 保存每一个训练模型参数
    save_config(path=save_config_path, configs_name="config_{}.yaml".format(configs_count),
                dataset_param=dataset_param, train_param=train_param,
                net_param=net_param, loss_param=loss_param, optimizer_param=optimizer_param)
    return  record_scores #
def parallel_grid_search(dataset_params,train_params,net_params,loss_params,optimizer_params,records_path):
    grid_search_path = records_path + "/grid_search"
    if os.path.exists(grid_search_path):
        print("waring:grid_search文件夹目录已存在")
    else:
        os.mkdir(grid_search_path)
    pool=multiprocessing.Pool()
    results_async=list()
    configs_count=0
   # configs_record_scores=list()

    for params_values in it.product(  *dataset_params.values() ):
        dataset_param=dict(zip(dataset_params.keys(),params_values))
      #  print(dataset_param)
        dataset = pre_DataSet(file_path=dataset_param["dataset_path"], features=dataset_param["dataset_features"],
                          targets=dataset_param["target"], filter=dataset_param["graph_filter"])
        for params_values in it.product(*train_params.values()):
            train_param = dict(zip(train_params.keys(), params_values))
          #  print(train_param)

            net_params_copy=net_params.copy()
            net_params_copy.pop("prelayers_gnn_param")
            net_params_copy.pop("enclayers_gnn_param")
            net_params_copy.pop("mask_model")
            for params_values in it.product(*net_params_copy.values()):
                net_param = dict(zip(net_params_copy.keys(), params_values))
                net_param["num_features"]=dataset[0].num_node_features
                net_param["batch_nodes_train"]=train_param["batch_size"]*dataset[0].num_nodes# stablenet w
                print(net_param["prelayers_gnn"])
                print(net_param["enclayers_gnn"])
                for prelayers_values in it.product(*net_params["prelayers_gnn_param"][net_param["prelayers_gnn"]].values() ):
                    net_param["prelayers_gnn_param"]=dict(zip(net_params["prelayers_gnn_param"][net_param["prelayers_gnn"]].keys(),prelayers_values))
                    print(net_param["prelayers_gnn_param"])
                    for enclayers_values in it.product( *net_params["enclayers_gnn_param"][net_param["enclayers_gnn"]].values()):
                        net_param["enclayers_gnn_param"] = dict( zip(net_params["enclayers_gnn_param"][net_param["enclayers_gnn"]].keys(), enclayers_values))
                        print(net_param["enclayers_gnn_param"])
                        mask_net_params_copy=net_params["mask_model"]
                        mask_net_params_copy.pop("prelayers_gnn_param")
                        for mask_net_values in it.product(*mask_net_params_copy.values()):
                            net_param["mask_model"] = dict(zip(mask_net_params_copy.keys(), mask_net_values))
                            for mask_prelayers_values in it.product(*net_params["mask_model"]["prelayers_gnn_param"][net_param["mask_model"]["prelayers_gnn"]].values() ):
                                net_param["mask_model"]["prelayers_gnn_param"]=dict(zip(net_params["mask_model"]["prelayers_gnn_param"][net_param["mask_model"]["prelayers_gnn"]].keys(),mask_prelayers_values))

                                for params_values in it.product(*loss_params.values()):
                                    loss_param = dict(zip(loss_params.keys(), params_values))
                                    for params_values in it.product(*optimizer_params.values()):
                                        optimizer_param = dict(zip(optimizer_params.keys(), params_values))


                                        results_async.append(
                                            pool.apply_async(func=train_wrap,
                                                             args=(
                                                                   dataset,
                                                                   dataset_param,
                                                                   train_param,
                                                                   net_param,
                                                                   loss_param,
                                                                   optimizer_param,
                                                                   grid_search_path,
                                                                   configs_count
                                                                   )
                                                             )
                                        )

                                        configs_count+=1
                                        #各模型性能评价指标记录，用于分析模型超参数选择

    #
    pool.close()
    pool.join()

    configs_record_scores=[scores.get() for scores in results_async]
    with open(grid_search_path + '/configs_record_scores.json', 'w') as f:
        json.dump(configs_record_scores, f, indent=4, separators=(',', ':'))




if __name__ == '__main__':
    args = parse_args()

    with open(args.cfg,'r') as f:
        config_params=yaml.safe_load(f)
    dataset_param=config_params["dataset"]
    train_param =config_params["train"]
    net_param   =config_params['net']
    loss_param  =config_params['loss']
    optimizer_param=config_params['optimizer']

    records_path=config_params["out_dir"]
    if os.path.exists(records_path):
        print("records_path:{}文件夹目录已存在".format(records_path))
    else:
        os.mkdir(records_path)

    if args.train_mode == "grid":
        grid_search(dataset_params=dataset_param,train_params=train_param,
                net_params=net_param, loss_params=loss_param,
                optimizer_params=optimizer_param, records_path=records_path)
        # parallel_grid_search(dataset_params=dataset_param, train_params=train_param,
        #             net_params=net_param, loss_params=loss_param,
        #             optimizer_params=optimizer_param, records_path=records_path)
    elif args.train_mode=="hold_out":
        dataset = pre_DataSet(file_path=dataset_param["dataset_path"], features=dataset_param["dataset_features"],
                              targets=dataset_param["target"], filter=dataset_param["graph_filter"])
        hold_out_score( dataset=dataset, train_param=train_param,
                        net_param=net_param, loss_param=loss_param,
                        optimizer_param=optimizer_param, records_path=records_path)
        save_config(path=records_path, configs_name="configs_hold_out.yaml",
                    dataset_param=dataset_param, train_param=train_param,
                    net_param=net_param, loss_param=loss_param, optimizer_param=optimizer_param)
    elif args.train_mode == "cross_val":
        dataset = pre_DataSet(file_path=dataset_param["dataset_path"], features=dataset_param["dataset_features"],
                              targets=dataset_param["target"], filter=dataset_param["graph_filter"])
        cross_val_score(dataset=dataset, train_param=train_param,
                       net_param=net_param, loss_param=loss_param,
                       optimizer_param=optimizer_param, records_path=records_path)
        save_config(path=records_path, configs_name="configs_cross_val.yaml",
                    dataset_param=dataset_param, train_param=train_param,
                    net_param=net_param, loss_param=loss_param, optimizer_param=optimizer_param)
