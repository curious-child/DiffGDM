import os
import json
import time

from torch_geometric.utils import degree
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import  DataLoader
from utils.utils import save_checkpoint
from loss_functions.loss_functions import train_loss,evaluation_score,loss_wrapper
from optimizers.optimizers import train_optimizers,train_schedulers
from models.models import diffusion_models

def run_training( trainset, validationset,train_param,net_param,loss_param,optimizer_param,records_path):
    # 评价指标 记录变量初始化
    record_scores={
        "epoch":list(),
        "train_scores":list(),
        "val_scores":list(),
    }
    # 加载数据集

    train_loader = DataLoader(trainset, batch_size=train_param["train_batch_size"], shuffle=True, drop_last=False)
    val_loader   = DataLoader(validationset, batch_size=train_param["val_batch_size"], shuffle=False, drop_last=False)
    #设置训练模型参数、优化器参数、损失函数参数
    net_param["device"] = train_param["device"]
    train_GNN = diffusion_models(task_model=net_param["task_model"],net_param=net_param).to(train_param["device"])
    optimizer = train_optimizers(train_GNN.parameters(),optimizer_param)
    if optimizer_param["scheduler_set"]==True:
        scheduler= train_schedulers(optimizer,optimizer_param)

    #设置模型性能评价指标参数

    #开始训练
    for epoch in range(train_param["train_epochs"]):
        train_GNN.train()
        # 模型性能评价指标值 初始化
        train_score=0
        val_score=0

        #批量训练
        for n,data in enumerate(train_loader):
            optimizer.zero_grad()  # Clear gradients.
            #print("\r******************train_epoch:{} batch:{}********************\n".format(epoch,n),end=" ")
            data = data.to(train_param["device"])
           # time_step=time.time()
            #out: [N,2]
            loss = train_GNN.training_step(batch=data, device=train_param["device"])  # diffusion training

            #time_train=time.time()
            #print("train time_cost:{}".format(time_train-time_step))

            loss.backward()
            optimizer.step()
            train_score=loss.cpu().detach().item()

        if train_GNN.timewarping_set and epoch>0:
            train_GNN.scheduler_train.update_optimal_parameters(history=2000)
            print("update optimal parameters")
        if optimizer_param["scheduler_set"] == True:
            scheduler.step()
        print("\rtrain_epoch:{}".format(epoch),end=" ")
        print("train_loss:{}".format(train_score))
        #validation
        with torch.no_grad():
            if  train_param["test_set"]:

                train_GNN.eval()
                for n, data in enumerate(val_loader):
                    data = data.to(train_param["device"])
                    loss = train_GNN.training_step(batch=data, device=train_param["device"])  # diffusion training
                   # _=train_GNN.decode_step(batch=data, device=train_param["device"])
                    val_score = loss.cpu().detach().item()


        print("val_loss:{}".format(val_score))
        record_scores["epoch"].append(epoch)

        record_scores["train_scores"].append(train_score)

        record_scores["val_scores"].append(val_score)
        #save model parameters in training process
        if epoch%train_param["ckpt_period"]==0 and epoch!=0 and train_param["ckpt"]:
            sckpt_path =os.path.join(records_path,"ckpt")
            if os.path.exists(sckpt_path):
                print("ckpt文件夹目录已存在")
                pass
            else:
                os.mkdir(sckpt_path)
            save_checkpoint(path=sckpt_path,model_name="NDlearner_diffusion_{}iter".format(epoch),model=train_GNN,net_param=net_param)
    #save fininal trainde model 文件夹可略去
    model_path = os.path.join(records_path,"trained_model")
    if os.path.exists(model_path):
        print("trained_model文件夹目录已存在")
    else:
        os.mkdir(model_path)
    save_checkpoint(path=model_path, model_name="NDlearner_diffusion_trained", model=train_GNN,
                    net_param=net_param)
    #保存训练模型性能指标数据 文件夹可略去
    data_path = os.path.join(records_path, "train_trace")
    if os.path.exists(data_path):
        print("warning:train_trace文件夹目录已存在")
    else:
        os.mkdir(data_path)
    with open(data_path+'/record_scores.json', 'w') as f:
        json.dump(record_scores, f, indent=4, separators=(',', ':'))
    return record_scores









