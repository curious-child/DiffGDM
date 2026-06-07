import random
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import pandas as pd
import os
def kendall_correlation_analysis(ax,values_dic,title=False,dataname="test"):
    """

    :param fig:
    :param ax:
    :param values_dic: dict,输入为节点的重要性度量值
    :param title:
    :param dataname:
    :return:
    """
    #共有维度指标
    dimension_set={"8","9","13","21","Degree", "CC", "Kcore", "Chi degree"}#
   # dimension_set = {"5", "6", "11", "20","25","28","31","36","38", "Degree", "CC", "Kcore", "Chi degree"}
   #  dimension_set = {"2","5", "6","7", "9","12","15","17", "20", "21","22","39", "Degree", "CC", "Kcore", "Chi degree"}
   #  dimension_set={"Degree", "CC", "Kcore", "Chi degree"}
    rank_correlation={}

    for method_name_1, values_1 in values_dic.items():
        rank_correlation[method_name_1]={}

        for method_name_2,values_2 in values_dic.items():

           rank_correlation[method_name_1][method_name_2],_=stats.kendalltau(x=values_1,y=values_2)

    df=pd.DataFrame(rank_correlation)
    df = df.dropna(axis=0, how="all")  # 删除全为nan的行
    df = df.dropna(axis=1, how="all")  # 删除全为nan的列
    indices_set = set(df.index.tolist())
    redundant_indices = indices_set - dimension_set

    ax=sns.heatmap(df,ax=ax,annot=True, vmax=1, vmin=-1, square=False, cmap="RdBu_r", fmt='.2g',annot_kws={"fontsize":20})
    plt.setp(ax.get_xticklabels(),rotation="horizontal",fontsize=15)# "vertical"竖立  horizontal 横放
    plt.setp(ax.get_yticklabels(), fontsize=15)  # "vertical"竖立  horizontal 横放
    print("redundant indices: {}".format(redundant_indices))
    if title == True:
        ax.set_title("{} \n redundant indices: {}".format(dataname,redundant_indices),fontsize=30)
    plt.tight_layout()
def set_correlation_analysis_average(ax,values_dic,graphs_list,title=False,dataname="test"):
    """

    :param fig:
    :param ax:
    :param values_dic: dict,输入为节点的重要性度量值
    :param title:
    :param dataname:
    :return:
    """
    rank_correlation={}

    for method_name_1, dict_set_1 in values_dic.items():
        rank_correlation[method_name_1]={}
        for method_name_2, dict_set_2 in values_dic.items():
            correlation_coefficient=0
            for graph in graphs_list:
                intersection_set_len=len(dict_set_1[graph]&dict_set_2[graph])
                union_set_len=len(dict_set_1[graph]|dict_set_2[graph])
                #difference_set_len = len(dict_set_1[graph]-dict_set_2[graph])# 对称差集 ^ 差集-
                correlation_coefficient += (intersection_set_len ) / union_set_len
            rank_correlation[method_name_1][method_name_2]=correlation_coefficient/len(graphs_list)
    df=pd.DataFrame(rank_correlation)

    ax = sns.heatmap(df, ax=ax, annot=True, vmax=1, square=False, cmap="Blues", fmt='.2g', annot_kws={"fontsize": 20})
    plt.setp(ax.get_xticklabels(), rotation="horizontal", fontsize=15)  # "vertical"竖立  horizontal 横放
    plt.setp(ax.get_yticklabels(), fontsize=15)  # "vertical"竖立  horizontal 横放
    if title == True:
        ax.set_title("{}".format(dataname))
    plt.tight_layout()

def set_correlation_analysis(ax,values_dic,title=False,dataname="test"):
    """

    :param fig:
    :param ax:
    :param values_dic: dict,输入为节点的重要性度量值
    :param title:
    :param dataname:
    :return:
    """
    rank_correlation={}

    for method_name_1, set_1 in values_dic.items():
        rank_correlation[method_name_1]={}
        for method_name_2, set_2 in values_dic.items():
            intersection_set_len=len(set_1&set_2)
            union_set_len=len(set_1|set_2)
           # difference_set_len=len(set_1-set_2)
            
            correlation_coefficient=(intersection_set_len)/union_set_len
            rank_correlation[method_name_1][method_name_2]=correlation_coefficient
    df=pd.DataFrame(rank_correlation)
    ax = sns.heatmap(df, ax=ax, annot=True, vmax=1, square=False, cmap="Blues", fmt='.2g', annot_kws={"fontsize": 20})
    plt.setp(ax.get_xticklabels(), rotation="horizontal", fontsize=15)  # "vertical"竖立  horizontal 横放
    plt.setp(ax.get_yticklabels(), fontsize=15)  # "vertical"竖立  horizontal 横放
    if title == True:
        ax.set_title("{}".format(dataname))
    plt.tight_layout()


if __name__=="__main__":
    import numpy as np
    x=np.random.random(4)#4 2 3 1   remove 4 2 3 1 4213
    y=np.random.random(4)#3 4 2 1   remove 4 3 1 2 4321

    tao,_=stats.kendalltau(x,y)
    p,_=stats.spearmanr(x,y)
    print(tao,p)
    predictions = list(zip(range(4), x))
    print(predictions)
    sorted_prediction = sorted(predictions, key=itemgetter(1), reverse=True)
    removal_order = list(map(itemgetter(0), sorted_prediction))
    x = list(map(int, removal_order))
    predictions = list(zip(range(4), y))
    print(predictions)
    sorted_prediction = sorted(predictions, key=itemgetter(1), reverse=True)
    removal_order = list(map(itemgetter(0), sorted_prediction))
    y = list(map(int, removal_order))
    tao, _ = stats.kendalltau(x, y)
    p, _ = stats.spearmanr(x, y)
    print(tao, p)