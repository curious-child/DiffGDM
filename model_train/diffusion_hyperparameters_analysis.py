import os
import numpy as np
import yaml
import json
import matplotlib.pyplot as plt
def load_hyperparameters(yaml_file):
    with open(yaml_file, 'r') as file:
        hyperparameters = yaml.safe_load(file)
    return hyperparameters
def load_statistics_methods(target,file_path="result"):
    #
    if target=="dismantling":
        with open(file_path+"/graphs_all_dismantle_size.json","r") as f:
            graphs_reomve_size=json.load(f)
    elif target=="decycling":
        with open(file_path+"/graphs_all_decycling_size.json","r") as f:
            graphs_reomve_size=json.load(f)
    return graphs_reomve_size


def match_nested_dicts(dict1, dict2,match=True):
    matches = {}
    for key, value in dict1.items():
        if key in dict2:
            if isinstance(value, dict) and isinstance(dict2[key], dict):
                # 如果值是字典，则递归匹配
                nested_matches,match = match_nested_dicts(value, dict2[key],match)
                if nested_matches:
                    matches[key] = nested_matches
            elif isinstance(value, list):
                if dict2[key] in value:
                    matches[key] = dict2[key]
                else:
                    match=False
                # 如果值相等，记录匹配
    return matches,match
def flatten_dict(d, parent_key='', sep='*'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
def record_models_hyperparameters_and_scores(models_path, HP_file_path, statis_file_path, target,result_save_path):
    records = {}
    graphs_reomve_size=load_statistics_methods(target,statis_file_path)
    hyperparameters = load_hyperparameters(HP_file_path)
    print(hyperparameters)

    filelist = os.listdir(models_path)
    print(filelist)
    if "configs_record_scores.json" in filelist:
        filelist.remove("configs_record_scores.json")
    filelist.sort(key=lambda x: int(x[7:]))
    for method_name in filelist:
        if ".json" in method_name:
            continue
        method_path = models_path + "/" + method_name
        method_config_path = method_path + "/{}.yaml".format(method_name)
        with open(method_config_path, 'r') as f:
            method_config_param = yaml.safe_load(f)
        method_config_param_match,match = match_nested_dicts(hyperparameters, method_config_param)
        method = method_name.replace("config_", "")
        method_config_param_match[target] = graphs_reomve_size[method]
        records[method_name] = method_config_param_match
    if os.path.exists(result_save_path):
        print("waring:result_save_path文件夹目录已存在")
    else:
        os.mkdir(result_save_path)
    with open(result_save_path + '/records_statistics.json', 'w') as f:
        json.dump(records, f, indent=4, separators=(',', ':'))

def fliter_records(records,hyperparameters,target,result_save_path):
    filtered_records = {}
    for record_name, record_dict_value in records.items():
        filtered_record_dict_value,match = match_nested_dicts(hyperparameters, record_dict_value)
        if  match:
            filtered_record_dict_value[target]=record_dict_value[target]
            filtered_records[record_name]=filtered_record_dict_value

    with open(result_save_path + '/filtered_records_statistics.json', 'w') as f:
        json.dump(filtered_records, f, indent=4, separators=(',', ':'))
    return filtered_records
def multi_violin_box_scatter(name,data_dict, colors=None):
    """
    将多组数据的数据分布（以多种形式），展示在同一个图中。
    data_dict: key是数据名称，会显示在x轴上。
    """
    if colors is None:
        colors = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#E7DAD2"]
    if len(colors) < len(data_dict):
        raise ValueError("colors length must be greater than or equal to data_dict length")

    # 准备数据
    data = []
    categories = []
    for key in data_dict:
        data.append(data_dict[key])
        categories.append(key)

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6),num=name)

    # 绘制右半边的小提琴图
    for i in range(len(data)):
        parts = ax.violinplot(data[i], positions=[i], showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:  # 手动调整小提琴图形状，使其只显示右半边
            verts = pc.get_paths()[0].vertices  # 获取小提琴图的顶点
            verts[:, 0] = np.clip(verts[:, 0], i, np.inf)  # 只保留x方向大于中心位置的部分
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

    # 绘制箱线图（在每个类别的中间位置）
    box_positions = np.arange(len(data))
    ax.boxplot(data, positions=box_positions, widths=0.1, patch_artist=True,
               boxprops=dict(facecolor='white', color='black'),
               medianprops=dict(color='black'),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'),
               flierprops=dict(marker='o', color='black', markersize=1),  # 设置异常值点的样式
               showfliers=True  # 显示异常值
               )

    # 绘制散点图
    for i, d in enumerate(data):
        jittered_x = np.random.normal(loc=i - 0.1, scale=0.05, size=len(d))  # 添加随机偏移，使点分散开、不要在一条竖线上
        ax.scatter(jittered_x, d, color=colors[i], alpha=0.8, s=1)

    # 设置x轴和y轴
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylabel('Average size of {} set (%)'.format(target))
    ax.set_title("diffusion model: DFM")# D3PM,DFM
    # 显示图像
    plt.tight_layout()
    plt.show()

def analysis_records(records_path,HP_file_path,target,result_save_path):

    if os.path.exists(records_path + '/analysis_records_statistics.json'):
        with open(records_path + '/analysis_records_statistics.json', 'r') as f:
            analysis_result = json.load(f)
    else:
        hyperparameters = load_hyperparameters(HP_file_path)
        hyperparameters_flat = flatten_dict(hyperparameters)
        analysis_result = {}
        with open(records_path + '/records_statistics.json', 'r') as f:
            records = json.load(f)
        records = fliter_records(records,hyperparameters,target,result_save_path)
        for hyperparameter_name, hyperparameter_value in hyperparameters_flat.items():
            analysis_result[hyperparameter_name] = {}
            for value in hyperparameter_value:
                analysis_result[hyperparameter_name][value] = []
            for method_name, method_config_param_match in records.items():
                method_config_param_match_flat = flatten_dict(method_config_param_match)

                method_config_param_match_flat_value = method_config_param_match_flat[hyperparameter_name]
                if method_config_param_match_flat_value in hyperparameter_value:
                    analysis_result[hyperparameter_name][method_config_param_match_flat_value].append(method_config_param_match[target])
                # else:
                #     raise ValueError("hyperparameter_value not in hyperparameter_value")
        if os.path.exists(result_save_path):
            print("waring:result_save_path文件夹目录已存在")
        else:
            os.mkdir(result_save_path)
        with open(result_save_path + '/analysis_records_statistics.json', 'w') as f:
            json.dump(analysis_result, f, indent=4, separators=(',', ':'))
    for hyperparameter_name, hyperparameter_value in analysis_result.items():
        print(hyperparameter_name)

     #   plt.figure(figsize=(10, 5),num=hyperparameter_name)
        # scores_list=[]
        # values_list=[]
        # for value, scores in hyperparameter_value.items():
        #     values_list.append(value)
        #     scores_list.append(scores)
        # plt.boxplot(scores_list, labels=values_list)
        # plt.legend()
        # plt.show()
        multi_violin_box_scatter(name=hyperparameter_name, data_dict=hyperparameter_value)

if __name__ == '__main__':
    models_grid = "trained_model_discretFlow_final"
    target = "decycling"

    models_path = "results/diffusion_model/{}/grid_infer".format(models_grid)
    HP_file_path = 'HP_analysis_result/{}_{}/hyperparameters.yaml'.format(models_grid,target)  # 直接人为构建 超参数文件中 可变化的部分及范围
    statis_file_path = "analysis/result/{}_{}".format(models_grid, target)
    result_save_path = "HP_analysis_result/{}_{}".format(models_grid, target)
    records_path=result_save_path

    record_models_hyperparameters_and_scores(models_path, HP_file_path, statis_file_path, target,result_save_path=result_save_path)
    analysis_records(records_path, HP_file_path, target, result_save_path)

