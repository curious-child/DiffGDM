import networkx as nx
import numpy as np
import pandas as pd
import os
from pathlib import Path


def calculate_network_metrics(graphml_path):
    """
    计算 GraphML 文件的多种网络指标

    参数:
    graphml_path (str): GraphML 文件路径

    返回:
    dict: 包含各种网络指标的字典
    """
    # 读取 GraphML 文件
    G = nx.read_graphml(graphml_path)

    metrics = {}

    # 计算 IDP(1,3,4) - Interval Degree Probabilities
    degrees = [d for _, d in G.degree()]
    n_nodes = len(degrees)

    if n_nodes == 0:
        raise ValueError("图中没有节点")

    # 计算均值和标准差
    mu = np.mean(degrees)
    sigma = np.std(degrees, ddof=0)  # 总体标准差

    # 处理标准差为0的情况（所有节点度相同）
    if sigma == 0:
        # 特殊情况：所有度相同，IDP(4) 为1（因为 μ+σ = μ > μ-σ）
        metrics['IDP'] = {'IDP(1)': 0.0, 'IDP(3)': 0.0, 'IDP(4)': 1.0}
    else:
        # 计算 IDP(1): [1, μ-σ)
        high1 = mu - sigma
        idp1 = 0.0
        if high1 > 1:
            idp1 = np.mean((np.array(degrees) >= 1) & (np.array(degrees) < high1))

        # 计算 IDP(3): [μ, μ+σ)
        idp3 = np.mean((np.array(degrees) >= mu) & (np.array(degrees) < mu + sigma))

        # 计算 IDP(4): [μ+σ, ∞)
        idp4 = np.mean(np.array(degrees) >= mu + sigma)

        metrics['IDP'] = {'IDP(1)': idp1, 'IDP(3)': idp3, 'IDP(4)': idp4}

    # 计算 Assortativity - 同配系数
    try:
        if nx.is_directed(G):
            # 对于有向图，使用有向图的同配系数
            metrics['assortativity'] = nx.degree_assortativity_coefficient(G)
        else:
            # 对于无向图，使用无向图的同配系数
            metrics['assortativity'] = nx.degree_assortativity_coefficient(G)
    except:
        # 如果无法计算同配系数，返回 NaN
        metrics['assortativity'] = float('nan')

    # 计算 Average Clustering Coefficient - 平均局部聚类系数
    try:
        metrics['avg_clustering'] = nx.average_clustering(G)
    except:
        metrics['avg_clustering'] = float('nan')

    # 计算 Average Path Length Log - 平均路径长度除以节点数量的对数
    try:
        if nx.is_connected(G):
            # 对于连通图，直接计算平均最短路径长度
            avg_path_length = nx.average_shortest_path_length(G)
        else:
            # 对于非连通图，计算最大连通分量的平均路径长度
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            if len(largest_cc) > 1:
                avg_path_length = nx.average_shortest_path_length(subgraph)
            else:
                avg_path_length = 0

        n = G.number_of_nodes()
        if n > 1:
            avg_path_length_log = avg_path_length / np.log(n)
        else:
            avg_path_length_log = 0

        metrics['avg_path_length_log'] = avg_path_length_log
    except:
        metrics['avg_path_length_log'] = float('nan')

    # 计算 Max. deg. n. - 归一化最大度 (归一化度中心性)
    try:
        if n_nodes > 1:
            max_degree = max(degrees)
            # 归一化最大度 = 最大度 / (节点数 - 1)，这是理论上的最大可能度
            normalized_max_degree = max_degree / (n_nodes - 1) if n_nodes > 1 else 0
        else:
            normalized_max_degree = 0
        metrics['normalized_max_degree'] = normalized_max_degree
    except:
        metrics['normalized_max_degree'] = float('nan')

    # 计算 Max. eigen. - 最大特征向量中心性
    try:
        # 计算特征向量中心性
        eig_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
        # 获取最大的特征向量中心性值
        max_eigen_centrality = max(eig_centrality.values()) if eig_centrality else 0
        metrics['max_eigenvector_centrality'] = max_eigen_centrality
    except:
        # 如果无法计算特征向量中心性，返回 NaN
        metrics['max_eigenvector_centrality'] = float('nan')

    return metrics


def process_dataset():
    """
    处理 Excel 表格中的网络文件名列表，计算每个网络的指标并将结果保存到 Excel 中
    """
    # 读取 Excel 表格
    excel_file = "dataset.xlsx"  # 假设 Excel 文件名为 dataset.xlsx

    if not os.path.exists(excel_file):
        print(f"错误: 找不到 Excel 文件 '{excel_file}'")
        return

    df = pd.read_excel(excel_file)

    # 假设第一列是网络文件名（列名可能需要根据实际情况调整）
    network_column = df.columns[0]  # 第一列
    network_filenames = df[network_column].tolist()

    # 创建 dataset 目录路径
    dataset_dir = Path("../dataset")

    # 初始化结果列
    result_columns = ['IDP(1)', 'IDP(3)', 'IDP(4)', 'Assortativity', 'Avg. Clustering',
                      'Avg. Path Length Log', 'Max. Deg. N.', 'Max. Eigen.']

    # 为结果添加列到 DataFrame
    for col in result_columns:
        df[col] = np.nan

    # 遍历每个网络文件名
    for idx, filename in enumerate(network_filenames):
        # 构建完整的文件路径
        network_path = dataset_dir / f"{filename}.graphml"

        if not network_path.exists():
            print(f"警告: 文件不存在 '{network_path}'，跳过此文件")
            continue

        try:
            # 计算网络指标
            metrics = calculate_network_metrics(str(network_path))

            # 将结果写入对应的行
            df.at[idx, 'IDP(1)'] = metrics['IDP']['IDP(1)']
            df.at[idx, 'IDP(3)'] = metrics['IDP']['IDP(3)']
            df.at[idx, 'IDP(4)'] = metrics['IDP']['IDP(4)']
            df.at[idx, 'Assortativity'] = metrics['assortativity']
            df.at[idx, 'Avg. Clustering'] = metrics['avg_clustering']
            df.at[idx, 'Avg. Path Length Log'] = metrics['avg_path_length_log']
            df.at[idx, 'Max. Deg. N.'] = metrics['normalized_max_degree']
            df.at[idx, 'Max. Eigen.'] = metrics['max_eigenvector_centrality']

            print(f"已处理: {filename} ({idx + 1}/{len(network_filenames)})")

        except Exception as e:
            print(f"错误: 处理文件 '{network_path}' 时出错: {str(e)}")
            continue

    # 将结果保存回 Excel 文件
    output_filename = "dataset_with_results.xlsx"
    df.to_excel(output_filename, index=False)
    print(f"结果已保存到 '{output_filename}'")


# 主程序入口
if __name__ == "__main__":
    process_dataset()