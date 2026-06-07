import os

import igraph
import numpy as np
from igraph import *
import networkx as nx
import random
import math


def generate_random_sbm(n_total=2000):
    # 1. 随机确定社区数量 k (例如 5 到 15 之间)
    k = random.randint(5, 15)

    # 2. 随机分配每个社区的节点数 (总和为 n_total)
    # 使用狄利克雷分布生成不均匀的比例
    proportions = np.random.dirichlet([2.0] * k)
    sizes = np.random.multinomial(n_total, proportions).tolist()
    # 确保没有规模为 0 的社区
    sizes = [max(1, s) for s in sizes]

    # 3. 构造随机概率矩阵
    # 预设概率范围 (针对 500-5000 规模优化的经验值)
    p_in_range = (0.02, 0.2)
    p_out_range = (0.005, 0.02)

    probs = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            if i == j:
                # 社区内部概率随机
                probs[i][j] = random.uniform(*p_in_range)
            else:
                # 社区间连接概率随机
                p_val = random.uniform(*p_out_range)
                probs[i][j] = p_val
                probs[j][i] = p_val  # 保证矩阵对称

    # 4. 生成模型
    G = nx.stochastic_block_model(sizes, probs.tolist(),seed=random.randint(0, 10000))
    G.graph['partition'] = str(G.graph['partition'])
    #print(G)
    return G
def gen_graph(g_type, cur_n=None, num_min=20, num_max=40):
    if cur_n is None:
        max_n = num_max
        min_n = num_min
        cur_n = np.random.randint(max_n - min_n + 1) + min_n

    if g_type == 'erdos_renyi':
        while True:
            ER_p = random.uniform(1./cur_n,(np.log(cur_n)+1.)/cur_n)
            # NetworkX 的 Erdos-Renyi 图生成函数
            g = nx.erdos_renyi_graph(n=cur_n, p=ER_p, directed=False)
            if nx.is_connected(g):
                break
            else:
                G_giant = max(nx.connected_components(g), key=len)
                if len(G_giant) > num_min:
                    g = g.subgraph(G_giant).copy()
                    break

    elif g_type == 'small-world':
        while True:
            nei_max = 7
            nei_min = 2
            WS_nei = 2*random.randint(nei_min, nei_max)
            WS_p = random.uniform(0.02, 0.2)
            # NetworkX 的小世界图生成函数
            g = nx.watts_strogatz_graph(n=cur_n, k=WS_nei, p=WS_p)
            if nx.is_connected(g):
                break
            else:
                G_giant = max(nx.connected_components(g), key=len)
                if len(G_giant) > num_min:
                    g = g.subgraph(G_giant).copy()
                    break

    elif g_type == 'barabasi_albert':
        while True:
            m_max = 4
            m_min = 2
            BA_m = random.randint(m_min, m_max)
            # NetworkX 的 BA 无标度图生成函数
            g = nx.barabasi_albert_graph(n=cur_n, m=BA_m)
            if nx.is_connected(g):
                break
            else:
                G_giant = max(nx.connected_components(g), key=len)
                if len(G_giant) > num_min:
                    g = g.subgraph(G_giant).copy()
                    break

    elif g_type == "static_power_law":
        while True:
         #   print("generate static power-law graph")
            # NetworkX 中没有直接对应的静态幂律图函数
            # 可以使用配置模型生成具有幂律度分布的图
            exp = random.uniform(2, 3)
            # 生成幂律度序列
            sequence = nx.utils.powerlaw_sequence(cur_n, exp)
            # 使用 round 或 ceil 通常比 int(向下取整) 更符合统计分布
            sequence = [max(1, int(round(s))) for s in sequence]

            # 3. 修正总度数为偶数 (配置模型必备)
            if sum(sequence) % 2 != 0:
                sequence[0] += 1

            g = nx.configuration_model(sequence)
            # 将配置模型转换为简单图
            g = nx.Graph(g)
            g.remove_edges_from(nx.selfloop_edges(g))
            if nx.is_connected(g):
                break
            else:
                G_giant = max(nx.connected_components(g), key=len)
                if len(G_giant) > num_min:
                    g = g.subgraph(G_giant).copy()
                    break

    elif g_type == "K_Regular":
        while True:
            k_regular = random.randint(round(0.2 * cur_n), cur_n - 2)
            if k_regular * cur_n % 2 == 0 and cur_n >= (k_regular + 1):
                # NetworkX 的随机正则图生成函数
                g = nx.random_regular_graph(d=k_regular, n=cur_n)
                if nx.is_connected(g):
                    break
                else:
                    G_giant=max(nx.connected_components(g),key=len)
                    if len(G_giant)>num_min:
                        g=g.subgraph(G_giant).copy()
                        break
    elif g_type=="SBM":
        while True:
            g=generate_random_sbm(cur_n)
         #   print("generate g",g)
            if nx.is_connected(g):
                break
            else:
                G_giant = max(nx.connected_components(g), key=len)
                if len(G_giant) > num_min:
                    g = g.subgraph(G_giant).copy()
                  #  print("g giant",g)
                    break
    return g
if __name__ =="__main__":

    g_type_list=["static_power_law","barabasi_albert","small-world","erdos_renyi"]#"static_power_law","barabasi_albert","small-world","erdos_renyi",
    directory="dataset/dataset_synth_size/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    # for g_type in g_type_list:
    #     print(g_type)
    #
    #     for i in range(100):
    #         G = gen_graph(g_type=g_type, num_min=500, num_max=5000)
    #         file_path=directory+"/"+g_type+"_"+str(i)+".graphml"
    #         try:
    #             with open(file_path, 'wb') as f:
    #                 nx.write_graphml(G, f)
    #             # print(f"成功保存: {file_path}")
    #         except Exception as e:
    #             print(f"保存文件 {file_path} 时出错: {e}")
    for g_type in g_type_list:
        print(g_type)

        for i in range(100):
            G = gen_graph(g_type=g_type, num_min=500, num_max=5000)
            file_path=directory+"/"+g_type+"_"+str(i)+".graphml"
            try:
                with open(file_path, 'wb') as f:
                    nx.write_graphml(G, f)
                # print(f"成功保存: {file_path}")
            except Exception as e:
                print(f"保存文件 {file_path} 时出错: {e}")

