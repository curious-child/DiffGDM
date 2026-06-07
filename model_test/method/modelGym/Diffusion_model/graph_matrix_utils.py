import copy
import os
import time

import networkx as nx
import torch
import matplotlib.pyplot as plt
import torch_geometric as tg
import torch_sparse
import numpy as np


def my_all_shortest_paths(G, source, target):
    pred = nx.predecessor(G, source)
    if target not in pred:
        raise nx.NetworkXNoPath(
            f"Target {target} cannot be reached" f"from given sources"
        )
    sources = {source}
    seen = {target}
    stack = [[target, 0]]
    top = 0
    while top >= 0:
        node, i = stack[top]
        if node in sources:
            yield [p for p, n in reversed(stack[: top + 1])]
        if len(pred[node]) > i:
            stack[top][1] = i + 1
            next = pred[node][i]
            if next in seen:
                continue
            else:
                seen.add(next)
            top += 1
            if top == len(stack):
                stack.append([next, 0])
            else:
                stack[top][:] = [next, 0]
        else:
            seen.discard(node)
            top -= 1


def getSmallestCycles(Mygraph):
    SmallestCycles = set()
    NodeGirth = dict()
    SmallestCyclesOfNodes = {}  #
    CycleRatio = {}
    DEF_IMPOSSLEN = Mygraph.number_of_nodes() + 1
    Coreness = nx.core_number(Mygraph)

    removeNodes = set()
    NodeList = list(Mygraph.nodes())
    for i in Mygraph.nodes():  #
        SmallestCyclesOfNodes[i] = set()
        CycleRatio[i] = 0
        if Mygraph.degree(i) <= 1 or Coreness[i] <= 1:
            NodeGirth[i] = 0
            removeNodes.add(i)
        else:
            NodeGirth[i] = DEF_IMPOSSLEN
    NodeList.sort()
    #setp 1
    curCyc = list()
    for ix in NodeList[:-2]:  #v1
        if NodeGirth[ix] == 0:
            continue
        curCyc.append(ix)
        for jx in NodeList[NodeList.index(ix) + 1 : -1]:  #v2
            if NodeGirth[jx] == 0:
                continue
            curCyc.append(jx)
            if Mygraph.has_edge(ix,jx):
                for kx in NodeList[NodeList.index(jx) + 1:]:      #v3
                    if NodeGirth[kx] == 0:
                        continue
                    if Mygraph.has_edge(kx,ix):
                        curCyc.append(kx)
                        if Mygraph.has_edge(kx,jx):
                            SmallestCycles.add(tuple(curCyc))
                            for i in curCyc:
                                NodeGirth[i] = 3
                        curCyc.pop()
            curCyc.pop()
        curCyc.pop()
    # setp 2
    ResiNodeList = []  # Residual Node List
    for nod in NodeList:
        if NodeGirth[nod] == DEF_IMPOSSLEN:
            ResiNodeList.append(nod)
    if len(ResiNodeList) == 0:
        return SmallestCycles
    else:
        visitedNodes = dict.fromkeys(ResiNodeList,set())
        for nod in ResiNodeList:
            if Coreness[nod] == 2 and NodeGirth[nod] < DEF_IMPOSSLEN:
                continue
            for nei in list(Mygraph.neighbors(nod)):
                if Coreness[nei] == 2 and NodeGirth[nei] < DEF_IMPOSSLEN:
                    continue
                if not nei in visitedNodes.keys() or not nod in visitedNodes[nei]:
                    visitedNodes[nod].add(nei)
                    if nei not in visitedNodes.keys():
                        visitedNodes[nei] = set([nod])
                    else:
                        visitedNodes[nei].add(nod)
                    if Coreness[nei] == 2 and NodeGirth[nei] < DEF_IMPOSSLEN:
                        continue
                    Mygraph.remove_edge(nod, nei)
                    if nx.has_path(Mygraph, nod, nei):
                        for path in my_all_shortest_paths(Mygraph, nod, nei):
                            lenPath = len(path)
                            path.sort()
                            SmallestCycles.add(tuple(path))
                            for i in path:
                                if NodeGirth[i] > lenPath:
                                    NodeGirth[i] = lenPath
                    Mygraph.add_edge(nod, nei)
    return SmallestCycles
def Non_Backtracking_matrix(orG):
    G = copy.deepcopy(orG).to_directed()

    # 首先，获取边列表
    edges = list(sorted(G.edges()))
    n = len(edges)
    edge_index = {e: i for i, e in enumerate(edges)}

    # 初始化非回溯矩阵
    row_indices = []
    col_indices = []
    values = []

    # 填充非回溯矩阵
    for u, v in edges:
        for v2 in G.neighbors(v):
            if v2 != u:  # 确保没有回溯
                if (v, v2) in edge_index:
                    row_indices.append(edge_index[(u, v)])
                    col_indices.append(edge_index[(v, v2)])
                    values.append(1)

    # 创建稀疏矩阵
    B_sparse = torch.sparse_coo_tensor(
        indices=torch.tensor([row_indices, col_indices]),
        values=torch.tensor(values),
        size=(n, n),
        dtype=torch.int
    )

    return B_sparse
def cycle_count_matrix(G):
    # 计算图的圈基
    cycles = getSmallestCycles(G)


    # 初始化圈数矩阵
    num_nodes = G.number_of_nodes()
    cycle_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.int)

    # 建立节点索引
    node_index = {node: idx for idx, node in enumerate(G.nodes())}

    # 计算每个节点和节点对经过的圈的数量
    for cycle in cycles:
        for i in range(len(cycle)):
            u = cycle[i]
            idx_u = node_index[u]

            # 更新经过节点的圈数目
            cycle_matrix[idx_u, idx_u] += 1

            for j in range(i + 1, len(cycle)):
                v = cycle[j]
                idx_v = node_index[v]

                # 更新经过节点对的圈数目
                cycle_matrix[idx_u, idx_v] += 1
                cycle_matrix[idx_v, idx_u] += 1

    return cycle_matrix
def normalize_cycle_matrix(cycle_matrix):
    # 归一化圈数矩阵
    num_nodes = cycle_matrix.size(0)
    for i in range(num_nodes):
        diag_value = float(cycle_matrix[i, i].item())
        if diag_value > 0:
            cycle_matrix[i] = cycle_matrix[i] / diag_value
        else:
            cycle_matrix[i] = torch.zeros(num_nodes, dtype=torch.float)  # 如果对角线元素为0，整行设为0

    # 转换为稀疏矩阵
    indices = torch.nonzero(cycle_matrix, as_tuple=False).t()
    values = cycle_matrix[indices[0], indices[1]]

    sparse_matrix = torch.sparse_coo_tensor(indices, values, cycle_matrix.size())

    return sparse_matrix

#邻接矩阵特征值的分散程度 自然连通性衡量 不可微
def natural_connectivity_sparse(tensor_G,node_v,device="cuda"):
    """
    使用稀疏矩阵计算自然连通性，适合处理大规模图。
    """
    from scipy.sparse.linalg import eigsh
    # 获取拉普拉斯稀疏矩阵
    from scipy.sparse import coo_matrix
    adj_matrix = torch.sparse_coo_tensor(indices=tensor_G.coalesce().edge_index,
                                         values=torch.ones(tensor_G.edge_index.size(1)).to(device),
                                         size=(tensor_G.num_nodes, tensor_G.num_nodes), dtype=torch.float).to(device)
    node_v=node_v.to("cpu")#取值范围为0-1
    sparse_diag_node_v = torch.sparse.spdiags(node_v, torch.tensor([0]), (node_v.size(0), node_v.size(0))).to(
        torch.float).to(device)

    adj_result = torch.sparse.mm(adj_matrix, sparse_diag_node_v)
    adj_result = torch.sparse.mm(sparse_diag_node_v, adj_result)
    tensor_G.weight = torch.round(adj_result.values())# 四舍五入取整
    #tensor_G.weight = torch.bernoulli(adj_result.values().clamp(0, 1))# 伯努利分布取值
    net_G = tg.utils.to_networkx(tensor_G,edge_attrs=["weight"] ,to_undirected=True)
    estrada_index = nx.estrada_index(net_G)

    # 自然连通性
    n = tensor_G.num_nodes
    natural_conn = np.log(estrada_index / n)

    return torch.tensor(natural_conn).to(device)

###与矩阵某特征值大小相关的网络鲁棒性指标 可微化###
def target_eigenvalue_guided(matrix_type,graph_name,edge_index,node_v,device,graph_root_path = "analysis/analysis_dataset/dataset_real"):
    node_v=1-node_v#每个节点不删除的概率
    tensor_G = tg.data.Data(x=node_v,edge_index=edge_index).to(device)
    # graph matrix : non_backtracking, cycle_radio, laplacian,adjacency
    # graph_name (data type):sparse tensor
    graph_root_path=None
    if matrix_type=="non_backtracking":
        if graph_root_path is not None:
            graph_save_path = os.path.join(graph_root_path ,matrix_type)
            if not os.path.exists(graph_save_path):
                os.mkdir(graph_save_path)
            graph_path = graph_save_path + "/{}.pt".format( graph_name)
            if os.path.exists(graph_path):
                graph_matrix = torch.load(graph_path).to(device)
            else:
                print("Creating non_backtracking matrix of networx {}".format(graph_name))
                net_G=tg.utils.to_networkx(tensor_G,to_undirected=True)
                graph_matrix = Non_Backtracking_matrix(net_G).to(device)
                torch.save(graph_matrix, graph_path)
        else:

            net_G = tg.utils.to_networkx(tensor_G, to_undirected=True)
            graph_matrix = Non_Backtracking_matrix(net_G).to(device)

        index_node_v_extend = tensor_G.coalesce().edge_index[0]
        ex_node_v = torch.gather(node_v, 0, index_node_v_extend)

        col_indices = graph_matrix._indices()[1]  # 列索引
        values = graph_matrix._values()  # 非零元素的值
        cp_values = values * ex_node_v[col_indices]
        cp_matrix = torch.sparse_coo_tensor(graph_matrix._indices(), cp_values, graph_matrix.size())
    elif matrix_type=="cycle_radio":
        if graph_root_path is not None:
            graph_save_path = os.path.join(graph_root_path, matrix_type)
            if not os.path.exists(graph_save_path):
                os.mkdir(graph_save_path)
            graph_path = graph_save_path + "/{}.pt".format(graph_name)
            if os.path.exists(graph_path):
                graph_matrix = torch.load(graph_path).to(device)
            else:
                print("Creating cycle_radio matrix of networx {}".format(graph_name))
                net_G = tg.utils.to_networkx(tensor_G, to_undirected=True)
                cycle_matrix = cycle_count_matrix(net_G)
                graph_matrix = normalize_cycle_matrix(cycle_matrix.float()).to(device)
                torch.save(graph_matrix, graph_path)
        else:
            net_G = tg.utils.to_networkx(tensor_G, to_undirected=True)
            cycle_matrix = cycle_count_matrix(net_G)
            graph_matrix = normalize_cycle_matrix(cycle_matrix.float()).to(device)
        row_indices = graph_matrix._indices()[0]  # 行索引
        col_indices = graph_matrix._indices()[1]  # 列索引
        values = graph_matrix._values()  # 非零元素的值
        cp_values = values * node_v[col_indices]*node_v[row_indices]
        cp_matrix = torch.sparse_coo_tensor(graph_matrix._indices(), cp_values, graph_matrix.size())
    elif matrix_type=="laplacian":
        adj_matrix = torch.sparse_coo_tensor(indices=tensor_G.coalesce().edge_index,
                                             values=torch.ones(tensor_G.edge_index.size(1)).to(device),
                                             size=(tensor_G.num_nodes, tensor_G.num_nodes), dtype=torch.float)

        row_indices = adj_matrix._indices()[0]  # 行索引
        col_indices = adj_matrix._indices()[1]  # 列索引
        values = adj_matrix._values()  # 非零元素的值
        cp_values = values * node_v[col_indices] * node_v[row_indices]
        adj_result = torch.sparse_coo_tensor(adj_matrix._indices(), cp_values, adj_matrix.size())

        laplacian_matrix = tg.utils.get_laplacian(edge_index=adj_result.coalesce().indices(),
                                                  edge_weight=adj_result.coalesce().values(),
                                                  normalization='sym',
                                                  num_nodes=tensor_G.num_nodes)

        cp_matrix = torch.sparse_coo_tensor(indices=laplacian_matrix[0],
                                                   values=laplacian_matrix[1],
                                                   size=(tensor_G.num_nodes, tensor_G.num_nodes), dtype=torch.float).to(device)
    elif matrix_type=="adjacency":
        adj_matrix = torch.sparse_coo_tensor(indices=tensor_G.coalesce().edge_index,
                                             values=torch.ones(tensor_G.edge_index.size(1)).to(device),
                                             size=(tensor_G.num_nodes, tensor_G.num_nodes), dtype=torch.float)
        row_indices = adj_matrix._indices()[0]  # 行索引
        col_indices = adj_matrix._indices()[1]  # 列索引
        values = adj_matrix._values()  # 非零元素的值
        cp_values = values * node_v[col_indices] * node_v[row_indices]
        cp_matrix = torch.sparse_coo_tensor(adj_matrix._indices(), cp_values, adj_matrix.size())

    else:
        raise ValueError("Invalid matrix type \n"
                         "Please choose from 'non_backtracking', 'cycle_radio', 'laplacian', 'adjacency'")
    #
    # 直接计算 Frobenius 范数 作为目标函数
    values = cp_matrix.coalesce().values()

    frobenius_norm = torch.sqrt(torch.sum(values ** 2)).to(device)
    return frobenius_norm
###与矩阵特征值的分散程度相关的网络鲁棒性指标  可微化？？###
#拉普拉斯矩阵特征值的分散程度 R-entropy 可不可微 待测试
def WR_energy_guided(edge_index,node_v,device):
    node_v = 1 - node_v  # 每个节点不删除的概率
    tensor_G = tg.data.Data(x=node_v, edge_index=edge_index).to(device)
    adj_matrix = torch.sparse_coo_tensor(indices=tensor_G.coalesce().edge_index,
                                        values=torch.ones(tensor_G.edge_index.size(1)).to(device),
                                        size=(tensor_G.num_nodes,tensor_G.num_nodes), dtype=torch.float).to(device)

    row_indices = adj_matrix._indices()[0]  # 行索引
    col_indices = adj_matrix._indices()[1]  # 列索引
    values = adj_matrix._values()  # 非零元素的值
    cp_values = values * node_v[col_indices] * node_v[row_indices]
    adj_result = torch.sparse_coo_tensor(adj_matrix._indices(), cp_values, adj_matrix.size())

    # 获取图的基本信息
    edge_index = adj_result.coalesce().indices() # 边索引矩阵 [2, num_edges]
    edge_weight = adj_result.coalesce().values()  # 边权重 [num_edges, 1]
    num_nodes = tensor_G.num_nodes  # 节点数 n
    num_edges = tensor_G.num_edges  # 边数

    # 确保边权重存在
    if edge_weight is None:
        raise ValueError("边权重 (edge_attr) 不能为空！请提供加权网络。")


    #  计算每个节点的加权度数 deg(v_i)
    deg=adj_matrix.sum(dim=1).to_dense()#邻接矩阵的第i行的和，即节点i的度数


    # 计算每条边的贡献 (w_ij ** 2) / (deg[vi] * deg[vj])
    vi = edge_index[0]  # 起始节点 vi
    vj = edge_index[1]  # 目标节点 vj
    e_terms = (edge_weight ** 2) / (deg[vi] * deg[vj])
    e = e_terms.sum()  # 所有边的贡献之和
    # Step 3: 归一化 WR-energy
    e = e / (num_nodes - 1) - num_nodes / ((num_nodes - 1) ** 2)

    # 返回 WR-energy 值
    return e

if __name__ == '__main__':
    import math
    # 读取 .graphml 文件
    # G = nx.read_graphml('your_graph.graphml')
    nodes_list=[1,2,3,4,5,6]
    edges_list=[(1,2),(2,3),(2,5),(3,4),(3,5),(5,6)]
    G = nx.Graph()
    G.add_nodes_from(nodes_list)
    G.add_edges_from(edges_list)
    # file = "dataset/dataset_real/cfinder-google.graphml"
    # G = nx.read_graphml(file)
    # print(G)
    # lcc = max(nx.connected_components(G), key=len)
    # lcc_size = len(lcc)
    # print("lcc size:", lcc_size)
    # start_time = time.time()
 #trada index time:", time.time() - start_time)
   # print("adjacency matrix:", nx.adjacency_matrix(G))
    # # 绘制图
    # plt.figure(figsize=(10, 10))
    # nx.draw(G, with_labels=True,ax=plt.gca())
    # plt.show()
    # 计算非回溯矩阵

    B = Non_Backtracking_matrix(G)
    print("非回溯矩阵:",B.to_dense())
    node_v=torch.arange(1,G.number_of_nodes()+1)
    tensor_G=tg.utils.from_networkx(G)
    print("节点向量:",node_v)
    index_node_v_extend=tensor_G.coalesce().edge_index[0]
    print("图数据:",tensor_G.coalesce().edge_index)
    ex_node_v=torch.gather(node_v,0,index_node_v_extend)
    print("拓展后的图数据:",ex_node_v)
    #对向量进行操作，得到稀疏对角化矩阵
    row_indices = B._indices()[0]  # 行索引
    col_indices = B._indices()[1]  # 列索引
    values = B._values()  # 非零元素的值
    cp_values=values*ex_node_v[col_indices]*ex_node_v[row_indices]
    #
    cp_result=torch.sparse_coo_tensor(B._indices(),cp_values,B.size())
    print("稀疏矩阵乘以稀疏对角化矩阵:",cp_result.to_dense())

    #稀疏化邻接矩阵
  #   tensor_G = tg.utils.from_networkx(G)
  #   node_v=0.9*torch.randn(G.number_of_nodes()).to(torch.float)-0.9
  #   #node_v = 0.01*torch.ones(G.number_of_nodes()).to(torch.float)
  #   net_G = tg.utils.to_networkx(tensor_G, to_undirected=True)
  #   adj_matrix = torch.sparse_coo_tensor(indices=tensor_G.coalesce().edge_index,
  #                                        values=torch.ones(tensor_G.edge_index.size(1)),
  #                                        size=(tensor_G.num_nodes, tensor_G.num_nodes), dtype=torch.float)
  #   node_v = node_v.to("cpu")
  #   sparse_diag_node_v = torch.sparse.spdiags(node_v, torch.tensor([0]), (node_v.size(0), node_v.size(0))).to(
  #       torch.float)
  #
  #   adj_result = torch.sparse.mm(adj_matrix, sparse_diag_node_v)
  #   adj_result = torch.sparse.mm(sparse_diag_node_v, adj_result)
  #
  #   print(adj_result.values())
  #   mask=torch.round(adj_result.values()).to(torch.bool)
  # #  mask = tensor_G.weight > 0
  #   filtered_edge_index = tensor_G.edge_index[:, mask]
  #   filtered_edge_index = tg.utils.to_undirected(filtered_edge_index)
  #   tensor_G.edge_index = filtered_edge_index
  #
  #   net_G = tg.utils.to_networkx(tensor_G,  to_undirected=True)
  #   print(net_G)
  #   lcc = max(nx.connected_components(net_G), key=len)
  #   lcc_size = len(lcc)
  #   print("lcc size:", lcc_size)
    # start_time = time.time()
    # print("estrada index:",nx.estrada_index(net_G))
    # print("estrada index time:", time.time() - start_time)
    # # 计算自然连通性
    # start_time = time.time()
    # subgraph_centrality_exp = nx.subgraph_centrality_exp(net_G)
    # natural_conn = 0
    # for node, centrality in subgraph_centrality_exp.items():
    #     natural_conn +=centrality
    # natural_conn = natural_conn/len(subgraph_centrality_exp)
    # natural_conn=math.log(natural_conn)
    # print("自然连通性 计算时间", time.time() - start_time)
    # nar = natural_connectivity_sparse(tensor_G,node_v,device="cpu")
    # print("自然连通性:", nar)
    # print("自然连通性 计算时间", time.time() - start_time)

    # adj_matrix=torch.sparse_coo_tensor(indices=tensor_G.coalesce().edge_index,
    #                                    values=torch.ones(tensor_G.edge_index.size(1)),
    #                                    size=(G.number_of_nodes(),G.number_of_nodes()),dtype=torch.float)
    # print("邻接矩阵:",adj_matrix)
    # print("邻接矩阵:",adj_matrix.to_dense())
    # sparse_diag_node_v=torch.sparse.spdiags(node_v,torch.tensor([0]),(node_v.size(0),node_v.size(0))).to(torch.float)
    # adj_result=torch.sparse.mm(adj_matrix,sparse_diag_node_v)
    # adj_result=torch.sparse.mm(sparse_diag_node_v,adj_result)
    # print("邻接矩阵乘以节点向量:",adj_result)
    # print("邻接矩阵乘以节点向量:",adj_result.to_dense())
    # 计算图的拉普拉斯矩阵
    # laplacian_matrix = tg.utils.get_laplacian(edge_index=adj_result.indices(),
    #                                           num_nodes=tensor_G.num_nodes)
    # print("拉普拉斯矩阵:", laplacian_matrix)
    # laplacian_matrix = torch.sparse_coo_tensor(indices=laplacian_matrix[0],
    #                                            values=laplacian_matrix[1],
    #                                            size=(tensor_G.num_nodes, tensor_G.num_nodes), dtype=torch.float)
    # print("拉普拉斯矩阵:", laplacian_matrix.to_dense())
    #计算图的拉普拉斯矩阵（考虑节点向量）
    # laplacian_matrix=tg.utils.get_laplacian(edge_index=adj_result.indices(),
    #                                         edge_weight=adj_result.values(),
    #                                         normalization='sym',
    #                                         num_nodes=tensor_G.num_nodes)
    # print("拉普拉斯矩阵:",laplacian_matrix)
    # laplacian_matrix=torch.sparse_coo_tensor(indices=laplacian_matrix[0],
    #                                           values=laplacian_matrix[1],
    #                                           size=(tensor_G.num_nodes,tensor_G.num_nodes),dtype=torch.float)
    # print("拉普拉斯矩阵:",laplacian_matrix)
    # print("拉普拉斯矩阵:",laplacian_matrix.coalesce())
    #
    # # 直接计算 Frobenius 范数 作为目标函数
    # values = laplacian_matrix.coalesce().values()
    # print(values)
    # frobenius_norm = torch.sqrt(torch.sum(values ** 2))

    # plt.figure(figsize=(10, 10))
    # nx.draw(G, with_labels=True, ax=plt.gca())
    # plt.show()
    # cycle_matrix = cycle_count_matrix(G)
    #
    # print("圈数矩阵:",cycle_matrix)
    # # 归一化为圈比矩阵
    # sparse_cycle_ratio_matrix = normalize_cycle_matrix(cycle_matrix.float())
    #
    #
    # # 输出稀疏矩阵
    # print(sparse_cycle_ratio_matrix)
    # print(sparse_cycle_ratio_matrix.to_dense())