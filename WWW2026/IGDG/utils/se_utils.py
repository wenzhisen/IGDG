import math
import random
import time
import torch
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs

def VNGE_ESSEN(edges, num_nodes):
    """
        基于有向边列表直接计算 VNGE_FINGER 指标。
        :param edges: Tensor，形状为 (2, 边的数量)，表示图的边 (row 表示起点，col 表示终点)
        :param num_nodes: int，图中节点的数量
        """
    import time
    import numpy as np
    from scipy.sparse.linalg import LinearOperator, eigsh

    start = time.time()

    # 提取起点和终点
    row, col = edges[0].numpy(), edges[1].numpy()

    # 计算入度和出度
    in_degrees = np.bincount(col, minlength=num_nodes)  # 入度
    out_degrees = np.bincount(row, minlength=num_nodes)  # 出度
    entropy = ((1 - (1/num_nodes) - (1/(edges.size(1)**2 * in_degrees * out_degrees)))/ edges.size(1) )
    return entropy

def VNGE_FINGER_directed(edges, num_nodes):
    """
    基于有向边列表直接计算 VNGE_FINGER 指标。
    :param edges: Tensor，形状为 (2, 边的数量)，表示图的边 (row 表示起点，col 表示终点)
    :param num_nodes: int，图中节点的数量
    """
    import time
    import numpy as np
    from scipy.sparse.linalg import LinearOperator, eigsh

    start = time.time()

    # 提取起点和终点
    row, col = edges[0].numpy(), edges[1].numpy()

    # 计算入度和出度
    in_degrees = np.bincount(col, minlength=num_nodes)  # 入度
    out_degrees = np.bincount(row, minlength=num_nodes)  # 出度

    # 总度数，用于归一化因子 c
    total_degrees = np.sum(in_degrees + out_degrees)
    c = 1.0 / total_degrees

    # 边权重平方和（权重默认为1）
    edge_weights_squared_sum = len(row)

    # 节点度平方和（入度和出度平方的总和）
    degrees_squared_sum = np.sum(in_degrees**2 + out_degrees**2)

    # 近似值
    approx = 1.0 - c**2 * (degrees_squared_sum + edge_weights_squared_sum)

    # 构建拉普拉斯矩阵的向量乘法（有向图版本）
    def laplacian_matvec(vec):
        """
        计算有向图拉普拉斯矩阵与向量的乘积。
        :param vec: 输入向量
        :return: 拉普拉斯矩阵与输入向量的乘积
        """
        # 贡献来自出度和入度
        result = out_degrees * vec  # 出度部分
        result -= np.bincount(row, weights=vec[col], minlength=num_nodes)  # 邻接关系减去出边的贡献
        result += in_degrees * vec  # 入度部分
        result -= np.bincount(col, weights=vec[row], minlength=num_nodes)  # 邻接关系减去入边的贡献
        return result

    # 使用 SciPy 的 LinearOperator 和 eigsh 求最大特征值
    laplacian_op = LinearOperator((num_nodes, num_nodes), matvec=laplacian_matvec)
    eig_max, _ = eigsh(laplacian_op, 1, which='LM')
    eig_max = eig_max[0] * c

    # 计算冯诺依曼熵
    H_vn = -approx * np.log2(eig_max)

    print('H_vn approx:', H_vn)
    print('Time:', time.time() - start)
    return H_vn

def VNGE_FINGER(edge_index, num_nodes):
    """
    edge_index: shape (2, num_edges), 表示边的索引 (src, dst)
    num_nodes: int，图中的节点数（节点实际数量，编号可能超出范围）。
    """
    start_time = time.time()
    if num_nodes == 0 or edge_index.size(1) <= 1:
        return torch.tensor(0,dtype=torch.float32)
    # 在 PyTorch 中计算 unique 和 remapping
    unique_nodes, remapped_indices = torch.unique(edge_index, return_inverse=True)

    # 检查 unique_nodes 是否超出 num_nodes
    if unique_nodes.numel() > num_nodes:
        num_nodes = unique_nodes.numel()
    # 映射 edge_index 到范围 [0, num_nodes-1]
    remapped_edge_index = remapped_indices.reshape(edge_index.shape)

    # 将边索引转换为稀疏矩阵 A (邻接矩阵)
    row, col = remapped_edge_index[0].cpu().numpy(), remapped_edge_index[1].cpu().numpy()
    data = torch.ones(row.shape[0], device=edge_index.device).cpu().numpy()  # 边的权重，这里假设全为 1
    A = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    # 计算节点度 d
    d = torch.tensor(A.sum(axis=1)).flatten().float()  # 节点度
    c = 1 / torch.sum(d).item()  # 归一化常数
    edge_weight = torch.tensor(A.data).float()  # 提取边权重（全为1）
    GEapprox = 1 - c**2 * (torch.sum(d**2).item() + torch.sum(edge_weight**2).item())

    # 构造对角矩阵 D 和拉普拉斯矩阵 L
    D = coo_matrix((d.cpu().numpy(), (range(num_nodes), range(num_nodes))), shape=(num_nodes, num_nodes))
    L = D - A

    # 检查是否为小矩阵
    if L.shape[0] <= 2:
        eig_vals = torch.tensor(np.linalg.eigvals(L.toarray()))  # 转为稠密矩阵计算特征值
        eig_max = c * torch.max(eig_vals.real).item()  # 最大特征值
    else:
        eig_max = c * eigs(L, k=1, which='LM', return_eigenvectors=False).real

    VNGE = -GEapprox * torch.log(torch.tensor(eig_max).float())
    elapsed_time = time.time() - start_time

    # print('H_vn approx:', VNGE)
    # print('Time:', elapsed_time)
    if math.isfinite(VNGE.item()):
        return torch.tensor(VNGE.item(), dtype=torch.float32)
    else:
        return torch.tensor(0,dtype=torch.float32)


from scipy.sparse.linalg import LinearOperator
def VNGE_exact(edges, num_nodes):
    """
    基于边列表直接计算 VNGE 指标（精确值）。
    :param edges: Tensor，形状为 (2, 边的数量)，表示图的边
    :param num_nodes: int，图中节点的数量
    """
    start = time.time()

    # 计算节点度
    row, col = edges[0].numpy(), edges[1].numpy()
    degrees = np.bincount(np.concatenate([row, col]), minlength=num_nodes)
    lp_size = max(len(degrees),num_nodes)

    # 归一化因子 c
    c = 1.0 / np.sum(degrees)

    # 定义拉普拉斯矩阵的乘法函数
    def laplacian_matvec(vec):
        """
        拉普拉斯矩阵与向量的乘积计算。
        :param vec: 输入向量
        :return: 拉普拉斯矩阵与输入向量的乘积
        """
        result = degrees * vec - np.bincount(row, weights=vec[col], minlength=lp_size)
        result -= np.bincount(col, weights=vec[row], minlength=lp_size)
        return c * result

    # 构造线性算子表示拉普拉斯矩阵
    laplacian_op = LinearOperator((lp_size, lp_size), matvec=laplacian_matvec)

    # 计算所有特征值
    eigenvalues, _ = eigsh(laplacian_op, k=num_nodes - 1, which='SM')  # 求所有非零特征值
    eigenvalues[eigenvalues < 0] = 0  # 修正负数特征值为0

    # 计算熵
    pos = eigenvalues > 0
    H_vn = -np.sum(eigenvalues[pos] * np.log2(eigenvalues[pos]))

    print('H_vn exact:', H_vn)
    print('Time:', time.time() - start)
    return H_vn

def split_graph(edge_index, num_splits):
    """
    将一个图的边张量按照起点排序，并随机分配子图的边数量进行切分。

    参数:
        edge_index (torch.Tensor): 边张量，形状为 (2, num_edges)。
        num_splits (int): 切分的子图数量。

    返回:
        list[torch.Tensor]: 每个元素为子图的边张量，形状为 (2, 子图边的数量)。
    """
    # 按照起点（edge_index[0]）排序
    sorted_indices = torch.argsort(edge_index[0])  # 获取排序的索引
    sorted_edge_index = edge_index[:, sorted_indices]  # 对边张量重新排序

    # 总边数
    num_edges = edge_index.size(1)
    if num_splits > num_edges:
        num_splits = 1
    # 随机生成每个子图的边数量
    random_sizes = [random.randint(1, max(1, num_edges // num_splits * 2)) for _ in range(num_splits)]
    total_random_size = sum(random_sizes)
    random_sizes = [max(1, size * num_edges // total_random_size) for size in random_sizes]

    # 调整最后一个子图的大小以确保边数总和匹配
    random_sizes[-1] += num_edges - sum(random_sizes)

    # 按随机大小切分边张量
    subgraphs = []
    start = 0
    for size in random_sizes:
        end = start + size
        subgraph_edge_index = sorted_edge_index[:, start:end]
        subgraphs.append(subgraph_edge_index)
        start = end

    return subgraphs

def cal_mutual_Information(edge_index, num_nodes, num_splits):
    subgraphs = split_graph(edge_index,num_splits)
    H_list = torch.zeros(num_splits)
    for index, subgraph in enumerate(subgraphs):
        H_list[index] = ( VNGE_FINGER( subgraph, subgraph.size(1) ))
    H_total = VNGE_FINGER(edge_index, num_nodes)
    I_struct = H_list.sum(dim=-1) - H_total

    return I_struct

if __name__ == '__main__':
    num_splits = 1
    tmp = torch.load('mathoverflow')
    x = tmp['x']
    graph = tmp['train']['edge_index_list'][0]
    subgraphs = split_graph(graph,num_splits)
    H_list = torch.zeros(num_splits)
    for index, subgraph in enumerate(subgraphs):
        H_list[index] = ( VNGE_FINGER( subgraph, subgraph.size(1) ))
    H_total = VNGE_FINGER(graph, x.size(0))
    I_struct = H_list.sum(dim=-1) - H_total
    # VNGE_exact(graph, x.size(0))
    # num_nodes = 3000
    # num_edges = 20000
    #
    # # 随机生成无向图的边 (确保边的起点和终点不同且无重复)
    # np.random.seed(int(time.time()))  # 为了结果可复现
    # edges_np = np.random.randint(0, num_nodes, size=(2, num_edges))
    #
    # # 确保无重复边（无向图中的边 (u, v) 与 (v, u) 视为同一条边）
    # edges_np = np.unique(np.sort(edges_np, axis=0), axis=1)
    #
    # # 将边转换为 PyTorch 张量
    # edges = torch.tensor(edges_np, dtype=torch.long)
    # VNGE_exact(edges,num_nodes)
    # VNGE_FINGER(edges,num_nodes)
    print(I_struct)