"""
节点特征编码模块
将图转换为有序序列，提取拓扑特征并归一化
"""
import numpy as np
import networkx as nx
from typing import Tuple, List


def extract_node_features(G: nx.Graph) -> Tuple[np.ndarray, List[int]]:
    """
    将图转换为有序序列，提取节点拓扑特征

    Parameters
    ----------
    G : nx.Graph
        输入图（节点标签应为 0 到 n-1 的整数）

    Returns
    -------
    features : np.ndarray
        归一化特征矩阵，形状 (n_nodes, 4)
        每行对应一个节点的 4 项特征：[度, k-core, PageRank, 接近中心性]
    node_ids : List[int]
        对应的节点 ID 列表（按度降序排列）
    """
    n = G.number_of_nodes()

    if n == 0:
        return np.zeros((0, 4)), []

    # 提取特征
    # 1. 节点度
    degree_dict = dict(G.degree())
    degrees = np.array([degree_dict.get(i, 0) for i in range(n)], dtype=np.float32)

    # 2. k-core 值
    core_dict = nx.core_number(G)
    cores = np.array([core_dict.get(i, 0) for i in range(n)], dtype=np.float32)

    # 3. PageRank 值
    try:
        pagerank_dict = nx.pagerank(G, max_iter=100)
        pageranks = np.array([pagerank_dict.get(i, 0) for i in range(n)], dtype=np.float32)
    except:
        # 如果 PageRank 计算失败，使用度中心性作为替代
        pageranks = degrees / (degrees.sum() + 1e-8)

    # 4. 接近中心性
    try:
        # 对于大图，接近中心性计算可能很慢，这里只对连通分量计算
        if nx.is_connected(G):
            closeness_dict = nx.closeness_centrality(G)
            closeness = np.array([closeness_dict.get(i, 0) for i in range(n)], dtype=np.float32)
        else:
            # 对于非连通图，分别计算各连通分量的接近中心性
            closeness = np.zeros(n, dtype=np.float32)
            for component in nx.connected_components(G):
                if len(component) > 1:
                    subgraph = G.subgraph(component)
                    closeness_dict = nx.closeness_centrality(subgraph)
                    for node in component:
                        closeness[node] = closeness_dict[node]
    except:
        # 如果计算失败，使用度作为替代
        closeness = degrees / (degrees.max() + 1e-8)

    # 构建特征矩阵
    features = np.stack([degrees, cores, pageranks, closeness], axis=1)

    # 按度降序排列
    sorted_indices = np.argsort(degrees)[::-1]
    node_ids = sorted_indices.tolist()
    features = features[sorted_indices]

    # Min-Max 归一化每个特征
    features_normalized = np.zeros_like(features)
    for i in range(features.shape[1]):
        col = features[:, i]
        col_min = col.min()
        col_max = col.max()
        if col_max - col_min > 1e-8:
            features_normalized[:, i] = (col - col_min) / (col_max - col_min)
        else:
            # 如果特征值全部相同，归一化为 0
            features_normalized[:, i] = 0.0

    return features_normalized, node_ids
