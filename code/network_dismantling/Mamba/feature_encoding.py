"""
节点特征编码模块
将图转换为有序序列，提取拓扑特征并归一化
"""
import numpy as np
import networkx as nx
from collections import deque
from typing import Tuple, List

# 支持的节点序列排序方式。
# 关键性质：排序**只影响 Mamba 分支**——邻接矩阵与特征会跟着一起重排，
# 而 GCN 对节点置换等变，所以换排序不会改变 GCN 的输出。这使它成为一个
# 干净的「Mamba 看到什么顺序」的单变量实验旋钮。
ORDERS = ("degree", "bfs", "dfs", "core")


def _bfs_order(G: nx.Graph) -> List[int]:
    """
    层序遍历：从每个连通分量中度最大的节点出发做 BFS。

    注意实测性质：BFS 出队的相邻两个节点是**同层兄弟**（共享父节点），
    彼此通常**不是**图邻居——BA-300 上序列相邻且在图中有边的比例只有 2.0%，
    反而低于度序的 3.7%。它的语义是「按到 hub 的跳数分层」，而非「沿路径走」。
    想要相邻即邻居请用 'dfs'。

    邻居按度降序入队、分量按大小降序处理，保证结果确定（可复现）。
    """
    order: List[int] = []
    seen = set()
    for comp in sorted(nx.connected_components(G), key=len, reverse=True):
        remaining = comp - seen
        if not remaining:
            continue
        start = max(remaining, key=lambda v: (G.degree(v), -v))
        queue = deque([start])
        seen.add(start)
        while queue:
            u = queue.popleft()
            order.append(u)
            for w in sorted(G.neighbors(u), key=lambda v: (-G.degree(v), v)):
                if w not in seen:
                    seen.add(w)
                    queue.append(w)
    return order


def _dfs_order(G: nx.Graph) -> List[int]:
    """
    深度优先遍历序：沿图上的路径走，**序列相邻的两个节点大概率是图邻居**。

    这是与「按度降序」最本质的差别：度排序下相邻节点在图里可能毫不相关，
    Mamba 沿序列积累的隐状态没有结构含义；DFS 序下隐状态相当于沿一条图路径
    累积上下文（与 random-walk 序列建模的思路一致）。

    邻居按度**升序**压栈 -> 出栈时度大的先被访问（优先走向 hub），
    保证确定性与「先探主干」的直觉。
    """
    order: List[int] = []
    seen = set()
    for comp in sorted(nx.connected_components(G), key=len, reverse=True):
        remaining = comp - seen
        if not remaining:
            continue
        start = max(remaining, key=lambda v: (G.degree(v), -v))
        stack = [start]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            order.append(u)
            for w in sorted(G.neighbors(u), key=lambda v: (G.degree(v), v)):
                if w not in seen:
                    stack.append(w)
    return order


def _core_order(G: nx.Graph) -> List[int]:
    """
    k-core 剥壳序：按 core number 降序排，同 core 内按度降序。

    动机：网络瓦解的骨架正是 k-core，而 core number 是**全局**量，GCN 的局部
    消息传递难以推断。把它编码进序列顺序，Mamba 沿序列就能看到「谁会先被剥掉」
    的全局分层信息。
    """
    core_dict = nx.core_number(G)
    node_list = list(G.nodes())
    degrees = np.array([G.degree(v) for v in node_list], dtype=np.float64)
    cores = np.array([core_dict[v] for v in node_list], dtype=np.float64)
    # lexsort 的主键在最后：先按 core 降序，再按度降序，最后按位置升序（稳定）
    idx = np.lexsort((np.arange(len(node_list)), -degrees, -cores))
    return [node_list[i] for i in idx]


def extract_node_features(
    G: nx.Graph,
    feature_set: str = "all",
    order: str = "degree",
) -> Tuple[np.ndarray, List[int]]:
    """
    将图转换为有序序列，提取节点拓扑特征

    Parameters
    ----------
    G : nx.Graph
        输入图（节点标签应为 0 到 n-1 的整数）
    feature_set : str
        特征集选择：
        - 'all'    : [度, k-core, PageRank, 接近中心性]（4 维，默认）
        - 'degree' : 仅节点度（1 维，用于消融实验，避免 O(n^2) 中心性计算）
    order : str
        节点序列排序方式（只影响序列顺序，不改变每个节点自己的特征值）：
        - 'degree' : 按度降序（默认，原行为）
        - 'bfs'    : 从最大度节点出发的层序遍历（按到 hub 的跳数分层）
        - 'dfs'    : 深度优先遍历序（沿图路径走，相邻位置多为图邻居）
        - 'core'   : 按 k-core 剥壳序（core number 降序，同 core 内按度降序）

    Returns
    -------
    features : np.ndarray
        归一化特征矩阵，形状 (n_nodes, D)，D 由 feature_set 决定（1 或 4）
        每行对应一个节点的特征（顺序与 node_ids 一致）
    node_ids : List[int]
        对应的节点 ID 列表，顺序由 order 决定

    Notes
    -----
    不要求节点标签是 0..n-1：内部一律按实际节点列表索引。取 LCC 后标签会不连续，
    若按 range(n) 索引会静默取到不存在的节点。
    """
    n = G.number_of_nodes()

    n_feat = 1 if feature_set == "degree" else 4
    if n == 0:
        return np.zeros((0, n_feat)), []

    # 一切按**节点**索引，不假定标签是 0..n-1。
    # 这个假定原先潜伏在实现里（用 range(n) 索引度数），取 LCC 后的图标签不连续
    # （如 ER-300 取 LCC 后剩 299 个节点但标签到 299），会让度数静默取到不存在
    # 的节点、值为 0；换成返回标签的 bfs/dfs 排序则直接越界崩溃。
    node_list = list(G.nodes())
    label_to_pos = {v: i for i, v in enumerate(node_list)}
    degree_dict = dict(G.degree())
    degrees = np.array([degree_dict[v] for v in node_list], dtype=np.float32)

    if feature_set == "degree":
        # 仅度特征，跳过中心性计算（O(m) 复杂度）
        features = degrees.reshape(-1, 1)
    else:
        # 2. k-core 值
        core_dict = nx.core_number(G)
        cores = np.array([core_dict[v] for v in node_list], dtype=np.float32)

        # 3. PageRank 值
        try:
            pagerank_dict = nx.pagerank(G, max_iter=100)
            pageranks = np.array([pagerank_dict[v] for v in node_list], dtype=np.float32)
        except Exception:
            # 如果 PageRank 计算失败，使用度中心性作为替代
            pageranks = degrees / (degrees.sum() + 1e-8)

        # 4. 接近中心性
        try:
            # 对于大图，接近中心性计算可能很慢，这里只对连通分量计算
            if nx.is_connected(G):
                closeness_dict = nx.closeness_centrality(G)
                closeness = np.array([closeness_dict[v] for v in node_list], dtype=np.float32)
            else:
                # 对于非连通图，分别计算各连通分量的接近中心性
                closeness = np.zeros(n, dtype=np.float32)
                for component in nx.connected_components(G):
                    if len(component) > 1:
                        subgraph = G.subgraph(component)
                        closeness_dict = nx.closeness_centrality(subgraph)
                        for node in component:
                            closeness[label_to_pos[node]] = closeness_dict[node]
        except Exception:
            # 如果计算失败，使用度作为替代
            closeness = degrees / (degrees.max() + 1e-8)

        # 构建特征矩阵
        features = np.stack([degrees, cores, pageranks, closeness], axis=1)

    # 按 order 指定的方式排列节点序列。
    # 'degree' 用稳定排序（kind='stable'）保证同度节点的相对顺序确定为原始节点
    # 出现顺序：WS/ER 等近正则图上同度节点极多，非稳定排序会给出任意顺序，而
    # Mamba 对输入序列顺序敏感，等于往输入里注入噪声。
    if order == "degree":
        order_pos = np.argsort(-degrees, kind="stable")
        order_labels = [node_list[p] for p in order_pos]
    else:
        if order == "bfs":
            order_labels = _bfs_order(G)
        elif order == "dfs":
            order_labels = _dfs_order(G)
        elif order == "core":
            order_labels = _core_order(G)
        else:
            raise ValueError(f"未知排序 '{order}'，可选 {ORDERS}")
        try:
            order_pos = np.asarray([label_to_pos[v] for v in order_labels], dtype=np.int64)
        except KeyError as e:
            raise ValueError(f"排序 '{order}' 返回了不属于该图的节点 {e}") from e

    if len(order_pos) != n:
        raise ValueError(f"排序 '{order}' 返回 {len(order_pos)} 个节点，但图有 {n} 个")

    node_ids = [node_list[int(p)] for p in order_pos]
    features = features[order_pos]

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
