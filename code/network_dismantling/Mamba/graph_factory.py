"""
多拓扑图工厂（BA / ER / WS）

训练集与评测集的**唯一**图生成入口，保证「训练看到的图」与「评测用的图」
参数口径一致（否则跨拓扑泛化的结论不可信）。

三种拓扑：
    ba : Barabási–Albert 无标度网络，m ∈ [2, 5]（每个新节点连 m 条边）
         度分布重尾，存在 hub —— 拆解任务里 hub 是关键
    er : Erdős–Rényi 随机网络，按**平均度**参数化 p = k_avg/(n-1)，k_avg ∈ [4, 10]
         度分布近泊松，没有 hub
    ws : Watts–Strogatz 小世界网络，k ∈ {4,6,8}（必须偶数）+ beta ∈ [0.1, 0.3]
         近正则图，度分布极窄（n=500 时只有 [4,13]）—— 最难的一种：
         度特征几乎退化，且 CoreHD 的标签质量也接近噪声

连通性：三种图都取最大连通分量（LCC）。ER 在 k_avg 较小时可能产生孤立点，
会重采样若干次尽量拿到 LCC >= 0.95n 的图；取完 LCC 后节点数会变，
所以调用方必须用**返回图的** number_of_nodes() 作为后续指标的分母。
"""
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

# 支持的拓扑
GRAPH_TYPES = ("ba", "er", "ws", "mixed")

# 各拓扑的默认参数范围（与 dataset_generator 的命令行默认值保持一致）
DEFAULT_BA_M_RANGE = (2, 5)
DEFAULT_ER_K_AVG_RANGE = (4.0, 10.0)
DEFAULT_WS_K_CHOICES = (4, 6, 8)
DEFAULT_WS_BETA_RANGE = (0.1, 0.3)

# ER 重采样：尽量让 LCC 覆盖至少这个比例，否则接受当前最好的
_ER_MIN_LCC_FRACTION = 0.95
_ER_MAX_RETRIES = 8


def largest_cc(G: nx.Graph) -> nx.Graph:
    """取最大连通分量（返回副本）"""
    if nx.is_connected(G):
        return G
    biggest = max(nx.connected_components(G), key=len)
    return G.subgraph(biggest).copy()


def build_graph(
    n: int,
    graph_type: str = "ba",
    seed: Optional[int] = None,
    m: int = 3,
    k_avg: float = 6.0,
    k: int = 6,
    beta: float = 0.2,
    keep_largest_cc: bool = True,
) -> nx.Graph:
    """
    构造单张指定拓扑的图。

    Parameters
    ----------
    n : int
        目标节点数（ba/ws 精确；er 取 LCC 后可能略少）
    graph_type : str
        'ba' | 'er' | 'ws'
    seed : int, optional
        随机种子（可复现）
    m : int
        BA 的连边参数
    k_avg : float
        ER 的目标平均度，p = k_avg/(n-1)
    k : int
        WS 的邻居环宽度，必须为偶数
    beta : float
        WS 的重连概率
    keep_largest_cc : bool
        是否取最大连通分量（默认 True）

    Returns
    -------
    G : nx.Graph
    """
    if graph_type not in ("ba", "er", "ws"):
        raise ValueError(f"未知图类型 '{graph_type}'，可选 ba/er/ws")
    if n < 2:
        raise ValueError(f"n 必须 >= 2，当前 {n}")

    if graph_type == "ba":
        G = nx.barabasi_albert_graph(n, max(1, int(m)), seed=seed)

    elif graph_type == "er":
        p = float(k_avg) / (n - 1)
        p = min(max(p, 0.0), 1.0)
        G = _build_er_with_lcc(n, p, seed)

    else:  # ws
        k_even = int(k)
        if k_even % 2 != 0:
            k_even += 1  # WS 要求偶数度环
        k_even = max(2, min(k_even, n - (n % 2)))
        G = nx.watts_strogatz_graph(n, k_even, float(beta), seed=seed)

    if keep_largest_cc and not nx.is_connected(G):
        G = largest_cc(G)
    return G


def _build_er_with_lcc(n: int, p: float, seed: Optional[int]) -> nx.Graph:
    """
    生成 ER 图并尽量拿到大连通分量。

    ER 在小平均度下会产生孤立点/小分量，直接取 LCC 会丢节点。这里重采样若干次，
    优先返回 LCC 占比最高的一张（>= 0.95 即提前返回）。
    """
    rng = np.random.default_rng(seed)
    best_G, best_frac = None, -1.0
    for _ in range(_ER_MAX_RETRIES):
        s = int(rng.integers(0, 2**31))
        G = nx.gnp_random_graph(n, p, seed=s)
        if nx.is_connected(G):
            return G
        frac = len(max(nx.connected_components(G), key=len)) / n
        if frac > best_frac:
            best_G, best_frac = G, frac
        if frac >= _ER_MIN_LCC_FRACTION:
            break
    return best_G


def sample_graph(
    n_range: Tuple[int, int],
    graph_type: str,
    rng: np.random.Generator,
    m_range: Tuple[int, int] = DEFAULT_BA_M_RANGE,
    k_avg_range: Tuple[float, float] = DEFAULT_ER_K_AVG_RANGE,
    ws_k_choices: Tuple[int, ...] = DEFAULT_WS_K_CHOICES,
    ws_beta_range: Tuple[float, float] = DEFAULT_WS_BETA_RANGE,
) -> Tuple[nx.Graph, Dict]:
    """
    在给定范围内随机采样参数并生成一张图（供数据集批量生成使用）。

    参数从传入的 rng 流采样，**调用方应给每种拓扑传独立的 rng**——
    否则改变样本数或拓扑配比会重排所有图的随机种子，破坏「BA 子集与之前
    单独生成的 BA 数据集逐图一致」这种受控对比。

    Returns
    -------
    (G, params) : Tuple[nx.Graph, dict]
        params 含 n_requested / n_actual / m / k_avg / k / beta / seed / graph_type，
        可直接写进数据集 meta 以便复现。
    """
    n = int(rng.integers(n_range[0], n_range[1] + 1))
    seed = int(rng.integers(0, 2**31))
    params = {"graph_type": graph_type, "n_requested": n, "seed": seed}

    if graph_type == "ba":
        m = int(rng.integers(m_range[0], m_range[1] + 1))
        G = build_graph(n, "ba", seed=seed, m=m)
        params["m"] = m

    elif graph_type == "er":
        k_avg = float(rng.uniform(k_avg_range[0], k_avg_range[1]))
        G = build_graph(n, "er", seed=seed, k_avg=k_avg)
        params["k_avg"] = round(k_avg, 4)

    elif graph_type == "ws":
        k = int(rng.choice(list(ws_k_choices)))
        beta = float(rng.uniform(ws_beta_range[0], ws_beta_range[1]))
        G = build_graph(n, "ws", seed=seed, k=k, beta=beta)
        params["k"] = k
        params["beta"] = round(beta, 4)

    else:
        raise ValueError(f"未知图类型 '{graph_type}'")

    params["n_actual"] = G.number_of_nodes()
    params["n_edges"] = G.number_of_edges()
    return G, params


def sample_mixed_graphs(
    num_samples: int,
    topology_weights: Optional[Dict[str, float]] = None,
    n_range: Tuple[int, int] = (500, 1500),
    seed: Optional[int] = None,
    m_range: Tuple[int, int] = DEFAULT_BA_M_RANGE,
    k_avg_range: Tuple[float, float] = DEFAULT_ER_K_AVG_RANGE,
    ws_k_choices: Tuple[int, ...] = DEFAULT_WS_K_CHOICES,
    ws_beta_range: Tuple[float, float] = DEFAULT_WS_BETA_RANGE,
) -> Tuple[List[nx.Graph], List[Dict]]:
    """
    按拓扑配比批量生成混合拓扑图。

    每种拓扑使用**独立的 rng**（由主 seed 派生），保证：
      - 结果可复现
      - 改变配比/总数时，各拓扑已生成的图不变（受控对比）

    Parameters
    ----------
    num_samples : int
        总样本数
    topology_weights : dict, optional
        如 {'ba': 1, 'er': 1, 'ws': 1}，默认三者等权
    n_range : tuple
        节点数范围

    Returns
    -------
    (graphs, params_list)
    """
    if topology_weights is None:
        topology_weights = {"ba": 1.0, "er": 1.0, "ws": 1.0}
    types = [t for t in ("ba", "er", "ws") if topology_weights.get(t, 0) > 0]
    if not types:
        raise ValueError("topology_weights 至少要有一个正权重")

    weights = np.array([float(topology_weights[t]) for t in types], dtype=np.float64)
    weights = weights / weights.sum()

    # 按配比分配样本数（最大余数法，保证总数精确）
    raw = weights * num_samples
    counts = np.floor(raw).astype(int)
    remainder = num_samples - counts.sum()
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for i in range(remainder):
            counts[order[i % len(counts)]] += 1

    master = np.random.default_rng(seed)
    graphs: List[nx.Graph] = []
    params_list: List[Dict] = []

    for t, cnt in zip(types, counts):
        if cnt <= 0:
            continue
        # 每种拓扑独立 rng，互不干扰
        type_seed = int(master.integers(0, 2**31))
        rng = np.random.default_rng(type_seed)
        for _ in range(int(cnt)):
            G, params = sample_graph(
                n_range, t, rng,
                m_range=m_range, k_avg_range=k_avg_range,
                ws_k_choices=ws_k_choices, ws_beta_range=ws_beta_range,
            )
            graphs.append(G)
            params_list.append(params)

    # 打乱顺序，避免同拓扑样本在训练集里扎堆
    order = master.permutation(len(graphs))
    graphs = [graphs[i] for i in order]
    params_list = [params_list[i] for i in order]
    return graphs, params_list
