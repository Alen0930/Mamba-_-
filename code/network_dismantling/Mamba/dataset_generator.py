"""
批量数据集生成器

生成 BA 无标度网络数据集，内部调用 CoreHD 生成拆解标签，
自动构建 DismantlingDataset 并按 8:2 划分训练/验证集。
输出可直接传入 DataLoader（配合 trainer.collate_fn 使用）。

使用示例:
    from network_dismantling.Mamba.dataset_generator import (
        generate_ba_graphs,
        build_datasets,
        build_dataloaders,
        generate_and_save_datasets,
        load_datasets,
    )

    # 方式 1: 一键生成并保存（BA 网络 500-1500 节点，CoreHD 标签）
    train_ds, val_ds = generate_and_save_datasets(
        out_dir="datasets/ba_corehd",
        num_samples=16,          # 样本总数可设置
        n_range=(500, 1500),     # 节点数范围
        m_range=(2, 5),          # BA 连边参数范围
        seed=42,
    )

    # 方式 2: 手动构建（自定义图列表）
    graphs = generate_ba_graphs(num_samples=16, n_range=(500, 1500), m_range=(2, 5))
    train_ds, val_ds = build_datasets(graphs, split_ratio=0.8, seed=42)

    # 方式 3: 从本地文件加载
    train_ds, val_ds = load_datasets("datasets/ba_corehd")

    # 构建 DataLoader（与 trainer.collate_fn 兼容）
    train_loader, val_loader = build_dataloaders(train_ds, val_ds, batch_size=2)

命令行:
    python -m network_dismantling.Mamba.dataset_generator \
        --num-samples 16 --out-dir datasets/ba_corehd --seed 42
"""
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
from torch.utils.data import DataLoader, Sampler

from .trainer import DismantlingDataset, collate_fn
from .feature_encoding import ORDERS
from .graph_factory import (
    DEFAULT_BA_M_RANGE,
    DEFAULT_ER_K_AVG_RANGE,
    DEFAULT_WS_BETA_RANGE,
    DEFAULT_WS_K_CHOICES,
    GRAPH_TYPES,
    sample_graph,
    sample_mixed_graphs,
)
from network_dismantling.unified_interface import dismantle

logger = logging.getLogger(__name__)

# 保存文件的固定命名
TRAIN_FILE = "train_dataset.pkl"
VAL_FILE = "val_dataset.pkl"

# 默认配置
DEFAULT_N_RANGE = (500, 1500)
DEFAULT_M_RANGE = DEFAULT_BA_M_RANGE
DEFAULT_SPLIT_RATIO = 0.8
DEFAULT_LABEL_METHOD = "CoreHD"
DEFAULT_GRAPH_TYPE = "ba"
DEFAULT_ORDER = "degree"
# 混合拓扑的默认配比（三拓扑等权）
DEFAULT_TOPOLOGY_WEIGHTS = {"ba": 1.0, "er": 1.0, "ws": 1.0}


# ---------------------------------------------------------------------------
# 图生成
# ---------------------------------------------------------------------------
def generate_ba_graphs(
    num_samples: int = 16,
    n_range: Tuple[int, int] = DEFAULT_N_RANGE,
    m_range: Tuple[int, int] = DEFAULT_M_RANGE,
    seed: Optional[int] = None,
) -> List[nx.Graph]:
    """
    批量生成 BA 无标度网络

    Parameters
    ----------
    num_samples : int
        样本总数
    n_range : Tuple[int, int]
        节点数范围 [min, max]（含端点）
    m_range : Tuple[int, int]
        每个新节点连接的边数范围 [min, max]（含端点）
    seed : int, optional
        随机种子，保证可复现

    Returns
    -------
    graphs : List[nx.Graph]
        生成的 BA 图列表
    """
    rng = np.random.default_rng(seed)
    graphs = []

    for i in range(num_samples):
        n = int(rng.integers(n_range[0], n_range[1] + 1))
        m = int(rng.integers(m_range[0], m_range[1] + 1))
        graph_seed = int(rng.integers(0, 2**31))
        G = nx.barabasi_albert_graph(n, m, seed=graph_seed)
        graphs.append(G)
        logger.info("Generated BA graph %d/%d: n=%d, m=%d", i + 1, num_samples, n, m)

    return graphs


def generate_graphs(
    num_samples: int = 16,
    graph_type: str = DEFAULT_GRAPH_TYPE,
    n_range: Tuple[int, int] = DEFAULT_N_RANGE,
    m_range: Tuple[int, int] = DEFAULT_M_RANGE,
    er_k_avg_range: Tuple[float, float] = DEFAULT_ER_K_AVG_RANGE,
    ws_k_choices: Tuple[int, ...] = DEFAULT_WS_K_CHOICES,
    ws_beta_range: Tuple[float, float] = DEFAULT_WS_BETA_RANGE,
    topology_weights: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
) -> Tuple[List[nx.Graph], List[Dict]]:
    """
    批量生成图，支持 BA / ER / WS / mixed 四种模式。

    与 generate_ba_graphs 的区别：除了返回图列表，还返回每张图的**参数记录**
    （graph_type / n_requested / n_actual / m / k_avg / k / beta / seed），
    供写入数据集 meta 以便复现。ER 取 LCC 后节点数会变，n_actual 是后续所有
    指标（AUC/stop/fc）的分母，必须记录下来。

    'mixed' 模式下每种拓扑使用独立 rng，保证改变配比或样本总数时，
    各拓扑已生成的图不变（BA 子集可单独复现，便于受控对比）。

    Returns
    -------
    (graphs, params_list) : Tuple[List[nx.Graph], List[Dict]]
    """
    if graph_type not in GRAPH_TYPES:
        raise ValueError(f"未知图类型 '{graph_type}'，可选 {GRAPH_TYPES}")

    if graph_type == "mixed":
        weights = topology_weights or DEFAULT_TOPOLOGY_WEIGHTS
        logger.info("Generating %d mixed graphs: weights=%s, n=%s",
                    num_samples, weights, n_range)
        return sample_mixed_graphs(
            num_samples, topology_weights=weights, n_range=n_range, seed=seed,
            m_range=m_range, k_avg_range=er_k_avg_range,
            ws_k_choices=ws_k_choices, ws_beta_range=ws_beta_range,
        )

    rng = np.random.default_rng(seed)
    graphs: List[nx.Graph] = []
    params_list: List[Dict] = []
    for i in range(num_samples):
        G, params = sample_graph(
            n_range, graph_type, rng,
            m_range=m_range, k_avg_range=er_k_avg_range,
            ws_k_choices=ws_k_choices, ws_beta_range=ws_beta_range,
        )
        graphs.append(G)
        params_list.append(params)
        logger.info(
            "Generated %s graph %d/%d: n=%d->%d, %s",
            graph_type.upper(), i + 1, num_samples,
            params["n_requested"], params["n_actual"],
            {k: v for k, v in params.items()
             if k in ("m", "k_avg", "k", "beta")},
        )

    return graphs, params_list


# ---------------------------------------------------------------------------
# 数据集构建
# ---------------------------------------------------------------------------
def build_datasets(
    graphs: List[nx.Graph],
    split_ratio: float = DEFAULT_SPLIT_RATIO,
    seed: Optional[int] = None,
    label_method: str = DEFAULT_LABEL_METHOD,
    cache_features: bool = True,
    feature_set: str = "all",
    order: str = "degree",
) -> Tuple[DismantlingDataset, DismantlingDataset]:
    """
    构建训练/验证 DismantlingDataset，按 split_ratio 划分（默认 8:2）

    标签由 label_method 指定的拆解算法生成（默认 CoreHD），
    内部自动完成图标准化、特征编码与排序标签生成。

    Parameters
    ----------
    graphs : List[nx.Graph]
        图列表
    split_ratio : float
        训练集占比，默认 0.8
    seed : int, optional
        划分随机种子
    label_method : str
        监督信号算法名（unified_interface 注册名），默认 'CoreHD'
    cache_features : bool
        是否缓存特征与标签（推荐 True，加速训练）
    feature_set : str
        特征集选择 'all' | 'degree'（'degree' 用于消融实验）

    Returns
    -------
    (train_dataset, val_dataset) : Tuple[DismantlingDataset, DismantlingDataset]
        可直接传入 DataLoader 的数据集
    """
    if not graphs:
        raise ValueError("graphs 不能为空")
    if not 0 < split_ratio < 1:
        raise ValueError(f"split_ratio 必须在 (0, 1) 范围内，当前值：{split_ratio}")

    # 随机打乱后按比例划分
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(graphs))
    n_train = max(1, int(len(graphs) * split_ratio))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # CoreHD 等算法生成拆解标签（内部自动标准化图并提取特征）
    dismantler_fn = lambda G: dismantle(G, method=label_method, stop_condition=1)

    logger.info(
        "Building datasets: %d graphs -> %d train / %d val (label=%s, feature_set=%s)",
        len(graphs), len(train_idx), len(val_idx), label_method, feature_set,
    )

    train_dataset = DismantlingDataset(
        [graphs[i] for i in train_idx],
        dismantler_fn=dismantler_fn,
        cache_features=cache_features,
        feature_set=feature_set,
        order=order,
    )
    val_dataset = DismantlingDataset(
        [graphs[i] for i in val_idx],
        dismantler_fn=dismantler_fn,
        cache_features=cache_features,
        feature_set=feature_set,
        order=order,
    )

    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# 保存 / 加载
# ---------------------------------------------------------------------------
def save_dataset(dataset: DismantlingDataset, path, extra_meta: Optional[Dict] = None) -> str:
    """
    保存单个数据集到本地文件（pickle 格式）

    保存内容：图列表 + 预处理缓存（特征/标签），
    加载后无需重新运行 CoreHD。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "version": 1,
        "graphs": dataset.graphs,
        "cached_data": dataset.cached_data,
        "meta": extra_meta or {},
    }

    with open(path, "wb") as f:
        pickle.dump(data, f)

    logger.info("Dataset saved: %s (%d samples)", path, len(dataset))
    return str(path)


def load_dataset(path, feature_set: str = "all") -> DismantlingDataset:
    """
    从本地文件加载数据集

    加载时直接使用缓存的特征/标签，不重新运行标签生成算法。
    feature_set='degree' 时，在取样本阶段对缓存特征取 degree 子集（复用缓存，无需重新生成）。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    if data.get("version") != 1:
        raise ValueError(f"不支持的数据集版本: {data.get('version')}")

    # 重建 DismantlingDataset：不触发预处理，直接填充缓存
    dataset = DismantlingDataset(
        data["graphs"], dismantler_fn=None, cache_features=False, feature_set=feature_set
    )
    dataset.cached_data = data["cached_data"]
    dataset.cache_features = True

    logger.info("Dataset loaded: %s (%d samples, feature_set=%s)", path, len(dataset), feature_set)
    return dataset


def save_datasets(
    train_dataset: DismantlingDataset,
    val_dataset: DismantlingDataset,
    out_dir,
    extra_meta: Optional[Dict] = None,
) -> str:
    """保存训练/验证数据集到指定目录（train_dataset.pkl / val_dataset.pkl）"""
    out_dir = Path(out_dir)
    meta = extra_meta or {}
    meta.setdefault("saved_at", datetime.now().isoformat())
    meta.setdefault("num_train", len(train_dataset))
    meta.setdefault("num_val", len(val_dataset))

    save_dataset(train_dataset, out_dir / TRAIN_FILE, extra_meta=meta)
    save_dataset(val_dataset, out_dir / VAL_FILE, extra_meta=meta)

    return str(out_dir)


def load_datasets(out_dir, feature_set: str = "all") -> Tuple[DismantlingDataset, DismantlingDataset]:
    """从指定目录加载训练/验证数据集"""
    out_dir = Path(out_dir)
    train_dataset = load_dataset(out_dir / TRAIN_FILE, feature_set=feature_set)
    val_dataset = load_dataset(out_dir / VAL_FILE, feature_set=feature_set)
    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# DataLoader 构建
# ---------------------------------------------------------------------------
class LengthBucketBatchSampler(Sampler):
    """
    按序列长度分桶的 batch sampler。

    动机（修一个现存 bug）：collate_fn 会把一个 batch 内的序列 pad 到批内最大长度，
    而 GNNMambaModel.encode 只在 GNN 之后把 padding 位置归零——**padding 仍然作为
    token 送进 BiMamba**。反向那一路 Mamba 从序列尾部起算，必须穿过 L_max - n 个
    零向量才能到达真实节点，隐状态被一长串 padding 污染。n ∈ [500,1500] 时这个
    gap 最大可达 1000，train_shuffle=True 又让批内长度差最大化。

    分桶把长度接近的样本放进同一批，把 padding gap 压到最小，且**不需要改模型结构**。

    算法（torchtext BucketIterator 的经典做法）：
      1. 随机取一个 pool（大小 = batch_size × pool_multiplier）
      2. pool 内按长度排序后切成 batch —— 每批长度接近
      3. 打乱 batch 之间的顺序 —— 避免长度单调导致的梯度偏置
    """

    def __init__(
        self,
        lengths,
        batch_size: int = 4,
        shuffle: bool = True,
        pool_multiplier: int = 20,
        seed: int = 0,
    ):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pool_size = max(batch_size, batch_size * pool_multiplier)
        self.rng = np.random.default_rng(seed)
        self._n = len(self.lengths)

    def __iter__(self):
        order = (self.rng.permutation(self._n) if self.shuffle
                 else np.arange(self._n))
        batches = []
        for i in range(0, self._n, self.pool_size):
            pool = order[i:i + self.pool_size]
            # pool 内按长度排序 -> 每批长度接近（padding gap 小）
            pool = pool[np.argsort(self.lengths[pool], kind="stable")]
            for j in range(0, len(pool), self.batch_size):
                chunk = pool[j:j + self.batch_size]
                if len(chunk) > 0:
                    batches.append([int(x) for x in chunk])
        if self.shuffle:
            self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (self._n + self.batch_size - 1) // self.batch_size


def build_dataloaders(
    train_dataset: DismantlingDataset,
    val_dataset: DismantlingDataset,
    batch_size: int = 2,
    num_workers: int = 0,
    train_shuffle: bool = True,
    length_bucketing: bool = True,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    构建训练/验证 DataLoader，直接用于 trainer.train()

    Parameters
    ----------
    batch_size : int
        批次大小。500-1500 节点的图显存占用较大，建议 2-4
    num_workers : int
        数据加载进程数，Windows 上建议 0
    train_shuffle : bool
        训练集是否打乱（分桶模式下由 LengthBucketBatchSampler 接管顺序）
    length_bucketing : bool
        是否按长度分桶（默认 True）。见 LengthBucketBatchSampler 的说明——
        它把 padding 对 BiMamba 的污染压到最小，多规模/多拓扑训练尤其重要。
    seed : int
        分桶采样的随机种子
    """
    if length_bucketing:
        train_lengths = [ds["features"].shape[0] if isinstance(ds, dict)
                         else ds.features.shape[0]
                         for ds in train_dataset]
        val_lengths = [ds["features"].shape[0] if isinstance(ds, dict)
                       else ds.features.shape[0]
                       for ds in val_dataset]
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=LengthBucketBatchSampler(
                train_lengths, batch_size=batch_size,
                shuffle=train_shuffle, seed=seed,
            ),
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=LengthBucketBatchSampler(
                val_lengths, batch_size=batch_size, shuffle=False, seed=seed,
            ),
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        return train_loader, val_loader

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# 一键生成
# ---------------------------------------------------------------------------
def generate_and_save_datasets(
    out_dir,
    num_samples: int = 16,
    n_range: Tuple[int, int] = DEFAULT_N_RANGE,
    m_range: Tuple[int, int] = DEFAULT_M_RANGE,
    split_ratio: float = DEFAULT_SPLIT_RATIO,
    seed: Optional[int] = None,
    label_method: str = DEFAULT_LABEL_METHOD,
    feature_set: str = "degree",
    order: str = DEFAULT_ORDER,
    graph_type: str = DEFAULT_GRAPH_TYPE,
    er_k_avg_range: Tuple[float, float] = DEFAULT_ER_K_AVG_RANGE,
    ws_k_choices: Tuple[int, ...] = DEFAULT_WS_K_CHOICES,
    ws_beta_range: Tuple[float, float] = DEFAULT_WS_BETA_RANGE,
    topology_weights: Optional[Dict[str, float]] = None,
) -> Tuple[DismantlingDataset, DismantlingDataset]:
    """
    一键生成数据集（CoreHD 标签）并保存到本地，支持多拓扑

    完整流程: 生成图 -> CoreHD 生成标签 -> 8:2 划分 -> 保存

    Parameters
    ----------
    out_dir : str
        输出目录
    num_samples : int
        样本总数（训练 + 验证）
    n_range : Tuple[int, int]
        节点数范围
    m_range : Tuple[int, int]
        BA 连边参数范围
    split_ratio : float
        训练集占比，默认 0.8
    seed : int, optional
        随机种子（图生成与划分共用）
    label_method : str
        监督信号算法名，默认 'CoreHD'
    feature_set : str
        节点特征集 'degree' | 'all'，默认 'degree'
    order : str
        节点序列排序 'degree' | 'bfs' | 'dfs' | 'core'，默认 'degree'。
        **排序会烘焙进缓存数据**（features/adj/ranks 都按该顺序重排），
        所以换排序必须重新构建数据集。
        （默认取 degree：训练侧 run_gnn_train.py 只用 degree，且 'all' 的
        closeness_centrality 在 n=1500 上是 O(n^2) 全对最短路，白算且拖慢生成）
    graph_type : str
        'ba' | 'er' | 'ws' | 'mixed'，默认 'ba'
    er_k_avg_range : tuple
        ER 目标平均度范围（p = k_avg/(n-1)）
    ws_k_choices : tuple
        WS 邻居环宽度候选（自动取偶数）
    ws_beta_range : tuple
        WS 重连概率范围
    topology_weights : dict, optional
        graph_type='mixed' 时的拓扑配比，默认三者等权

    Returns
    -------
    (train_dataset, val_dataset) : Tuple[DismantlingDataset, DismantlingDataset]
    """
    print(f"Step 1/4: Generating {num_samples} {graph_type.upper()} graphs (n={n_range})...")
    graphs, graph_params = generate_graphs(
        num_samples=num_samples, graph_type=graph_type, n_range=n_range,
        m_range=m_range, er_k_avg_range=er_k_avg_range,
        ws_k_choices=ws_k_choices, ws_beta_range=ws_beta_range,
        topology_weights=topology_weights, seed=seed,
    )
    n_by_type: Dict[str, int] = {}
    for p in graph_params:
        n_by_type[p["graph_type"]] = n_by_type.get(p["graph_type"], 0) + 1
    print(f"  拓扑分布: {n_by_type}   n_actual 范围: "
          f"[{min(p['n_actual'] for p in graph_params)}, "
          f"{max(p['n_actual'] for p in graph_params)}]")

    print(f"Step 2/4: Generating {label_method} labels and building datasets...")
    train_dataset, val_dataset = build_datasets(
        graphs,
        split_ratio=split_ratio,
        seed=seed,
        label_method=label_method,
        feature_set=feature_set,
        order=order,
    )

    print(f"Step 3/4: Saving datasets to {out_dir} ...")
    meta = {
        "label_method": label_method,
        "split_ratio": split_ratio,
        "seed": seed,
        "n_range": list(n_range),
        "m_range": list(m_range),
        "feature_set": feature_set,
        "order": order,
        # ---- 拓扑记录（复现用：ER 取 LCC 后 n 会变）----
        "graph_type": graph_type,
        "topology_counts": n_by_type,
        "er_k_avg_range": list(er_k_avg_range),
        "ws_k_choices": list(ws_k_choices),
        "ws_beta_range": list(ws_beta_range),
        "topology_weights": topology_weights or (
            DEFAULT_TOPOLOGY_WEIGHTS if graph_type == "mixed" else None
        ),
        "graph_params": graph_params,
    }
    save_datasets(train_dataset, val_dataset, out_dir, extra_meta=meta)

    print(
        f"Step 4/4: Done. train={len(train_dataset)} samples, val={len(val_dataset)} samples"
    )
    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def main():
    """命令行入口（仅 __main__ 使用）"""
    import io

    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    import argparse

    parser = argparse.ArgumentParser(description="批量生成网络拆解训练数据集（支持 BA/ER/WS/mixed）")
    parser.add_argument("--num-samples", type=int, default=16, help="样本总数 (默认 16)")
    parser.add_argument("--graph-type", type=str, default=DEFAULT_GRAPH_TYPE,
                        choices=list(GRAPH_TYPES), help="图拓扑 (默认 ba)；mixed = 三拓扑混合")
    parser.add_argument("--topology-weights", type=str, default=None,
                        help="mixed 模式配比，如 'ba:1,er:1,ws:1' (默认等权)")
    parser.add_argument("--n-min", type=int, default=DEFAULT_N_RANGE[0], help="节点数下限 (默认 500)")
    parser.add_argument("--n-max", type=int, default=DEFAULT_N_RANGE[1], help="节点数上限 (默认 1500)")
    parser.add_argument("--m-min", type=int, default=DEFAULT_M_RANGE[0], help="BA 连边参数下限 (默认 2)")
    parser.add_argument("--m-max", type=int, default=DEFAULT_M_RANGE[1], help="BA 连边参数上限 (默认 5)")
    parser.add_argument("--er-k-avg-min", type=float, default=DEFAULT_ER_K_AVG_RANGE[0],
                        help="ER 平均度下限 (默认 4)")
    parser.add_argument("--er-k-avg-max", type=float, default=DEFAULT_ER_K_AVG_RANGE[1],
                        help="ER 平均度上限 (默认 10)")
    parser.add_argument("--ws-k", type=int, nargs="+", default=list(DEFAULT_WS_K_CHOICES),
                        help="WS 邻居环宽度候选，自动取偶数 (默认 4 6 8)")
    parser.add_argument("--ws-beta-min", type=float, default=DEFAULT_WS_BETA_RANGE[0],
                        help="WS 重连概率下限 (默认 0.1)")
    parser.add_argument("--ws-beta-max", type=float, default=DEFAULT_WS_BETA_RANGE[1],
                        help="WS 重连概率上限 (默认 0.3)")
    parser.add_argument("--split-ratio", type=float, default=DEFAULT_SPLIT_RATIO, help="训练集占比 (默认 0.8)")
    parser.add_argument("--label-method", type=str, default=DEFAULT_LABEL_METHOD, help="标签算法 (默认 CoreHD)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--feature-set", type=str, default="degree",
                        choices=["degree", "all"], help="节点特征集 (默认 degree)")
    parser.add_argument("--order", type=str, default=DEFAULT_ORDER,
                        choices=list(ORDERS),
                        help="节点序列排序 (默认 degree)。排序会烘焙进缓存数据，"
                             "换排序需重建数据集")
    parser.add_argument("--out-dir", type=str, default="datasets/ba_corehd", help="输出目录")
    args = parser.parse_args()

    weights = None
    if args.topology_weights:
        weights = {}
        for part in args.topology_weights.split(","):
            k, v = part.split(":")
            weights[k.strip()] = float(v)

    generate_and_save_datasets(
        out_dir=args.out_dir,
        num_samples=args.num_samples,
        graph_type=args.graph_type,
        topology_weights=weights,
        n_range=(args.n_min, args.n_max),
        m_range=(args.m_min, args.m_max),
        er_k_avg_range=(args.er_k_avg_min, args.er_k_avg_max),
        ws_k_choices=tuple(args.ws_k),
        ws_beta_range=(args.ws_beta_min, args.ws_beta_max),
        split_ratio=args.split_ratio,
        seed=args.seed,
        label_method=args.label_method,
        feature_set=args.feature_set,
        order=args.order,
    )


if __name__ == "__main__":
    main()
