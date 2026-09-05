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
from torch.utils.data import DataLoader

from .trainer import DismantlingDataset, collate_fn
from network_dismantling.unified_interface import dismantle

logger = logging.getLogger(__name__)

# 保存文件的固定命名
TRAIN_FILE = "train_dataset.pkl"
VAL_FILE = "val_dataset.pkl"

# 默认配置
DEFAULT_N_RANGE = (500, 1500)
DEFAULT_M_RANGE = (2, 5)
DEFAULT_SPLIT_RATIO = 0.8
DEFAULT_LABEL_METHOD = "CoreHD"


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


# ---------------------------------------------------------------------------
# 数据集构建
# ---------------------------------------------------------------------------
def build_datasets(
    graphs: List[nx.Graph],
    split_ratio: float = DEFAULT_SPLIT_RATIO,
    seed: Optional[int] = None,
    label_method: str = DEFAULT_LABEL_METHOD,
    cache_features: bool = True,
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
        "Building datasets: %d graphs -> %d train / %d val (label=%s)",
        len(graphs), len(train_idx), len(val_idx), label_method,
    )

    train_dataset = DismantlingDataset(
        [graphs[i] for i in train_idx],
        dismantler_fn=dismantler_fn,
        cache_features=cache_features,
    )
    val_dataset = DismantlingDataset(
        [graphs[i] for i in val_idx],
        dismantler_fn=dismantler_fn,
        cache_features=cache_features,
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


def load_dataset(path) -> DismantlingDataset:
    """
    从本地文件加载数据集

    加载时直接使用缓存的特征/标签，不重新运行标签生成算法。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    if data.get("version") != 1:
        raise ValueError(f"不支持的数据集版本: {data.get('version')}")

    # 重建 DismantlingDataset：不触发预处理，直接填充缓存
    dataset = DismantlingDataset(data["graphs"], dismantler_fn=None, cache_features=False)
    dataset.cached_data = data["cached_data"]
    dataset.cache_features = True

    logger.info("Dataset loaded: %s (%d samples)", path, len(dataset))
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


def load_datasets(out_dir) -> Tuple[DismantlingDataset, DismantlingDataset]:
    """从指定目录加载训练/验证数据集"""
    out_dir = Path(out_dir)
    train_dataset = load_dataset(out_dir / TRAIN_FILE)
    val_dataset = load_dataset(out_dir / VAL_FILE)
    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# DataLoader 构建
# ---------------------------------------------------------------------------
def build_dataloaders(
    train_dataset: DismantlingDataset,
    val_dataset: DismantlingDataset,
    batch_size: int = 2,
    num_workers: int = 0,
    train_shuffle: bool = True,
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
        训练集是否打乱
    """
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
) -> Tuple[DismantlingDataset, DismantlingDataset]:
    """
    一键生成 BA 数据集（CoreHD 标签）并保存到本地

    完整流程: 生成图 -> CoreHD 生成标签 -> 8:2 划分 -> 保存

    Parameters
    ----------
    out_dir : str
        输出目录
    num_samples : int
        样本总数（训练 + 验证）
    n_range / m_range : Tuple[int, int]
        BA 网络参数范围
    split_ratio : float
        训练集占比，默认 0.8
    seed : int, optional
        随机种子（图生成与划分共用）
    label_method : str
        监督信号算法名，默认 'CoreHD'

    Returns
    -------
    (train_dataset, val_dataset) : Tuple[DismantlingDataset, DismantlingDataset]
    """
    print(f"Step 1/4: Generating {num_samples} BA graphs (n={n_range}, m={m_range})...")
    graphs = generate_ba_graphs(
        num_samples=num_samples, n_range=n_range, m_range=m_range, seed=seed
    )

    print(f"Step 2/4: Generating {label_method} labels and building datasets...")
    train_dataset, val_dataset = build_datasets(
        graphs, split_ratio=split_ratio, seed=seed, label_method=label_method
    )

    print(f"Step 3/4: Saving datasets to {out_dir} ...")
    meta = {
        "label_method": label_method,
        "split_ratio": split_ratio,
        "seed": seed,
        "n_range": list(n_range),
        "m_range": list(m_range),
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

    parser = argparse.ArgumentParser(description="批量生成 BA 网络拆解训练数据集")
    parser.add_argument("--num-samples", type=int, default=16, help="样本总数 (默认 16)")
    parser.add_argument("--n-min", type=int, default=DEFAULT_N_RANGE[0], help="节点数下限 (默认 500)")
    parser.add_argument("--n-max", type=int, default=DEFAULT_N_RANGE[1], help="节点数上限 (默认 1500)")
    parser.add_argument("--m-min", type=int, default=DEFAULT_M_RANGE[0], help="BA 连边参数下限 (默认 2)")
    parser.add_argument("--m-max", type=int, default=DEFAULT_M_RANGE[1], help="BA 连边参数上限 (默认 5)")
    parser.add_argument("--split-ratio", type=float, default=DEFAULT_SPLIT_RATIO, help="训练集占比 (默认 0.8)")
    parser.add_argument("--label-method", type=str, default=DEFAULT_LABEL_METHOD, help="标签算法 (默认 CoreHD)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--out-dir", type=str, default="datasets/ba_corehd", help="输出目录")
    args = parser.parse_args()

    generate_and_save_datasets(
        out_dir=args.out_dir,
        num_samples=args.num_samples,
        n_range=(args.n_min, args.n_max),
        m_range=(args.m_min, args.m_max),
        split_ratio=args.split_ratio,
        seed=args.seed,
        label_method=args.label_method,
    )


if __name__ == "__main__":
    main()
