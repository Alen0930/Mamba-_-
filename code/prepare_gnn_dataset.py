"""
为 GNN + 双向 Mamba 准备数据集：复用现有 CoreHD 标签，补算邻接矩阵

现有 datasets/ba_corehd_full 只缓存了 features/node_ids/ranks，缺邻接矩阵。
GNN 模型需要邻接矩阵做消息传递。此脚本加载现有数据集，对每个样本补算
标准化图 + 邻接矩阵（按 node_ids 重排，与特征序列同一节点顺序），保存到新目录。

不重跑 CoreHD 标签，仅做 O(n^2) 的邻接矩阵构建。

用法:
    python prepare_gnn_dataset.py \
        --src datasets/ba_corehd_full --dst datasets/ba_gnn
"""
import argparse
import io
import sys

import networkx as nx
import numpy as np

from network_dismantling.Mamba.dataset_generator import load_datasets, save_datasets


def add_adj_to_dataset(dataset) -> None:
    """对数据集每个缓存样本补算邻接矩阵（就地修改 cached_data）"""
    for sample, G in zip(dataset.cached_data, dataset.graphs):
        # 标准化图（与训练时 _standardize_graph 一致，节点重标为 0..n-1）
        G_std = dataset._standardize_graph(G)
        # 邻接矩阵按 node_ids 重排，与特征序列同一节点顺序
        adj = nx.to_numpy_array(G_std, dtype=np.uint8)
        adj = adj[np.ix_(sample['node_ids'], sample['node_ids'])]
        sample['adj'] = adj.astype(np.float32)


def main():
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="为 GNN 数据集补算邻接矩阵")
    parser.add_argument("--src", type=str, default="datasets/ba_corehd_full", help="源数据集目录")
    parser.add_argument("--dst", type=str, default="datasets/ba_gnn", help="目标数据集目录")
    args = parser.parse_args()

    print(f"加载源数据集: {args.src}")
    train_ds, val_ds = load_datasets(args.src, feature_set="degree")
    print(f"  train={len(train_ds)}, val={len(val_ds)}")

    print("补算邻接矩阵...")
    add_adj_to_dataset(train_ds)
    add_adj_to_dataset(val_ds)

    # 校验
    s0 = train_ds[0]
    print(f"校验 train[0]: features={tuple(s0['features'].shape)}, "
          f"adj={tuple(s0['adj'].shape)}, ranks={tuple(s0['ranks'].shape)}")
    assert s0['adj'].shape[0] == s0['adj'].shape[1] == s0['features'].shape[0], "邻接矩阵维度错误"

    print(f"保存到: {args.dst}")
    save_datasets(
        train_ds, val_ds, args.dst,
        extra_meta={"label_method": "CoreHD", "feature_set": "degree", "has_adj": True},
    )
    print("完成。")


if __name__ == "__main__":
    main()
