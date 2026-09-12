"""
按指定节点序列排序重建训练数据集

用途：为「结构编码排序」消融准备数据集（H8）。

背景：节点排序会**烘焙进缓存数据**（features / adj / ranks 都按该顺序重排），
所以换排序必须重建数据集，不能只改训练参数。

数据集来源：与 `datasets/mixed_gnn` 完全相同的 100 张图（BA/ER/WS 混合，
n∈[500,1500]），由同一个 seed=42 重新生成 —— 已验证边集与 train/val 划分
逐图一致，因此不同 order 之间的比较是干净的（唯一变量就是节点顺序）。

用法:
    python build_order_dataset.py --verify          # 仅验证 degree 序能复现 mixed_gnn
    python build_order_dataset.py --orders dfs core # 构建指定排序的数据集
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np

from network_dismantling.Mamba.dataset_generator import (
    DEFAULT_TOPOLOGY_WEIGHTS,
    build_datasets,
    load_datasets,
    save_datasets,
)
from network_dismantling.Mamba.feature_encoding import ORDERS
from network_dismantling.Mamba.graph_factory import sample_mixed_graphs

SRC_DIR = "datasets/mixed_gnn"
FALLBACK_OUT = "datasets/mixed_{order}"


def regenerate_graphs() -> list:
    """按 mixed_gnn 的生成参数复现那 100 张图（顺序也一致）"""
    graphs, _ = sample_mixed_graphs(
        100, topology_weights=DEFAULT_TOPOLOGY_WEIGHTS, n_range=(500, 1500), seed=42
    )
    return graphs


def build(order: str, graphs: list, out_dir: str, seed: int = 42):
    """用指定排序构建 train/val 数据集并保存"""
    tr, va = build_datasets(
        graphs, split_ratio=0.8, seed=seed, label_method="CoreHD",
        feature_set="degree", order=order,
    )
    meta = {
        "graph_type": "mixed",
        "topology_weights": DEFAULT_TOPOLOGY_WEIGHTS,
        "n_range": [500, 1500],
        "source": f"regenerated with same params as {SRC_DIR}",
        "order": order,
        "feature_set": "degree",
        "label_method": "CoreHD",
        "seed": seed,
    }
    save_datasets(tr, va, out_dir, extra_meta=meta)
    return tr, va


def verify_degree_order(graphs: list) -> bool:
    """
    用 order='degree' 重建，应当与现有 mixed_gnn **逐样本完全一致**。

    这是整套消融可信度的基础：若连 degree 序都复现不出来，说明重建流程有问题
    （图不同 / 标签不同 / 划分不同），后面所有 order 之间的比较都不成立。
    """
    ref_tr, ref_va = load_datasets(SRC_DIR, feature_set="degree")
    tr, va = build_datasets(
        graphs, split_ratio=0.8, seed=42, label_method="CoreHD",
        feature_set="degree", order="degree", cache_features=True,
    )

    ok = True
    for name, ref, new in [("train", ref_tr, tr), ("val", ref_va, va)]:
        if len(ref) != len(new):
            print(f"  ✗ {name} 样本数不同: {len(ref)} vs {len(new)}")
            ok = False
            continue
        for i in range(len(ref)):
            a, b = ref.cached_data[i], new.cached_data[i]
            same = (np.array_equal(a["node_ids"], b["node_ids"])
                    and np.array_equal(a["ranks"], b["ranks"])
                    and np.allclose(a["features"], b["features"])
                    and np.array_equal(a["adj"], b["adj"]))
            if not same:
                print(f"  ✗ {name} 样本 {i} 不一致")
                ok = False
                break
        else:
            print(f"  ✓ {name} {len(ref)} 个样本逐字段一致（node_ids/ranks/features/adj）")
    return ok


def main():
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True
        )

    ap = argparse.ArgumentParser(description="按节点排序重建数据集")
    ap.add_argument("--orders", type=str, nargs="*", default=[],
                    help=f"要构建的排序，可选 {ORDERS}（'degree' 除外，它已存在）")
    ap.add_argument("--verify", action="store_true",
                    help="只做验证：确认 degree 序能逐样本复现 mixed_gnn")
    ap.add_argument("--out-pattern", type=str, default=FALLBACK_OUT)
    args = ap.parse_args()

    print("=" * 76)
    print("按节点排序重建数据集")
    print("=" * 76)

    print("\n[1/3] 复现 mixed_gnn 的 100 张图...")
    graphs = regenerate_graphs()
    print(f"      {len(graphs)} 张（与 {SRC_DIR} 同参数同 seed）")

    print("\n[2/3] 验证 degree 序能否逐样本复现现有数据集...")
    if not verify_degree_order(graphs):
        raise SystemExit(
            "\n验证失败：重建流程与现有数据集不一致，后续 order 比较不可信。\n"
            "请先排查（图生成 / CoreHD 标签 / 划分），不要继续构建其它 order。"
        )
    print("      -> degree 序可完全复现，不同 order 间的比较是干净的")

    if args.verify:
        print("\n仅验证模式，结束。")
        return

    print(f"\n[3/3] 构建排序数据集: {args.orders or '(未指定)'}")
    for order in args.orders:
        if order not in ORDERS:
            print(f"  [跳过] 未知排序 '{order}'，可选 {ORDERS}")
            continue
        out_dir = args.out_pattern.format(order=order)
        if Path(out_dir).exists():
            print(f"  [跳过] {out_dir} 已存在")
            continue
        print(f"  构建 order='{order}' -> {out_dir} ...")
        tr, va = build(order, graphs, out_dir)
        print(f"    完成: train={len(tr)} val={len(va)}")

    print("\n完成。")


if __name__ == "__main__":
    main()
