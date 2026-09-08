"""
消融实验评测脚本

对比三种方法在多个独立测试网络上的拆解效果，回答「模型到底学到了什么」：

    1. degree         : 纯度启发式（无学习基线）
    2. mamba-degree   : Mamba 模型，仅用度作为输入特征（input_dim=1，有学习）
    3. mamba-4feat    : Mamba 模型，度+k-core+PageRank+closeness（input_dim=4，当前完整模型）

（可选 --include-random 增加随机初始化 Mamba 作为 sanity check）

指标（越小越好，来自 evaluate.calc_metrics）：
    stop_step : 移除多少个节点后最大连通分量 <= 10%
    fc_value  : 移除节点比例使最大连通分量 <= 1%（网络瓦解）

用法:
    python ablation.py \
        --degree-ckpt checkpoints/ablation_degree/best_model.pth \
        --four-ckpt checkpoints/full_train/best_model.pth \
        --n-sizes 500,1000,1500 --seeds-per-size 3
"""
import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np

from network_dismantling.unified_interface import dismantle
from network_dismantling.Mamba.model_io import unregister_model
from evaluate import calc_metrics


# ---------------------------------------------------------------------------
# 测试网络生成
# ---------------------------------------------------------------------------
def build_test_graph(n: int, m: int = 3, seed: int = 0) -> nx.Graph:
    """生成独立 BA 测试网络（取最大连通分量保证连通）"""
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G


# ---------------------------------------------------------------------------
# 评测主流程
# ---------------------------------------------------------------------------
def run_ablation(
    degree_ckpt: str,
    four_ckpt: str,
    n_sizes: List[int],
    seeds_per_size: int,
    m: int = 3,
    include_random: bool = False,
    gnn_ckpt: str = None,
    device: str = None,
    stop_ratio: float = 0.1,
    fc_threshold: float = 0.01,
) -> List[Dict]:
    """
    在多张测试图上运行三种（或四种）方法的拆解评测

    Returns
    -------
    rows : List[Dict]
        每张图一行结果，含 n / seed / 各方法的 stop_step 与 fc_value
    """
    device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    # 确保无全局注册权重残留（random 基线依赖此保证）
    unregister_model()

    method_specs = [
        ("degree", "degree", None),
        ("mamba-degree", "mamba", degree_ckpt),
        ("mamba-4feat", "mamba", four_ckpt),
    ]
    if gnn_ckpt is not None:
        method_specs.append(("mamba-gnn", "mamba_gnn", gnn_ckpt))
    if include_random:
        # 随机初始化 Mamba（不加载任何权重，input_dim=4）
        method_specs.append(("mamba-random", "mamba", None))

    rows: List[Dict] = []
    for n in n_sizes:
        for s in range(seeds_per_size):
            seed = 1000 * n + s
            G = build_test_graph(n, m=m, seed=seed)
            print(f"\n=== 测试网络 n={G.number_of_nodes()}, m={m}, seed={seed} ===")

            row = {"n": G.number_of_nodes(), "seed": seed}

            for name, method, ckpt in method_specs:
                print(f"  运行 {name:<14} ... ", end="", flush=True)

                if method in ("mamba", "mamba_gnn"):
                    kwargs = {"method": method}
                    if ckpt is not None:
                        kwargs["model_path"] = ckpt
                    seq = dismantle(G, **kwargs)
                else:
                    seq = dismantle(G, method=method)

                stop_step, fc_value, reach_stop, reach_fc = calc_metrics(
                    G, seq, stop_ratio=stop_ratio, fc_threshold=fc_threshold
                )
                row[f"{name}_stop"] = stop_step
                row[f"{name}_fc"] = round(fc_value, 4)
                print(f"stop={stop_step}, fc={fc_value:.4f}")

            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# 结果汇总
# ---------------------------------------------------------------------------
def print_summary(rows: List[Dict], method_names: List[str]):
    """打印每张图明细 + 跨图平均"""
    print("\n" + "=" * 76)
    print("消融结果明细")
    print("=" * 76)

    header = f"{'n':<6} {'seed':<8}" + "".join(
        f"{m + '-stop':<12} {m + '-fc':<10}" for m in method_names
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        line = f"{row['n']:<6} {row['seed']:<8}"
        for m in method_names:
            line += f"{row[f'{m}_stop']:<12} {row[f'{m}_fc']:<10.4f}"
        print(line)

    # 跨图平均
    print("\n" + "=" * 76)
    print("跨图平均（越小越好）")
    print("=" * 76)
    print(f"{'方法':<14} {'平均 stop_step':<16} {'平均 fc_value':<14}")
    print("-" * 76)
    avg = {}
    for m in method_names:
        avg_stop = np.mean([r[f"{m}_stop"] for r in rows])
        avg_fc = np.mean([r[f"{m}_fc"] for r in rows])
        avg[m] = (avg_stop, avg_fc)
        print(f"{m:<14} {avg_stop:<16.1f} {avg_fc:<14.4f}")
    print("=" * 76)

    return avg


def save_csv(rows: List[Dict], method_names: List[str], out_path: str):
    """保存结果到 CSV（便于后续论文表格整理）"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["n", "seed"]
    for m in method_names:
        fields += [f"{m}_stop", f"{m}_fc"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    print(f"\n结果已保存: {out_path}")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Mamba 拆解消融实验评测")
    parser.add_argument("--degree-ckpt", type=str, default="checkpoints/ablation_degree/best_model.pth",
                        help="degree-only Mamba 检查点路径")
    parser.add_argument("--four-ckpt", type=str, default="checkpoints/full_train/best_model.pth",
                        help="4 特征 Mamba 检查点路径")
    parser.add_argument("--n-sizes", type=str, default="500,1000,1500",
                        help="测试网络节点数列表（逗号分隔）")
    parser.add_argument("--seeds-per-size", type=int, default=3, help="每个规模的随机种子数")
    parser.add_argument("--m", type=int, default=3, help="BA 连边参数")
    parser.add_argument("--include-random", action="store_true", help="增加随机初始化 Mamba 基线")
    parser.add_argument("--gnn-ckpt", type=str, default="checkpoints/gnn_mamba/best_model.pth",
                        help="GNN+双向 Mamba 检查点路径（None 则不加入对比）")
    parser.add_argument("--device", type=str, default=None, help="设备（默认自动）")
    parser.add_argument("--output", type=str, default="results/ablation.csv", help="CSV 输出路径")
    args = parser.parse_args()

    n_sizes = [int(x) for x in args.n_sizes.split(",")]

    method_names = ["degree", "mamba-degree", "mamba-4feat"]
    gnn_ckpt = args.gnn_ckpt if args.gnn_ckpt and args.gnn_ckpt.lower() != "none" else None
    if gnn_ckpt is not None:
        method_names.append("mamba-gnn")
    if args.include_random:
        method_names.append("mamba-random")

    print("=" * 76)
    print("Mamba 拆解消融实验")
    print("=" * 76)
    print(f"degree-only 检查点: {args.degree_ckpt}")
    print(f"4 特征检查点: {args.four_ckpt}")
    if gnn_ckpt is not None:
        print(f"GNN+双向 Mamba 检查点: {gnn_ckpt}")
    print(f"测试网络: BA, n={n_sizes}, m={args.m}, seeds_per_size={args.seeds_per_size}")
    print(f"对比方法: {method_names}")
    print("=" * 76)

    rows = run_ablation(
        degree_ckpt=args.degree_ckpt,
        four_ckpt=args.four_ckpt,
        n_sizes=n_sizes,
        seeds_per_size=args.seeds_per_size,
        m=args.m,
        include_random=args.include_random,
        gnn_ckpt=gnn_ckpt,
        device=args.device,
    )

    print_summary(rows, method_names)
    save_csv(rows, method_names, args.output)
    print("\n消融实验完成。")


if __name__ == "__main__":
    main()
