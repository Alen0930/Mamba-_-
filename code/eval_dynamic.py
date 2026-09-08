"""
动态重算评测脚本

对比「静态一次性」与「动态批重算」的拆解效果，验证动态重算能否逼近 CoreHD：

    - degree          : 纯度启发式（静态下界参考）
    - CoreHD          : 动态算法（每步重算 2-core，上界参考）
    - gnn-static      : GNN+双向 Mamba，静态一次性（batch_size=None）
    - gnn-dyn-b{b}    : GNN+双向 Mamba，每移除 b 个节点重算一次（动态）

指标（越小越好）：stop_step（移除多少个节点使 LCC<=10%）、fc_value（使 LCC<=1% 的节点比例）

用法:
    python eval_dynamic.py --batch-sizes 50,25 --n-sizes 500,1000,1500 --seeds-per-size 3
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


def build_test_graph(n: int, m: int = 3, seed: int = 0) -> nx.Graph:
    """生成独立 BA 测试网络（取最大连通分量保证连通）"""
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G


def run_dynamic_eval(
    gnn_ckpt: str,
    n_sizes: List[int],
    seeds_per_size: int,
    batch_sizes: List[int],
    m: int = 3,
    device: str = None,
    stop_ratio: float = 0.1,
    fc_threshold: float = 0.01,
) -> List[Dict]:
    device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    unregister_model()

    method_specs = [
        ("degree", "degree", None, None),
        ("CoreHD", "CoreHD", None, None),
        ("gnn-static", "mamba_gnn", gnn_ckpt, None),
    ]
    for b in batch_sizes:
        method_specs.append((f"gnn-dyn-b{b}", "mamba_gnn", gnn_ckpt, b))

    rows: List[Dict] = []
    for n in n_sizes:
        for s in range(seeds_per_size):
            seed = 1000 * n + s
            G = build_test_graph(n, m=m, seed=seed)
            print(f"\n=== 测试网络 n={G.number_of_nodes()}, m={m}, seed={seed} ===")

            row = {"n": G.number_of_nodes(), "seed": seed}
            for name, method, ckpt, batch in method_specs:
                print(f"  运行 {name:<14} ... ", end="", flush=True)
                if method in ("mamba", "mamba_gnn"):
                    kwargs = {"method": method, "model_path": ckpt}
                    if batch is not None:
                        kwargs["batch_size"] = batch
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


def print_summary(rows: List[Dict], method_names: List[str]):
    print("\n" + "=" * 76)
    print("动态重算评测明细")
    print("=" * 76)
    header = f"{'n':<6} {'seed':<8}" + "".join(
        f"{m + '-stop':<14} {m + '-fc':<10}" for m in method_names
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['n']:<6} {row['seed']:<8}"
        for m in method_names:
            line += f"{row[f'{m}_stop']:<14} {row[f'{m}_fc']:<10.4f}"
        print(line)

    print("\n" + "=" * 76)
    print("跨图平均（越小越好）")
    print("=" * 76)
    print(f"{'方法':<16} {'平均 stop_step':<16} {'平均 fc_value':<14}")
    print("-" * 76)
    for m in method_names:
        avg_stop = np.mean([r[f"{m}_stop"] for r in rows])
        avg_fc = np.mean([r[f"{m}_fc"] for r in rows])
        print(f"{m:<16} {avg_stop:<16.1f} {avg_fc:<14.4f}")
    print("=" * 76)


def save_csv(rows: List[Dict], method_names: List[str], out_path: str):
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


def main():
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="动态重算评测")
    parser.add_argument("--gnn-ckpt", type=str, default="checkpoints/gnn_mamba/best_model.pth")
    parser.add_argument("--batch-sizes", type=str, default="50", help="动态批次大小列表（逗号分隔）")
    parser.add_argument("--n-sizes", type=str, default="500,1000,1500")
    parser.add_argument("--seeds-per-size", type=int, default=3)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/dynamic_eval.csv")
    args = parser.parse_args()

    n_sizes = [int(x) for x in args.n_sizes.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    method_names = ["degree", "CoreHD", "gnn-static"]
    method_names += [f"gnn-dyn-b{b}" for b in batch_sizes]

    print("=" * 76)
    print("动态重算评测")
    print("=" * 76)
    print(f"GNN 检查点: {args.gnn_ckpt}")
    print(f"测试网络: BA, n={n_sizes}, m={args.m}, seeds_per_size={args.seeds_per_size}")
    print(f"动态批次: {batch_sizes}")
    print(f"对比方法: {method_names}")
    print("=" * 76)

    rows = run_dynamic_eval(
        gnn_ckpt=args.gnn_ckpt,
        n_sizes=n_sizes,
        seeds_per_size=args.seeds_per_size,
        batch_sizes=batch_sizes,
        m=args.m,
        device=args.device,
    )
    print_summary(rows, method_names)
    save_csv(rows, method_names, args.output)
    print("\n动态重算评测完成。")


if __name__ == "__main__":
    main()
