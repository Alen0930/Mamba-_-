"""
RL（PPO）模型正式评测脚本

在 n=500/1000/1500 的独立 BA 测试图上，对比：
    - degree          : 纯启发式下界
    - CoreHD          : 动态算法（seed 固定，作为可复现上界）
    - gnn-dyn-b25     : 监督模型动态重算（此前最优 stop_step）
    - rl-greedy       : RL 模型贪心策略（每步选 logits 最高节点 + 动态重算）

指标：stop_step（LCC<=10% 移除数）、fc_value（LCC<=1% 节点占比）

用法:
    python eval_rl.py --ckpt checkpoints/gnn_mamba_rl/best_model.pth
"""
import argparse
import csv
import io
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from network_dismantling.Mamba.gnn_mamba_actor_critic import create_actor_critic_from_supervised
from network_dismantling.Mamba.rl_env import lcc_size
from network_dismantling.Mamba.feature_encoding import extract_node_features
from network_dismantling.unified_interface import dismantle
from network_dismantling.Mamba.model_io import unregister_model
from evaluate import calc_metrics


def build_test_graph(n, m=3, seed=0):
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G


def greedy_sequence(model, G, device, stop_condition):
    """RL 贪心：每步选 logits 最高节点，动态重算"""
    G_tmp = G.copy()
    seq = []
    while G_tmp.number_of_nodes() > 0:
        node_list = list(G_tmp.nodes())
        G_std = nx.relabel_nodes(G_tmp, {v: i for i, v in enumerate(node_list)})
        features, node_ids = extract_node_features(G_std, feature_set='degree')
        adj = nx.to_numpy_array(G_std, dtype=np.uint8)
        adj = adj[np.ix_(node_ids, node_ids)]
        with torch.no_grad():
            x = torch.from_numpy(features).float().to(device).unsqueeze(0)
            a = torch.from_numpy(adj).float().to(device).unsqueeze(0)
            logits, _ = model(x, a)
        best_pos = int(logits.squeeze(0).argmax())
        removed_orig = node_list[node_ids[best_pos]]
        seq.append(removed_orig)
        G_tmp.remove_node(removed_orig)
        if G_tmp.number_of_nodes() > 0 and lcc_size(G_tmp) <= stop_condition:
            break
    return seq


def main():
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True)

    parser = argparse.ArgumentParser(description="RL 模型正式评测")
    parser.add_argument("--ckpt", type=str, default="checkpoints/gnn_mamba_rl/best_model.pth")
    parser.add_argument("--gnn-ckpt", type=str, default="checkpoints/gnn_mamba/best_model.pth",
                        help="监督模型（对比 gnn-dyn）")
    parser.add_argument("--n-sizes", type=str, default="500,1000,1500")
    parser.add_argument("--seeds-per-size", type=int, default=3)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/rl_eval.csv")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    unregister_model()

    print("=" * 76)
    print("RL (PPO) 模型正式评测")
    print("=" * 76)
    print(f"RL 模型: {args.ckpt}")
    print(f"监督模型: {args.gnn_ckpt}")
    print(f"测试图: BA n={args.n_sizes}, m={args.m}, seeds={args.seeds_per_size}")
    print("=" * 76)

    rl_model = create_actor_critic_from_supervised(args.ckpt, device)
    rl_model.eval()

    n_sizes = [int(x) for x in args.n_sizes.split(",")]
    rows = []

    for n in n_sizes:
        for s in range(args.seeds_per_size):
            seed = 1000 * n + s
            G = build_test_graph(n, m=args.m, seed=seed)
            n_nodes = G.number_of_nodes()
            stop_cond = max(1, int(0.01 * n_nodes))
            print(f"\n=== n={n_nodes}, seed={seed} ===", flush=True)

            row = {"n": n_nodes, "seed": seed}

            # 基线（CoreHD 固定 seed 保证可复现）
            for name, kwargs in [
                ("degree", {"method": "degree"}),
                ("CoreHD", {"method": "CoreHD", "seed": 0}),
                ("gnn-dyn-b25", {"method": "mamba_gnn", "model_path": args.gnn_ckpt, "batch_size": 25}),
            ]:
                seq = dismantle(G, **kwargs)
                stop, fc, _, _ = calc_metrics(G, seq)
                row[f"{name}_stop"] = stop
                row[f"{name}_fc"] = round(fc, 4)
                print(f"  {name:<14} stop={stop:<5} fc={fc:.4f}", flush=True)

            # RL 贪心
            rl_seq = greedy_sequence(rl_model, G, device, stop_cond)
            stop, fc, _, _ = calc_metrics(G, rl_seq)
            row["rl_stop"] = stop
            row["rl_fc"] = round(fc, 4)
            print(f"  {'rl-greedy':<14} stop={stop:<5} fc={fc:.4f}", flush=True)

            rows.append(row)

    # 汇总
    print("\n" + "=" * 76)
    print("跨图平均（stop_step 越小越好，fc_value 越小越好）")
    print("=" * 76)
    methods = ["degree", "CoreHD", "gnn-dyn-b25", "rl"]
    print(f"{'方法':<16} {'平均 stop_step':<16} {'平均 fc_value':<14}")
    print("-" * 76)
    for m in methods:
        avg_stop = np.mean([r[f"{m}_stop"] for r in rows])
        avg_fc = np.mean([r[f"{m}_fc"] for r in rows])
        print(f"{m:<16} {avg_stop:<16.1f} {avg_fc:<14.4f}")

    # 保存
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["n", "seed"] + [f"{m}_{k}" for m in methods for k in ("stop", "fc")]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
