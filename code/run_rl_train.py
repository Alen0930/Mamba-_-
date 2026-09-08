"""
RL（PPO）网络拆解训练脚本

在监督预训练（CoreHD 标签）的 GNN+双向 Mamba 模型之上，用 PPO 做策略梯度微调，
把奖励从「拟合 CoreHD 标签」换成「每步 LCC 下降 / 拆解集大小」，突破行为克隆上界。

流程：
    1. 从监督检查点构建 Actor-Critic（复用 encoder + output_proj 作初始策略）
    2. 每轮迭代：随机生成 BA 图 -> 采样 rollout -> PPO 更新
    3. 定期用贪心策略（每步选 logits 最高节点 + 动态重算）评测 stop_step / fc，
       与 degree / CoreHD 对比，并保存检查点

用法:
    python run_rl_train.py                                          # 默认超参
    python run_rl_train.py --n 500 --iterations 200 --lr 3e-4
"""
import argparse
import io
import sys
import time
from datetime import datetime

import networkx as nx
import numpy as np
import torch

from network_dismantling.Mamba.gnn_mamba_actor_critic import create_actor_critic_from_supervised
from network_dismantling.Mamba.rl_env import RolloutBuffer, sample_trajectory, lcc_size
from network_dismantling.Mamba.ppo_trainer import PPOTrainer
from network_dismantling.Mamba.feature_encoding import extract_node_features
from network_dismantling.unified_interface import dismantle
from network_dismantling.Mamba.model_io import unregister_model
from evaluate import calc_metrics


def build_ba_graph(n: int, m: int = 3, seed: int = 0) -> nx.Graph:
    """生成独立 BA 测试网络（取最大连通分量保证连通）"""
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return G


def greedy_sequence(model, G: nx.Graph, device: str, stop_condition: int):
    """
    贪心策略：每步选 logits 最高节点移除，动态重算（batch_size=1 极限），
    返回完整移除序列（直到 LCC <= stop_condition）。
    """
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


def evaluate(model, device, n_sizes, seeds_per_size, m=3):
    """贪心策略评测：对比 degree / CoreHD / rl-greedy 的 stop_step 与 fc_value"""
    unregister_model()
    rows = []
    for n in n_sizes:
        for s in range(seeds_per_size):
            seed = 1000 * n + s
            G = build_ba_graph(n, m=m, seed=seed)
            n_nodes = G.number_of_nodes()
            stop_cond = max(1, int(0.01 * n_nodes))

            # 基线（CoreHD 固定 seed 保证可复现）
            degree_seq = dismantle(G, method="degree")
            corehd_seq = dismantle(G, method="CoreHD", seed=0)
            # RL 贪心
            rl_seq = greedy_sequence(model, G, device, stop_cond)
            rl_stop, rl_fc, _, _ = calc_metrics(G, rl_seq)

            rows.append({
                "n": n_nodes,
                "seed": seed,
                "degree": calc_metrics(G, degree_seq)[0],
                "corehd": calc_metrics(G, corehd_seq)[0],
                "rl_stop": rl_stop,
                "rl_fc": round(rl_fc, 4),
            })
    return rows


def main():
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True
        )

    parser = argparse.ArgumentParser(description="RL (PPO) 网络拆解训练")
    parser.add_argument("--ckpt", type=str, default="checkpoints/gnn_mamba/best_model.pth",
                        help="监督预训练检查点（作为初始策略）")
    parser.add_argument("--out-dir", type=str, default="checkpoints/gnn_mamba_rl",
                        help="RL 检查点保存目录")
    parser.add_argument("--n", type=int, default=500, help="训练图节点数")
    parser.add_argument("--m", type=int, default=3, help="BA 图 m 参数")
    parser.add_argument("--num-envs", type=int, default=8, help="每次迭代采样的图数")
    parser.add_argument("--action-k", type=int, default=10, help="每步移除的节点数（批动作）")
    parser.add_argument("--iterations", type=int, default=200, help="迭代次数")
    parser.add_argument("--stop-condition", type=int, default=1, help="RL 训练终止 LCC 阈值")
    parser.add_argument("--lr", type=float, default=3e-4, help="PPO 学习率")
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--k-epochs", type=int, default=3)
    parser.add_argument("--ppo-batch-size", type=int, default=16, help="PPO 更新 mini-batch 大小")
    parser.add_argument("--use-critic", action="store_true", help="用 critic(value baseline)+GAE；默认关闭，用 REINFORCE(batch-mean baseline)")
    parser.add_argument("--eval-interval", type=int, default=20, help="每多少迭代评测一次")
    parser.add_argument("--eval-sizes", type=str, default="500", help="评测图规模（逗号分隔）")
    parser.add_argument("--eval-seeds", type=int, default=3, help="每个规模评测种子数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 76)
    print("RL (PPO) 网络拆解训练")
    print("=" * 76)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"初始策略: {args.ckpt}")
    print(f"训练图: BA n={args.n}, m={args.m}")
    print(f"超参数: num_envs={args.num_envs}, action_k={args.action_k}, "
          f"iterations={args.iterations}, lr={args.lr}, "
          f"clip={args.clip_eps}, vf_coef={args.vf_coef}, ent_coef={args.ent_coef}, "
          f"gamma={args.gamma}, lam={args.lam}, k_epochs={args.k_epochs}")
    print("=" * 76)

    # ========== 1. 构建 Actor-Critic ==========
    model = create_actor_critic_from_supervised(args.ckpt, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = PPOTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        clip_eps=args.clip_eps,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        lam=args.lam,
        k_epochs=args.k_epochs,
        use_critic=args.use_critic,
    )
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 76)

    # ========== 2. 训练循环 ==========
    buffer = RolloutBuffer()
    eval_sizes = [int(x) for x in args.eval_sizes.split(",")]
    best_fc = float("inf")

    for it in range(1, args.iterations + 1):
        t0 = time.time()
        buffer.clear()

        # 采样 rollout
        for e in range(args.num_envs):
            seed = args.seed * 10000 + it * 100 + e
            G = build_ba_graph(args.n, m=args.m, seed=seed)
            sample_trajectory(model, G, args.stop_condition, device, buffer, action_k=args.action_k)

        # PPO 更新
        stats = trainer.update(buffer, batch_size=args.ppo_batch_size)
        dt = time.time() - t0

        print(f"[iter {it:4d}] steps={len(buffer):5d}  loss={stats.get('loss', 0):8.4f}  "
              f"actor={stats.get('actor_loss', 0):8.4f}  critic={stats.get('critic_loss', 0):8.4f}  "
              f"ent={stats.get('entropy', 0):7.4f}  ({dt:.1f}s)", flush=True)

        # 定期评测
        if it % args.eval_interval == 0:
            model.eval()
            rows = evaluate(model, device, eval_sizes, args.eval_seeds, m=args.m)
            model.train()

            avg_rl_stop = np.mean([r["rl_stop"] for r in rows])
            avg_rl_fc = np.mean([r["rl_fc"] for r in rows])
            avg_deg = np.mean([r["degree"] for r in rows])
            avg_corehd = np.mean([r["corehd"] for r in rows])
            print(f"  --- 评测 (n={eval_sizes}) ---")
            print(f"  degree={avg_deg:.1f}  CoreHD={avg_corehd:.1f}  "
                  f"rl-greedy stop={avg_rl_stop:.1f}  fc={avg_rl_fc:.4f}")

            # 保存检查点（按 fc 选最优）
            import os
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt = {
                "model_state_dict": model.state_dict(),
                "model_config": model.get_config(),
                "iteration": it,
                "rl_stop": float(avg_rl_stop),
                "rl_fc": float(avg_rl_fc),
            }
            torch.save(ckpt, f"{args.out_dir}/latest.pth")
            if avg_rl_fc < best_fc:
                best_fc = avg_rl_fc
                torch.save(ckpt, f"{args.out_dir}/best_model.pth")
                print(f"  ✓ 新最优 fc={best_fc:.4f} 已保存")

    print("\n" + "=" * 76)
    print(f"训练完成，best_fc={best_fc:.4f}，模型保存在 {args.out_dir}/")
    print("=" * 76)


if __name__ == "__main__":
    main()
