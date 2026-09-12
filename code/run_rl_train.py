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
from network_dismantling.Mamba.graph_factory import sample_graph
from network_dismantling.unified_interface import dismantle
from network_dismantling.Mamba.model_io import unregister_model
from evaluate import calc_metrics, calc_all_metrics, complete_sequence


def build_rl_graph(n: int, graph_type: str = "ba", seed: int = 0) -> nx.Graph:
    """
    生成 RL 训练用的图。

    graph_type='mixed' 时每次随机挑一种拓扑（BA/ER/WS），参数从 graph_factory
    的范围里采样——与监督训练集 datasets/mixed_gnn 的分布保持一致，
    否则 RL 微调会把监督阶段学到的跨拓扑能力又拉回 BA 上。
    """
    rng = np.random.default_rng(seed)
    topo = str(rng.choice(["ba", "er", "ws"])) if graph_type == "mixed" else graph_type
    G, _ = sample_graph((n, n), topo, rng)
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
        features, node_ids = extract_node_features(
            G_std, feature_set='degree', order=getattr(model, 'order', 'degree')
        )
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
    # 补齐到 n（与 dismantle() 的 _fill_remaining 一致），否则算 AUC 时
    # 曲线被截断会虚低，与 CoreHD/degree 不可比
    return complete_sequence(G, seq)


def evaluate(model, device, n_sizes, seeds_per_size, graph_type="ba"):
    """
    贪心策略评测：对比 degree / CoreHD@0.01n / rl-greedy 的 AUC / stop_step / fc_value。

    AUC 为主指标（见 evaluate.py 的口径说明）。基线用 CoreHD@0.01n——
    把 CoreHD 的 Sthreshold 对齐到 fc 目标才算同条件对比，用 stop_condition=1
    是在跟一个被调弱的 CoreHD 比。
    """
    unregister_model()
    rows = []
    for n in n_sizes:
        for s in range(seeds_per_size):
            seed = 1000 * n + s
            G = build_rl_graph(n, graph_type=graph_type, seed=seed)
            n_nodes = G.number_of_nodes()
            stop_cond = max(1, int(0.01 * n_nodes))

            row = {"n": n_nodes, "seed": seed}
            # 基线（CoreHD 的 seed 由 dismantle 默认注入 0，保证可复现）
            for name, kwargs in [
                ("degree", {"method": "degree"}),
                ("corehd", {"method": "CoreHD", "stop_condition": stop_cond}),
            ]:
                met = calc_all_metrics(G, dismantle(G, **kwargs))
                row[f"{name}_auc"] = round(met["auc"], 4)
                row[f"{name}_stop"] = met["stop_step"]

            # RL 贪心（序列已在 greedy_sequence 内补齐到 n）
            met = calc_all_metrics(G, greedy_sequence(model, G, device, stop_cond))
            row["rl_auc"] = round(met["auc"], 4)
            row["rl_stop"] = met["stop_step"]
            row["rl_fc"] = round(met["fc_value"], 4)
            rows.append(row)
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
    parser.add_argument("--m", type=int, default=3, help="(已废弃，参数由 graph_factory 采样)")
    parser.add_argument("--graph-type", type=str, default="ba",
                        choices=["ba", "er", "ws", "mixed"],
                        help="训练图拓扑；mixed = 每次随机选一种（与监督训练集分布一致）")
    parser.add_argument("--advantage", type=str, default="potential",
                        choices=["potential", "lcc", "frag"],
                        help="per-step advantage 形式：\n"
                             "  potential = (LCC_before-LCC_after)/n0（默认，只奖励最大分量变小）\n"
                             "  lcc       = -LCC_after/n0（A/B 对照）\n"
                             "  frag      = Σ(s_i/n0)² 的下降量（奖励任何分量破裂，"
                             "针对 fc_value 短板——图碎成中等分量后 potential 信号归零）")
    parser.add_argument("--num-envs", type=int, default=8, help="每次迭代采样的图数")
    parser.add_argument("--action-k", type=int, default=10, help="每步移除的节点数（批动作）")
    parser.add_argument("--iterations", type=int, default=200, help="迭代次数")
    parser.add_argument("--stop-condition", type=int, default=0,
                        help="RL 训练终止 LCC 阈值；0 = 自动取 0.01*n（与评测 greedy "
                             "和 fc_value 的目标对齐，避免训练/评测目标不一致）")
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

    # stop-condition=0 -> 自动对齐到 0.01n（与评测 greedy / fc_value 目标一致）
    stop_cond = args.stop_condition or max(1, int(0.01 * args.n))

    print("=" * 76)
    print("RL (PPO) 网络拆解训练")
    print("=" * 76)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"初始策略: {args.ckpt}")
    print(f"训练图: {args.graph_type.upper()} n={args.n}  终止 LCC 阈值={stop_cond}")
    print(f"advantage: {args.advantage}")
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
        advantage=args.advantage,
    )
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 76)

    # ========== 2. 训练循环 ==========
    buffer = RolloutBuffer()
    eval_sizes = [int(x) for x in args.eval_sizes.split(",")]
    best_auc = float("inf")

    for it in range(1, args.iterations + 1):
        t0 = time.time()
        buffer.clear()

        # 采样 rollout
        for e in range(args.num_envs):
            seed = args.seed * 10000 + it * 100 + e
            G = build_rl_graph(args.n, graph_type=args.graph_type, seed=seed)
            sample_trajectory(model, G, stop_cond, device, buffer, action_k=args.action_k)

        # PPO 更新
        stats = trainer.update(buffer, batch_size=args.ppo_batch_size)
        dt = time.time() - t0

        print(f"[iter {it:4d}] steps={len(buffer):5d}  loss={stats.get('loss', 0):8.4f}  "
              f"actor={stats.get('actor_loss', 0):8.4f}  critic={stats.get('critic_loss', 0):8.4f}  "
              f"ent={stats.get('entropy', 0):7.4f}  ({dt:.1f}s)", flush=True)

        # 定期评测（AUC 为主指标，与基线同条件对比）
        if it % args.eval_interval == 0:
            model.eval()
            rows = evaluate(model, device, eval_sizes, args.eval_seeds,
                            graph_type=args.graph_type)
            model.train()

            avg_rl_auc = np.mean([r["rl_auc"] for r in rows])
            avg_rl_stop = np.mean([r["rl_stop"] for r in rows])
            avg_rl_fc = np.mean([r["rl_fc"] for r in rows])
            avg_deg = np.mean([r["degree_auc"] for r in rows])
            avg_corehd = np.mean([r["corehd_auc"] for r in rows])
            print(f"  --- 评测 (n={eval_sizes}, {args.graph_type}) ---")
            print(f"  AUC:  degree={avg_deg:.4f}  CoreHD@0.01n={avg_corehd:.4f}  "
                  f"rl-greedy={avg_rl_auc:.4f}")
            print(f"  stop: degree={np.mean([r['degree_stop'] for r in rows]):.1f}  "
                  f"CoreHD={np.mean([r['corehd_stop'] for r in rows]):.1f}  "
                  f"rl={avg_rl_stop:.1f}   fc(rl)={avg_rl_fc:.4f}")

            # 保存检查点（**按 AUC 选最优**——AUC 是主指标，模型选择准则必须跟它对齐；
            # 之前按 fc 选会让 A/B 对比被选择偏差污染）
            import os
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt = {
                "model_state_dict": model.state_dict(),
                "model_config": model.get_config(),
                "order": getattr(model, "order", "degree"),
                "iteration": it,
                "rl_auc": float(avg_rl_auc),
                "rl_stop": float(avg_rl_stop),
                "rl_fc": float(avg_rl_fc),
            }
            torch.save(ckpt, f"{args.out_dir}/latest.pth")
            if avg_rl_auc < best_auc:
                best_auc = avg_rl_auc
                torch.save(ckpt, f"{args.out_dir}/best_model.pth")
                print(f"  ✓ 新最优 AUC={best_auc:.4f} 已保存")

    print("\n" + "=" * 76)
    print(f"训练完成，best_auc={best_auc:.4f}，模型保存在 {args.out_dir}/")
    print("=" * 76)


if __name__ == "__main__":
    main()
