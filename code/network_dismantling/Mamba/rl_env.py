"""
网络拆解 RL 环境与 rollout 采样

把「最小化拆解集大小」建模成序列决策问题：
    - 状态 s_t    : 当前剩余子图（节点度特征 + 邻接矩阵）
    - 动作 a_t    : 从剩余节点中采样 k 个节点移除（批动作，sequential without replacement）
    - 奖励 r_t    : -(k + LCC_after / n0)，同时惩罚「多移除 k 个节点」与「剩余 LCC 占比」
    - 终止        : 最大连通分量 LCC <= stop_condition

批动作动机：每步移除 k 个节点（而非 1 个），episode 步数除以 k，从而把「Mamba 前向
次数」降到原来的 1/k——这是 RTX 5060 上 Mamba 走 CPU 回退、单次前向昂贵的前提下的
关键性能优化（与动态重算 batch_size 旋钮一致）。

采样时用旧策略（torch.no_grad）跑 rollout，记录每一步的 (特征, 邻接, 动作位置列表,
奖励, 旧 log_prob, 旧 value)，供 PPO 阶段重新前向计算新策略 log_prob / value。
"""
from typing import List, Optional

import networkx as nx
import numpy as np
import torch

from .feature_encoding import extract_node_features


def lcc_size(G: nx.Graph) -> int:
    """最大连通分量大小"""
    if G.number_of_nodes() == 0:
        return 0
    return max(len(c) for c in nx.connected_components(G))


def graph_stats(G: nx.Graph, n0: int):
    """
    一次遍历所有分量，同时返回 (最大分量大小, 碎片化指数)。

    碎片化指数 P = Σ_i (s_i / n0)²
        - 完全连通时 P = 1（只有一个分量，s=n0）
        - 完全拆散成孤立点时 P = 1/n0 → 0
        - 它满足 Σ s_i = n0 恒成立，因此**分量越分散、P 越小**

    为什么要它：fc_value（LCC ≤ 1%）要求**所有**分量都小于 1%·n0，而只看
    最大分量的信号（LCC）在"图已碎成若干中等分量"时是平的——比如 5 个 20 节点
    的分量（n0=500）时 LCC=20 恒不变，reward 连续多步为 0，策略完全没有动力
    去拆散这些中等分量。P 对**任何**分量的破裂都下降，恰好补上这一段梯度。

    合并成一次遍历是为了省开销：原实现每步调用两次 lcc_size（移除前后），
    各走一遍 connected_components；这里一次拿到 LCC 与 P，调用两次，总开销不变。
    """
    if G.number_of_nodes() == 0:
        return 0, 0.0
    sizes = np.fromiter(
        (len(c) for c in nx.connected_components(G)), dtype=np.float64
    )
    return int(sizes.max()), float(np.sum((sizes / n0) ** 2))


class RolloutBuffer:
    """聚合一条 episode 每一步的采样数据，供 PPO 更新重放"""

    def __init__(self):
        self.features: List[np.ndarray] = []   # 每个 (n', 1) 度特征（度降序）
        self.adjs: List[np.ndarray] = []       # 每个 (n', n') 邻接矩阵
        self.action_positions: List[List[int]] = []  # 每个 step 的 k 个动作位置
        self.rewards: List[float] = []
        self.potentials: List[float] = []   # 每个 step 的 (lcc_before-lcc_after)/n0 即时信号
        self.lcc_fracs: List[float] = []    # 每个 step 移除后的 LCC/n0（advantage='lcc' 用）
        self.frag_potentials: List[float] = []  # 碎片化增益 P_before - P_after（advantage='frag' 用）
        self.log_probs_old: List[torch.Tensor] = []  # 标量 tensor（k 个节点 log_prob 之和）
        self.values_old: List[torch.Tensor] = []     # 标量 tensor
        self.dones: List[bool] = []

    def add(self, features, adj, action_positions, reward, log_prob, value, done,
            potential, lcc_frac, frag_potential):
        self.features.append(features)
        self.adjs.append(adj)
        self.action_positions.append(action_positions)
        self.rewards.append(reward)
        self.potentials.append(potential)
        self.lcc_fracs.append(lcc_frac)
        self.frag_potentials.append(frag_potential)
        self.log_probs_old.append(log_prob.detach())
        self.values_old.append(value.detach())
        self.dones.append(done)

    def clear(self):
        self.features.clear()
        self.adjs.clear()
        self.action_positions.clear()
        self.rewards.clear()
        self.potentials.clear()
        self.lcc_fracs.clear()
        self.frag_potentials.clear()
        self.log_probs_old.clear()
        self.values_old.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.rewards)


def _score_subgraph(model, G_std, device):
    """
    在标准化子图上做一次前向，返回 (features, adj, node_ids, logits, value)。
    node_ids 为度降序排列的 G_std 节点 ID，logits 的顺序与其一一对应。
    """
    # 序列排序由模型自带（与训练/监督预训练一致）
    order = getattr(model, 'order', 'degree')
    features, node_ids = extract_node_features(
        G_std, feature_set='degree', order=order
    )  # (n,1)
    adj = nx.to_numpy_array(G_std, dtype=np.uint8)
    adj = adj[np.ix_(node_ids, node_ids)]
    with torch.no_grad():
        x = torch.from_numpy(features).float().to(device).unsqueeze(0)  # (1,n,1)
        a = torch.from_numpy(adj).float().to(device).unsqueeze(0)       # (1,n,n)
        logits, value = model(x, a)  # (1,n), (1,)
    return features, adj, node_ids, logits.squeeze(0), value.squeeze(0)


def _sample_k(logits: torch.Tensor, k: int):
    """
    从 logits 采样 k 个不同节点（sequential without replacement）。
    返回 (action_positions: List[int], log_prob_sum: 标量 tensor)。
    """
    n = logits.shape[0]
    k = min(k, n)
    action_positions = []
    log_prob_sum = torch.zeros((), device=logits.device, dtype=logits.dtype)
    remaining_mask = torch.ones(n, dtype=torch.bool, device=logits.device)
    for _ in range(k):
        masked = logits.masked_fill(~remaining_mask, -1e9)
        dist = torch.distributions.Categorical(logits=masked)
        pos = dist.sample()
        log_prob_sum = log_prob_sum + dist.log_prob(pos)
        action_positions.append(pos.item())
        remaining_mask[pos] = False
    return action_positions, log_prob_sum


def _log_prob_of(logits: torch.Tensor, action_positions: List[int]):
    """按 sequential without replacement 规则计算给定动作序列的 log_prob（可微分）"""
    log_prob_sum = torch.zeros((), device=logits.device, dtype=logits.dtype)
    remaining_mask = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
    for pos in action_positions:
        masked = logits.masked_fill(~remaining_mask, -1e9)
        dist = torch.distributions.Categorical(logits=masked)
        log_prob_sum = log_prob_sum + dist.log_prob(torch.tensor(pos, device=logits.device))
        remaining_mask[pos] = False
    return log_prob_sum


def sample_trajectory(
    model,
    G: nx.Graph,
    stop_condition: int,
    device: str,
    buffer: RolloutBuffer,
    n0: Optional[int] = None,
    action_k: int = 10,
) -> int:
    """
    用当前策略在单张图上采样一条 episode，数据写入 buffer。

    Returns
    -------
    removed_count : int
        本 episode 到达终止条件时移除的节点数（stop_step）
    """
    n0 = n0 or G.number_of_nodes()
    G_tmp = G.copy()
    start_len = len(buffer)

    while G_tmp.number_of_nodes() > 0:
        node_list = list(G_tmp.nodes())
        G_std = nx.relabel_nodes(G_tmp, {v: i for i, v in enumerate(node_list)})

        features, adj, node_ids, logits, value = _score_subgraph(model, G_std, device)

        action_positions, log_prob = _sample_k(logits, action_k)

        # 移除前的图统计（potential-based shaping 需要）
        lcc_before, frag_before = graph_stats(G_tmp, n0)

        # 移除这 k 个节点
        for pos in action_positions:
            removed_orig = node_list[node_ids[pos]]
            G_tmp.remove_node(removed_orig)

        k_removed = len(action_positions)
        lcc, frag_after = graph_stats(G_tmp, n0)
        done = lcc <= stop_condition
        # potential-based reward shaping：-步数 + 势函数下降量。
        # 累积回报 = -stop_step + (Φ_0 - Φ_final)，不改变最优策略，但每步都有即时信号。
        #
        # 两种势函数，由 --advantage 选择（见 ppo_trainer）：
        #   potential : Φ = LCC/n0          -> 只奖励"最大分量变小"
        #   frag      : Φ = Σ(s_i/n0)²      -> 奖励"任何分量破裂"，包括拆散中等分量
        # frag 是为 fc_value（LCC ≤ 1%，要求**所有**分量都小）准备的：
        # 图碎成若干中等分量后 LCC 不再下降，potential 信号归零、策略失去动力，
        # 而 frag 仍然下降，继续提供梯度。
        potential = (lcc_before - lcc) / n0
        frag_potential = frag_before - frag_after
        lcc_frac = lcc / n0
        reward = -float(k_removed) + potential

        buffer.add(features, adj, action_positions, reward, log_prob, value, done,
                   potential, lcc_frac, frag_potential)

        if done:
            break

    return len(buffer) - start_len
