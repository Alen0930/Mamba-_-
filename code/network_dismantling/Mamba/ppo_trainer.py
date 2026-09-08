"""
PPO（Proximal Policy Optimization）训练器

在动态决策框架上做策略梯度微调：
    - 策略   : GNNMambaActorCritic 的 logits -> softmax -> 节点采样分布
    - 价值   : value_head 作为 baseline 降低方差
    - 优势   : GAE（广义优势估计）
    - 目标   : clipped surrogate objective + value MSE - 熵正则

监督预训练（CoreHD 标签）模型直接作为初始策略，PPO 用真实拆解 reward
（LCC 下降 / 步数）替换「拟合标签」，从而突破行为克隆上界。
"""
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rl_env import RolloutBuffer, _log_prob_of


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
):
    """
    广义优势估计（GAE）。

    dones 标记每条 episode 的终止步；终止后未来价值不传播，GAE 自动在
    episode 边界处重置。
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def compute_returns(rewards: np.ndarray, dones: np.ndarray, gamma: float = 0.99):
    """
    纯蒙特卡洛折扣回报（无 critic baseline），用于 REINFORCE。

    与 GAE 的区别：不依赖 value 预测，只按真实奖励累积。当 critic 学不好
    （value 预测误差与 return 同量级）时，GAE 的优势估计退化成噪声，反而
    破坏监督初始化的策略；此时用真实 MC 回报 + batch 标准化 baseline 更稳。
    """
    returns = np.zeros_like(rewards, dtype=np.float32)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g * (1.0 - dones[t])
        returns[t] = g
    return returns


class PPOTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        gamma: float = 0.99,
        lam: float = 0.95,
        k_epochs: int = 3,
        max_grad_norm: float = 0.5,
        use_critic: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.gamma = gamma
        self.lam = lam
        self.k_epochs = k_epochs
        self.max_grad_norm = max_grad_norm
        self.use_critic = use_critic

    def update(self, buffer: RolloutBuffer, batch_size: int = 16) -> Dict[str, float]:
        """
        用 buffer 里的 rollout 数据做一次 PPO 更新（多 epoch，mini-batch 批处理）。

        批处理：把不同长度（不同剩余节点数）的 step pad 到批内 L_max，
        一次前向/反向处理 batch_size 个 step——这是 Mamba 走 CPU 回退、单次前向
        昂贵的前提下，把更新阶段加速数倍的关键。
        """
        T = len(buffer)
        if T == 0:
            return {}

        rewards = np.array(buffer.rewards, dtype=np.float32)
        dones = np.array(buffer.dones, dtype=np.float32)
        log_probs_old = torch.stack(buffer.log_probs_old).to(self.device)

        if self.use_critic:
            values_old = torch.stack(buffer.values_old).cpu().numpy()
            advantages, returns = compute_gae(rewards, values_old, dones, self.gamma, self.lam)
            advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        else:
            # per-step 即时 advantage：potential 变化 (lcc_before-lcc_after)/n0 标准化。
            # 能区分「移除关键节点→LCC骤降」的好 step 与「移除无关节点→LCC不变」的差 step；
            # 比整条 return（REINFORCE）精细，比 critic（GAE）稳（不依赖 value 估计）。
            # 注：-k 常数项对所有动作无区分度，故不进入 advantage，标准化后自然消失。
            potentials = np.array(buffer.potentials, dtype=np.float32)
            advantages = torch.tensor(potentials, dtype=torch.float32, device=self.device)
            returns = advantages

        # 优势标准化（降低梯度方差）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "loss": 0.0}
        n_updates = 0

        for _ in range(self.k_epochs):
            indices = np.random.permutation(T)
            for start in range(0, T, batch_size):
                batch_idx = indices[start:start + batch_size]
                B = len(batch_idx)
                L_max = max(buffer.features[i].shape[0] for i in batch_idx)

                # pad 到批内 L_max
                x_batch = torch.zeros(B, L_max, 1, device=self.device)
                adj_batch = torch.zeros(B, L_max, L_max, device=self.device)
                mask_batch = torch.zeros(B, L_max, dtype=torch.bool, device=self.device)
                for j, i in enumerate(batch_idx):
                    n_i = buffer.features[i].shape[0]
                    x_batch[j, :n_i, :] = torch.from_numpy(buffer.features[i]).float().to(self.device)
                    adj_batch[j, :n_i, :n_i] = torch.from_numpy(buffer.adjs[i]).float().to(self.device)
                    mask_batch[j, :n_i] = True

                logits, value = self.model(x_batch, adj_batch, mask_batch)  # (B,L_max), (B,)

                # 每个 step 在有效位置（前 n_i）上算新 log_prob 与熵
                log_probs_new_b, entropies_b = [], []
                for j, i in enumerate(batch_idx):
                    n_i = buffer.features[i].shape[0]
                    li = logits[j, :n_i]
                    log_probs_new_b.append(_log_prob_of(li, buffer.action_positions[i]))
                    entropies_b.append(torch.distributions.Categorical(logits=li).entropy())
                log_probs_new_b = torch.stack(log_probs_new_b)   # (B,)
                entropy = torch.stack(entropies_b).mean()        # 标量

                adv_b = advantages[batch_idx]
                ret_b = returns[batch_idx]
                lpo_b = log_probs_old[batch_idx]

                # PPO clipped surrogate objective
                ratio = torch.exp(log_probs_new_b - lpo_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()
                if self.use_critic:
                    critic_loss = F.mse_loss(value, ret_b)
                    loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
                else:
                    critic_loss = torch.zeros((), device=self.device)
                    loss = actor_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                stats["actor_loss"] += actor_loss.item()
                stats["critic_loss"] += critic_loss.item()
                stats["entropy"] += entropy.item()
                stats["loss"] += loss.item()
                n_updates += 1

        for k in stats:
            stats[k] /= max(n_updates, 1)
        stats["mean_advantage"] = advantages.mean().item()
        return stats
