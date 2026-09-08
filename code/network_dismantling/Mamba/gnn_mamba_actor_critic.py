"""
GNN + 双向 Mamba Actor-Critic 模型（RL / PPO）

在监督预训练的 GNNMambaModel 之上扩展：
    - 复用同一个 GCN + BiMamba 编码器（encode）
    - output_proj 作为「策略头」，输出每个节点的移除 logits（越高越优先移除）
    - 新增 value_head，把图级嵌入（节点嵌入 mean-pool）映射到状态价值 V(s)

从监督检查点初始化：state_dict 与 GNNMambaModel 兼容（多出的 value_head 用
strict=False 忽略，随机初始化），因此监督预训练可直接作为 PPO 初始策略。
"""
import torch
import torch.nn as nn

from .gnn_mamba_model import GNNMambaModel


class GNNMambaActorCritic(GNNMambaModel):
    """
    Actor（策略）+ Critic（价值）共享编码器。

    forward 返回 (logits, value)：
        - logits: (B, L) 每个节点的移除优先级（softmax 后作为采样分布）
        - value : (B,)   当前状态的价值估计（用于 advantage 的 baseline）
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 图级价值头：节点嵌入 mean-pool -> MLP -> 标量
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Tanh(),
            nn.Linear(self.d_model, 1),
        )
        # value_head 独立初始化（不用 xavier_uniform 覆盖，保留默认即可）
        for m in self.value_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, adj, mask=None):
        """
        Returns
        -------
        logits : torch.Tensor
            (B, L) 节点移除优先级 logits
        value : torch.Tensor
            (B,) 状态价值估计
        """
        h = self.encode(x, adj, mask)              # (B, L, d_model)
        logits = self.output_proj(h).squeeze(-1)   # (B, L)

        # 图级池化（屏蔽 padding）
        if mask is not None:
            mask_f = mask.float().unsqueeze(-1)    # (B, L, 1)
            pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        else:
            pooled = h.mean(dim=1)                 # (B, d_model)

        value = self.value_head(pooled).squeeze(-1)  # (B,)
        return logits, value


def create_actor_critic_from_supervised(
    checkpoint_path: str,
    device: str,
) -> GNNMambaActorCritic:
    """
    从监督预训练检查点构建 Actor-Critic 模型。

    - 加载 GNNMambaModel 的 encoder + output_proj 权重（作为初始策略）
    - value_head 随机初始化
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('model_config') or {}
    else:
        state_dict = checkpoint
        config = {}

    model = GNNMambaActorCritic(**config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # 预期只缺 value_head（随机初始化即可）
    model = model.to(device)
    model.train()  # RL 需要梯度
    return model
