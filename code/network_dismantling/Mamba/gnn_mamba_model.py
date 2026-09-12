"""
GNN + 双向 Mamba 网络拆解模型

架构（阶段 B 重构）：
    输入: 节点度特征 (B, L, 1) + 邻接矩阵 (B, L, L)
    1. GCN 编码器: 用邻接矩阵做消息传递，得到节点嵌入（补上纯序列模型缺失的拓扑信息）
    2. 双向 Mamba: 对按度排序的节点嵌入序列做双向序列编码（前向 + 反向）
    3. 输出层: 映射到单一优先级分数

与旧 MambaDismantlingModel 的核心区别：
    - 显式使用邻接矩阵（GNN 消息传递），旧模型只把节点当一维序列、完全丢失边信息
    - 输入特征只用度（degree），删除 k-core / PageRank / closeness 中心性特征
    - 双向 Mamba（BiMamba），旧模型只有单向
"""
import torch
import torch.nn as nn
from mamba_ssm import Mamba

# RTX 5060 (sm_120) 兼容性修复：使用 CPU/Triton 回退路径（同 mamba_model.py）
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
    import mamba_ssm.modules.mamba_simple as mamba_simple_module
    mamba_simple_module.selective_scan_fn = lambda *args, **kwargs: selective_scan_ref(*args, **kwargs)
except ImportError:
    pass


class GCNLayer(nn.Module):
    """
    基于 dense 邻接矩阵的图卷积层（Kipf & Welling GCN）

    消息传递: h' = act( LayerNorm( D^{-1/2} (A + I) D^{-1/2} h W ) )
    采用 batch 内 block-diagonal 邻接矩阵，用 torch.bmm 一次处理整个 batch。
    适用于 500-1500 节点规模（dense 邻接矩阵内存可承受），不依赖 PyG/DGL。
    """

    def __init__(self, in_dim: int, out_dim: int, use_norm: bool = True, use_residual: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if use_norm else None
        self.act = nn.GELU()
        self.use_residual = use_residual
        self.residual = nn.Linear(in_dim, out_dim) if (use_residual and in_dim != out_dim) else None

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            节点特征 (B, L, in_dim)
        adj_norm : torch.Tensor
            对称归一化邻接矩阵（含 self-loop）(B, L, L)
        """
        h = torch.bmm(adj_norm, x)          # 聚合邻居 -> (B, L, in_dim)
        h = self.lin(h)                      # -> (B, L, out_dim)
        if self.norm is not None:
            h = self.norm(h)
        h = self.act(h)

        if self.use_residual:
            res = self.residual(x) if self.residual is not None else x
            h = h + res
        return h


class GNNMambaModel(nn.Module):
    """
    GNN 编码 + 双向 Mamba 拆解评分模型

    Parameters
    ----------
    input_dim : int
        输入特征维度（degree-only 时为 1）
    hidden_dim : int
        GCN 隐藏层维度
    d_model : int
        Mamba 隐藏层维度（GNN 输出投影到该维度）
    n_gnn_layers : int
        GCN 层数
    n_mamba_layers : int
        每个方向（前向/反向）的 Mamba 层数
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        d_model: int = 64,
        n_gnn_layers: int = 2,
        n_mamba_layers: int = 2,
        seq_model: str = "mamba",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.n_gnn_layers = n_gnn_layers
        self.n_mamba_layers = n_mamba_layers
        # 序列模型类型：
        #   'mamba'     : 双向 Mamba（本项目方法，O(N) 线性复杂度）
        #   'attention' : 标准多头自注意力（Transformer 式，O(N²)）+ FFN
        #   当 n_mamba_layers=0 时二者都不建，退化为纯 GCN
        # 这个开关用于验证方法的核心论断：Mamba 的线性复杂度在大规模网络上
        # 相对注意力的扩展性优势（显存/耗时随 N 的增长）。
        if seq_model not in ("mamba", "attention"):
            raise ValueError(f"seq_model 必须是 'mamba' 或 'attention'，当前 '{seq_model}'")
        self.seq_model = seq_model

        # ---- GCN 编码器 ----
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GCNLayer(input_dim, hidden_dim))
        for _ in range(n_gnn_layers - 1):
            self.gnn_layers.append(GCNLayer(hidden_dim, hidden_dim))
        self.gnn_head = nn.Linear(hidden_dim, d_model)

        # ---- 双向 Mamba ----
        def _make_mamba():
            return Mamba(
                d_model=d_model,
                d_state=16,
                d_conv=4,
                expand=2,
                use_fast_path=False,  # RTX 5060 兼容
            )

        # n_mamba_layers=0 -> 纯 GCN 模型（消融用：验证序列模型到底有没有贡献）
        self.mamba_fwd = nn.ModuleList()
        self.mamba_bwd = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        if n_mamba_layers > 0:
            if seq_model == "mamba":
                self.mamba_fwd = nn.ModuleList([_make_mamba() for _ in range(n_mamba_layers)])
                self.mamba_bwd = nn.ModuleList([_make_mamba() for _ in range(n_mamba_layers)])
            else:
                # 自注意力本身双向（不加因果掩码），与双向 Mamba 对齐；
                # dim_feedforward=2*d_model 与 Mamba 的 expand=2 量级对齐，
                # 使两者的参数量与计算量可比。
                self.attn_layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=d_model, nhead=4, dim_feedforward=2 * d_model,
                        dropout=0.0, batch_first=True, norm_first=True,
                    )
                    for _ in range(n_mamba_layers)
                ])

        # ---- 输出层 ----
        self.output_proj = nn.Linear(d_model, 1)

        self._init_weights()

    def get_config(self) -> dict:
        """返回模型配置，用于 checkpoint 保存与重建"""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'd_model': self.d_model,
            'n_gnn_layers': self.n_gnn_layers,
            'n_mamba_layers': self.n_mamba_layers,
            'seq_model': self.seq_model,
        }

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        """
        对称归一化邻接矩阵（含 self-loop）：
            A_norm = D^{-1/2} (A + I) D^{-1/2}
        padding 节点只有 self-loop，度 >= 1，不会除零。
        """
        L = adj.size(1)
        eye = torch.eye(L, device=adj.device, dtype=adj.dtype).unsqueeze(0)  # (1, L, L)
        adj_loop = adj + eye
        deg = adj_loop.sum(dim=-1).clamp(min=1.0)      # (B, L)
        deg_inv_sqrt = deg.pow(-0.5)                    # (B, L)
        return deg_inv_sqrt.unsqueeze(-1) * adj_loop * deg_inv_sqrt.unsqueeze(1)

    def encode(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        GCN + 双向 Mamba 编码器，返回节点嵌入（供评分/RL 策略共享复用）。

        Parameters
        ----------
        x : torch.Tensor
            节点特征 (B, L, input_dim)，按度降序排列
        adj : torch.Tensor
            邻接矩阵 (B, L, L)，与 x 同一节点顺序（0/1，无 self-loop）
        mask : torch.Tensor, optional
            有效位置掩码 (B, L)，True=有效节点，False=padding

        Returns
        -------
        h : torch.Tensor
            节点嵌入 (B, L, d_model)
        """
        adj_norm = self._normalize_adj(adj)

        # GCN 编码
        h = x
        for layer in self.gnn_layers:
            h = layer(h, adj_norm)
        h = self.gnn_head(h)  # (B, L, d_model)

        # 屏蔽 padding 节点（归零，避免污染序列传播）
        if mask is not None:
            h = h * mask.float().unsqueeze(-1)

        # 消融：n_mamba_layers=0 -> 纯 GCN 编码（不接序列模型）
        if self.n_mamba_layers == 0:
            return h

        # 自注意力路径（O(N²)，Transformer 式；本身双向，无需 flip）
        if self.seq_model == "attention":
            # 用 key_padding_mask 让 padding 不参与注意力（比 Mamba 路径更干净）
            pad_mask = (~mask) if mask is not None else None
            hh = h
            for layer in self.attn_layers:
                hh = layer(hh, src_key_padding_mask=pad_mask)
            return hh

        # 前向 Mamba
        h_fwd = h
        for layer in self.mamba_fwd:
            h_fwd = layer(h_fwd) + h_fwd

        # 反向 Mamba（flip 序列 -> 编码 -> flip 回来）
        h_bwd = torch.flip(h, dims=[1])
        for layer in self.mamba_bwd:
            h_bwd = layer(h_bwd) + h_bwd
        h_bwd = torch.flip(h_bwd, dims=[1])

        return h_fwd + h_bwd  # (B, L, d_model)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            节点特征 (B, L, input_dim)，按度降序排列
        adj : torch.Tensor
            邻接矩阵 (B, L, L)，与 x 同一节点顺序（0/1，无 self-loop）
        mask : torch.Tensor, optional
            有效位置掩码 (B, L)，True=有效节点，False=padding

        Returns
        -------
        scores : torch.Tensor
            节点优先级分数 (B, L)
        """
        h = self.encode(x, adj, mask)
        scores = self.output_proj(h).squeeze(-1)  # (B, L)
        return scores


def create_gnn_mamba_model(device: str = 'cuda', **kwargs) -> GNNMambaModel:
    """创建并初始化 GNN + 双向 Mamba 模型（推理模式）"""
    model = GNNMambaModel(**kwargs)
    model = model.to(device)
    model.eval()
    return model
