"""
Mamba 优先级评分模型
基于 mamba_ssm 实现序列编码和节点优先级评分
"""
import torch
import torch.nn as nn
from mamba_ssm import Mamba

# RTX 5060 (sm_120) 兼容性修复：使用 CPU/Triton 回退路径
# 参考：rtx5060-mamba-env.md 记忆文件
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
    import mamba_ssm.modules.mamba_simple as mamba_simple_module
    mamba_simple_module.selective_scan_fn = lambda *args, **kwargs: selective_scan_ref(*args, **kwargs)
except ImportError:
    pass  # 如果导入失败，使用默认路径


class MambaDismantlingModel(nn.Module):
    """
    Mamba 网络拆解模型

    使用 Mamba 块进行序列编码，输出每个节点的拆解优先级分数
    """

    def __init__(self, input_dim: int = 4, d_model: int = 64, n_layers: int = 2):
        """
        Parameters
        ----------
        input_dim : int
            输入特征维度（默认 4: 度、k-core、PageRank、接近中心性）
        d_model : int
            Mamba 隐藏层维度
        n_layers : int
            Mamba 层数
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.n_layers = n_layers

        # 输入投影层
        self.input_proj = nn.Linear(input_dim, d_model)

        # 堆叠 Mamba 块
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=16,  # SSM 状态维度
                d_conv=4,    # 卷积核大小
                expand=2,    # 扩展因子
                use_fast_path=False,  # RTX 5060 兼容：禁用 CUDA fast path
            )
            for _ in range(n_layers)
        ])

        # 输出层：将编码映射到单一优先级分数
        self.output_proj = nn.Linear(d_model, 1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """随机初始化权重"""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Parameters
        ----------
        x : torch.Tensor
            输入特征张量，形状 (batch_size, seq_len, input_dim)
            或 (seq_len, input_dim) 单样本

        Returns
        -------
        scores : torch.Tensor
            节点优先级分数，形状 (batch_size, seq_len) 或 (seq_len,)
        """
        # 处理单样本输入
        single_sample = False
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (seq_len, input_dim) -> (1, seq_len, input_dim)
            single_sample = True

        # 输入投影
        h = self.input_proj(x)  # (batch_size, seq_len, d_model)

        # 通过 Mamba 层编码
        for mamba_layer in self.mamba_layers:
            h = mamba_layer(h) + h  # 残差连接

        # 输出优先级分数
        scores = self.output_proj(h).squeeze(-1)  # (batch_size, seq_len)

        # 如果是单样本，移除 batch 维度
        if single_sample:
            scores = scores.squeeze(0)  # (seq_len,)

        return scores


def create_mamba_model(device: str = 'cuda') -> MambaDismantlingModel:
    """
    创建并初始化 Mamba 拆解模型

    Parameters
    ----------
    device : str
        设备 ('cuda' 或 'cpu')

    Returns
    -------
    model : MambaDismantlingModel
        初始化的模型实例
    """
    model = MambaDismantlingModel(
        input_dim=4,
        d_model=64,
        n_layers=2
    )

    model = model.to(device)
    model.eval()  # 推理模式

    return model
