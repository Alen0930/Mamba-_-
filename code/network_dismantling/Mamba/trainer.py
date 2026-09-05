"""
Mamba 网络拆解模型训练引擎

实现 ListMLE 排序损失，支持完整训练流程、验证评估和模型保存
"""
import os
import time
import logging
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import networkx as nx

from .mamba_model import MambaDismantlingModel
from .feature_encoding import extract_node_features

logger = logging.getLogger(__name__)


# ============================================================================
# ListMLE 排序损失函数
# ============================================================================

class ListMLELoss(nn.Module):
    """
    ListMLE (List Maximum Likelihood Estimation) 排序损失

    基于排列概率的学习排序损失函数，最大化正确排序的似然概率。

    参考文献:
    Xia et al. "Listwise approach to learning to rank: theory and algorithm." ICML 2008.
    """

    def __init__(self, epsilon: float = 1e-10):
        """
        Parameters
        ----------
        epsilon : float
            数值稳定性常数，避免 log(0)
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        pred_scores: torch.Tensor,
        true_ranks: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算 ListMLE 损失（向量化实现）

        Parameters
        ----------
        pred_scores : torch.Tensor
            模型预测分数，形状 (batch_size, seq_len)
            分数越高表示节点越应该被优先移除
        true_ranks : torch.Tensor
            真实排序标签，形状 (batch_size, seq_len)
            取值范围 [0, seq_len-1]，0 表示最优先移除，seq_len-1 表示最后移除
        mask : torch.Tensor, optional
            有效位置掩码 (batch_size, seq_len)，padding 位置为 False。
            提供时屏蔽 padding 元素的损失贡献。

        Returns
        -------
        loss : torch.Tensor
            标量损失值

        数学原理
        ---------
        L = -sum_i [s_i - logsumexp(s[j:])]   (s 已按真实排序重排)
        利用 logcumsumexp 的翻转实现后缀 logsumexp，完全向量化，
        与逐位置循环实现数学等价（除以 batch_size 归一化）。
        """
        batch_size, seq_len = pred_scores.shape

        # 验证输入
        assert true_ranks.shape == pred_scores.shape, \
            f"Shape mismatch: pred_scores {pred_scores.shape}, true_ranks {true_ranks.shape}"

        # 按真实排序重排预测分数
        sorted_indices = torch.argsort(true_ranks, dim=1)  # (B, L)
        s = torch.gather(pred_scores, 1, sorted_indices)   # (B, L)

        # 屏蔽 padding 位置（置 -inf 使其不影响 logsumexp 累积）
        # 注意: mask 必须按 sorted_indices 同步重排（与 s 对齐）
        mask_sorted = None
        if mask is not None:
            mask_sorted = torch.gather(mask, 1, sorted_indices)
            s = s.masked_fill(~mask_sorted, float("-inf"))

        # 后缀 logsumexp: lse_from[i] = logsumexp(s[i:])
        # 翻转 -> 前缀 logcumsumexp -> 翻转回来
        lse_from = torch.logcumsumexp(s.flip(1), dim=1).flip(1)  # (B, L)

        # 每位置贡献: lse_from[i] - s[i]（负对数似然的元素项）
        diff = lse_from - s
        if mask_sorted is not None:
            diff = diff.masked_fill(~mask_sorted, 0.0)  # 屏蔽 padding（-inf - -inf = nan）

        return diff.sum() / batch_size


# ============================================================================
# 数据集类
# ============================================================================

class DismantlingDataset(Dataset):
    """
    网络拆解数据集

    每个样本包含：
    - 图的节点特征序列
    - 节点 ID 映射
    - 真实拆解排序（由 CoreHD 或其他算法生成）
    """

    def __init__(
        self,
        graphs: List[nx.Graph],
        dismantler_fn: Callable[[nx.Graph], List[int]],
        cache_features: bool = True
    ):
        """
        Parameters
        ----------
        graphs : List[nx.Graph]
            训练图列表
        dismantler_fn : Callable
            拆解算法函数，输入图，输出拆解序列（原始节点ID）
            例如: lambda G: dismantle(G, method='corehd')
        cache_features : bool
            是否缓存特征和标签（加速训练，但消耗内存）
        """
        self.graphs = graphs
        self.dismantler_fn = dismantler_fn
        self.cache_features = cache_features

        # 缓存
        self.cached_data = None
        if cache_features:
            logger.info("预处理数据集并缓存...")
            self.cached_data = self._preprocess_all()
            logger.info(f"缓存完成，共 {len(self.cached_data)} 个样本")

    def _preprocess_all(self) -> List[Dict]:
        """预处理所有图，生成特征和标签"""
        data = []
        for idx, G in enumerate(self.graphs):
            try:
                sample = self._process_graph(G)
                data.append(sample)
            except Exception as e:
                logger.warning(f"处理图 {idx} 失败: {e}，跳过")
        return data

    def _process_graph(self, G: nx.Graph) -> Dict:
        """
        处理单个图，生成特征和标签

        Returns
        -------
        sample : Dict
            包含 'features', 'node_ids', 'ranks' 三个键
        """
        # 标准化图（节点重标记为 0..n-1）
        G_std = self._standardize_graph(G)

        # 提取特征序列
        features, node_ids = extract_node_features(G_std)

        # 生成真实拆解序列（标准化节点ID）
        dismantling_seq = self.dismantler_fn(G_std)

        # 将拆解序列转换为排序标签
        # dismantling_seq[i] 是第 i 个被移除的节点
        # 我们需要为每个节点分配排序位置
        n = G_std.number_of_nodes()
        ranks = np.zeros(n, dtype=np.int64)
        for rank, node_id in enumerate(dismantling_seq):
            if node_id < n:  # 防止越界
                ranks[node_id] = rank

        # 按特征序列顺序重排 ranks
        # node_ids[i] 是特征序列第 i 个位置对应的原始节点
        ranks_reordered = ranks[node_ids]

        return {
            'features': features.astype(np.float32),
            'node_ids': np.array(node_ids, dtype=np.int64),
            'ranks': ranks_reordered.astype(np.int64)
        }

    def _standardize_graph(self, G: nx.Graph) -> nx.Graph:
        """标准化图：转为无向简单图，节点重标记为 0..n-1"""
        if G.is_directed():
            G = G.to_undirected()
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))

        # 重标记节点
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        return G

    def __len__(self) -> int:
        if self.cache_features:
            return len(self.cached_data)
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_features:
            sample = self.cached_data[idx]
        else:
            sample = self._process_graph(self.graphs[idx])

        # 转换为 Tensor
        return {
            'features': torch.from_numpy(sample['features']),
            'node_ids': torch.from_numpy(sample['node_ids']),
            'ranks': torch.from_numpy(sample['ranks'])
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    自定义 collate 函数，处理不同长度的序列

    使用 padding 将序列对齐到 batch 中的最大长度
    """
    max_len = max(sample['features'].size(0) for sample in batch)
    batch_size = len(batch)

    # 初始化 padded tensors
    features_padded = torch.zeros(batch_size, max_len, 4)
    ranks_padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    masks = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, sample in enumerate(batch):
        seq_len = sample['features'].size(0)
        features_padded[i, :seq_len] = sample['features']
        ranks_padded[i, :seq_len] = sample['ranks']
        masks[i, :seq_len] = 1

    return {
        'features': features_padded,
        'ranks': ranks_padded,
        'mask': masks
    }


# ============================================================================
# 训练器类
# ============================================================================

class MambaTrainer:
    """
    Mamba 网络拆解模型训练器

    支持完整训练流程、验证评估、模型保存与加载
    """

    def __init__(
        self,
        model: MambaDismantlingModel,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        checkpoint_dir: str = 'checkpoints'
    ):
        """
        Parameters
        ----------
        model : MambaDismantlingModel
            待训练的模型
        device : str
            训练设备
        learning_rate : float
            学习率
        weight_decay : float
            权重衰减（L2 正则化）
        checkpoint_dir : str
            模型保存目录
        """
        self.model = model.to(device)
        self.device = device

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 损失函数
        self.criterion = ListMLELoss()

        # 检查点目录
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 训练状态
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        训练一个 epoch

        Returns
        -------
        avg_loss : float
            平均损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # 移动数据到设备
            features = batch['features'].to(self.device)
            ranks = batch['ranks'].to(self.device)
            mask = batch['mask'].to(self.device)

            # 前向传播（单次）
            pred_scores = self.model(features)  # (batch_size, seq_len)

            # 向量化 ListMLE 损失（padding 位置由 mask 屏蔽）
            loss = self.criterion(pred_scores, ranks, mask=mask)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """
        在验证集上评估

        Returns
        -------
        avg_loss : float
            平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            features = batch['features'].to(self.device)
            ranks = batch['ranks'].to(self.device)
            mask = batch['mask'].to(self.device)

            pred_scores = self.model(features)

            loss = self.criterion(pred_scores, ranks, mask=mask)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        完整训练流程

        Parameters
        ----------
        train_loader : DataLoader
            训练数据加载器
        val_loader : DataLoader, optional
            验证数据加载器
        num_epochs : int
            训练轮数
        patience : int
            早停耐心值（验证损失不下降的最大轮数）
        verbose : bool
            是否打印训练信息

        Returns
        -------
        history : Dict[str, List[float]]
            训练历史，包含 'train_loss' 和 'val_loss'
        """
        best_epoch = 0
        patience_counter = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # 验证
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)

                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    self.save_checkpoint('best_model.pth')
                else:
                    patience_counter += 1

            elapsed = time.time() - start_time

            # 打印信息
            if verbose:
                info = f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}"
                if val_loss is not None:
                    info += f" | Val Loss: {val_loss:.4f}"
                info += f" | Time: {elapsed:.2f}s"
                logger.info(info)
                print(info)

            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')

            # 早停
            if val_loader is not None and patience_counter >= patience:
                logger.info(f"早停触发，最佳 epoch: {best_epoch+1}")
                print(f"早停触发，最佳验证损失在 epoch {best_epoch+1}")
                break

        # 训练结束，加载最佳模型
        if val_loader is not None and (self.checkpoint_dir / 'best_model.pth').exists():
            self.load_checkpoint('best_model.pth')
            logger.info("加载最佳模型")

        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }

    def save_checkpoint(self, filename: str):
        """保存模型检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'model_config': {
                'input_dim': self.model.input_dim,
                'd_model': self.model.d_model,
                'n_layers': self.model.n_layers
            }
        }

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"检查点已保存: {save_path}")

    def load_checkpoint(self, filename: str):
        """加载模型检查点"""
        load_path = self.checkpoint_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"检查点不存在: {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        logger.info(f"检查点已加载: {load_path}, epoch {self.current_epoch}")


# ============================================================================
# 辅助函数
# ============================================================================

def create_trainer(
    input_dim: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    device: str = 'cuda',
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    checkpoint_dir: str = 'checkpoints'
) -> MambaTrainer:
    """
    创建训练器的便捷函数

    Returns
    -------
    trainer : MambaTrainer
        初始化的训练器实例
    """
    model = MambaDismantlingModel(
        input_dim=input_dim,
        d_model=d_model,
        n_layers=n_layers
    )

    trainer = MambaTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        checkpoint_dir=checkpoint_dir
    )

    return trainer


def load_trained_model(checkpoint_path: str, device: str = 'cuda') -> MambaDismantlingModel:
    """
    加载训练好的模型（仅模型权重，用于推理）

    Parameters
    ----------
    checkpoint_path : str
        检查点文件路径
    device : str
        设备

    Returns
    -------
    model : MambaDismantlingModel
        加载权重后的模型
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 从配置重建模型
    config = checkpoint['model_config']
    model = MambaDismantlingModel(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        n_layers=config['n_layers']
    )

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model
