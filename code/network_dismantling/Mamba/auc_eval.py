"""
GNN+Mamba 模型的 AUC 快速评估（静态打分）

用途：**训练过程中按主指标选最优模型**。

背景：trainer 原本按 val_loss（ListMLE 排序损失）存 best_model.pth，但
docs/阶段性成果.md 已经记录过「loss 与拆解指标错位」——排序拟合得像 ≠ 拆解得好。
现在 AUC 升为主指标，模型选择准则必须跟着换。

为什么用**静态打分**：每轮做一次前向拿到全图分数，按分数降序即得到一条完整
序列（长度 n），再走一次 LCC 轨迹。成本 = 1 次前向 + O(n(n+m)) 的轨迹遍历。
相比之下**动态重算**（每移除 b 个节点重算一次）要 (n/b) 次前向，每 epoch 都做
会让训练时间翻倍，所以只在训练结束后做一次动态复核。
"""
from typing import List, Optional, Sequence

import networkx as nx
import numpy as np
import torch

from .feature_encoding import extract_node_features
from evaluate import calc_auc, lcc_trace


def model_scores(model, G: nx.Graph, device: str, order: str = "degree") -> List[int]:
    """
    对整张图做一次前向，返回**按分数降序**的节点 ID 列表（= 静态拆解序列）。

    节点与特征的对齐方式与 mamba_dismantler 保持一致：
    先标准化节点标签为 0..n-1，提取 degree 特征（节点顺序由 order 决定），
    邻接矩阵按同一顺序重排，前向后把位置映射回原始节点 ID。

    order 必须与训练时一致，否则 Mamba 看到的序列分布不同，评估无效。
    """
    node_list = list(G.nodes())
    G_std = nx.relabel_nodes(G, {v: i for i, v in enumerate(node_list)})

    features, node_ids = extract_node_features(G_std, feature_set="degree", order=order)
    adj = nx.to_numpy_array(G_std, dtype=np.uint8)
    adj = adj[np.ix_(node_ids, node_ids)]

    with torch.no_grad():
        x = torch.from_numpy(features).float().to(device).unsqueeze(0)
        a = torch.from_numpy(adj).float().to(device).unsqueeze(0)
        out = model(x, a)
    logits = out[0] if isinstance(out, tuple) else out  # 兼容 Actor-Critic
    scores = logits.squeeze(0).detach().cpu().numpy()

    # 分数降序 -> 节点位置 -> 原始节点 ID
    order = np.argsort(-scores, kind="stable")
    return [node_list[node_ids[int(p)]] for p in order]


def graph_auc(model, G: nx.Graph, device: str, order: str = "degree") -> float:
    """单图静态 AUC（越低越好）"""
    seq = model_scores(model, G, device, order=order)
    n = G.number_of_nodes()
    return calc_auc(lcc_trace(G, seq), n=n)


def mean_auc(
    model,
    graphs: Sequence[nx.Graph],
    device: str,
    max_graphs: Optional[int] = None,
    order: str = "degree",
) -> float:
    """
    多图平均静态 AUC。

    Parameters
    ----------
    max_graphs : int, optional
        只评测前 k 张图（训练中每 epoch 调用，用子集控制开销；
        训练结束后可用 None 做全量复核）。取的是前 k 张，调用方应先自行
        打乱或固定顺序，保证各 epoch 评测的是同一批图，指标才可比。
    """
    if not graphs:
        return float("nan")
    subset = graphs[:max_graphs] if max_graphs else graphs
    was_training = model.training
    model.eval()
    try:
        aucs = [graph_auc(model, G, device, order=order) for G in subset]
    finally:
        if was_training:
            model.train()
    return float(np.mean(aucs))
