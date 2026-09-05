"""
Mamba 网络拆解算法主逻辑

权重加载优先级（从高到低）：
1. model_path 参数显式指定的检查点路径
2. weights 参数显式传入的 state_dict
3. register_trained_weights() 注册的全局训练权重（由 model_io 设置）
4. 随机初始化（未注册任何权重时）
"""
import numpy as np
import networkx as nx
import torch
from typing import List, Dict, Optional

from .feature_encoding import extract_node_features
from .mamba_model import MambaDismantlingModel, create_mamba_model

# ---------------------------------------------------------------------------
# 全局训练权重注册（由 model_io.register_model_for_inference 设置）
# 注册后 dismantle(G, method='mamba') 自动使用训练权重，无需修改调用方式
# ---------------------------------------------------------------------------
_registered_state_dict: Optional[Dict[str, torch.Tensor]] = None
_registered_config: Optional[Dict] = None
_registered_source: Optional[str] = None


def register_trained_weights(
    state_dict: Dict[str, torch.Tensor],
    config: Optional[Dict] = None,
    source: Optional[str] = None
):
    """
    注册训练权重到 Mamba 推理管线（模块级全局注册）

    Parameters
    ----------
    state_dict : Dict[str, torch.Tensor]
        模型权重（与 MambaDismantlingModel 结构一致）
    config : Dict, optional
        模型配置 {'input_dim', 'd_model', 'n_layers'}，
        与默认配置 (4, 64, 2) 不同时必须提供
    source : str, optional
        权重来源描述（如检查点路径），用于日志
    """
    global _registered_state_dict, _registered_config, _registered_source
    _registered_state_dict = {k: v.detach().clone() for k, v in state_dict.items()}
    _registered_config = dict(config) if config else None
    _registered_source = source


def unregister_trained_weights():
    """清除已注册的训练权重，恢复随机初始化"""
    global _registered_state_dict, _registered_config, _registered_source
    _registered_state_dict = None
    _registered_config = None
    _registered_source = None


def is_trained_weights_registered() -> bool:
    """是否已注册训练权重"""
    return _registered_state_dict is not None


def get_registered_weights_source() -> Optional[str]:
    """返回已注册权重的来源描述"""
    return _registered_source


# ---------------------------------------------------------------------------
# 模型构建辅助
# ---------------------------------------------------------------------------
def _load_checkpoint_model(checkpoint_path: str, device: str) -> MambaDismantlingModel:
    """
    从 trainer 生成的检查点构建推理模型

    兼容两种文件格式：
    - trainer.save_checkpoint 的完整检查点（含 model_state_dict / model_config）
    - 纯 state_dict 文件
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('model_config')
    else:
        state_dict = checkpoint
        config = None

    if config:
        model = MambaDismantlingModel(**config)
    else:
        model = create_mamba_model(device=device)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def _build_scoring_model(
    device: str,
    model_path: Optional[str] = None,
    weights: Optional[Dict[str, torch.Tensor]] = None
) -> MambaDismantlingModel:
    """按优先级构建评分模型并加载可用权重"""
    if model_path is not None:
        return _load_checkpoint_model(model_path, device)

    if weights is not None:
        model = create_mamba_model(device=device)
        model.load_state_dict(weights)
        model.eval()
        return model

    if _registered_state_dict is not None:
        if _registered_config:
            model = MambaDismantlingModel(**_registered_config).to(device)
        else:
            model = create_mamba_model(device=device)
        model.load_state_dict(_registered_state_dict)
        model.eval()
        return model

    return create_mamba_model(device=device)


# ---------------------------------------------------------------------------
# 拆解主逻辑
# ---------------------------------------------------------------------------
def mamba_dismantle(
    G: nx.Graph,
    stop_condition: int = 1,
    device: str = None,
    model_path: Optional[str] = None,
    weights: Optional[Dict[str, torch.Tensor]] = None
) -> List[int]:
    """
    使用 Mamba 模型进行网络拆解

    Parameters
    ----------
    G : nx.Graph
        输入图（节点标签应为 0 到 n-1 的整数）
    stop_condition : int
        停止条件：当最大连通分量大小 <= stop_condition 时停止
    device : str, optional
        计算设备 ('cuda' 或 'cpu')，默认自动检测
    model_path : str, optional
        检查点路径，显式指定训练权重（优先级最高）
    weights : Dict[str, torch.Tensor], optional
        直接传入 state_dict
    Returns
    -------
    removal_sequence : List[int]
        节点移除序列（仅到达 stop_condition 的部分）
    """
    n = G.number_of_nodes()

    if n == 0:
        return []

    # 自动选择设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 提取节点特征
    features, node_ids = extract_node_features(G)

    # 2. 构建 Mamba 模型（自动加载可用权重）
    model = _build_scoring_model(device, model_path=model_path, weights=weights)

    # 3. 将特征转换为张量并推理
    with torch.no_grad():
        features_tensor = torch.from_numpy(features).float().to(device)
        scores = model(features_tensor)  # (n_nodes,)
        scores = scores.cpu().numpy()

    # 4. 构建节点到分数的映射（原始节点ID -> 优先级分数）
    node_scores = np.zeros(n, dtype=np.float32)
    for i, node_id in enumerate(node_ids):
        node_scores[node_id] = scores[i]

    # 5. 渐进式拆解：按分数从高到低移除节点
    G_tmp = G.copy()
    removal_sequence = []

    while G_tmp.number_of_nodes() > 0:
        # 选择当前剩余节点中分数最高的节点
        remaining_nodes = list(G_tmp.nodes())
        if not remaining_nodes:
            break

        best_node = max(remaining_nodes, key=lambda v: node_scores[v])
        removal_sequence.append(best_node)
        G_tmp.remove_node(best_node)

        # 检查停止条件
        if G_tmp.number_of_nodes() > 0:
            components = list(nx.connected_components(G_tmp))
            lcc_size = max(len(c) for c in components) if components else 0
            if lcc_size <= stop_condition:
                break
        else:
            break

    return removal_sequence
