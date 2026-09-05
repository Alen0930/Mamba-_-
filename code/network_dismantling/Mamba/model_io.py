"""
模型权重管理工具

封装模型权重的保存、加载与推理管线注册：

1. save_model_weights: 保存模型权重（state_dict + 模型配置）
2. load_inference_model: 从检查点构建推理模型（eval 模式）
3. register_model_for_inference: 加载检查点并注册到 mamba 推理管线，
   注册后 dismantle(G, method='mamba') 直接使用训练权重，无需修改调用方式
4. unregister_model: 清除注册，恢复随机初始化

使用示例:
    from network_dismantling.Mamba.model_io import (
        save_model_weights,
        load_inference_model,
        register_model_for_inference,
        unregister_model,
        is_model_registered,
    )
    from network_dismantling.unified_interface import dismantle

    # 保存训练后的模型权重
    save_model_weights(trainer.model, "weights/mamba_trained.pth")

    # 方式 1: 注册到推理管线（之后 mamba 方法自动使用训练权重）
    register_model_for_inference("checkpoints/xxx/best_model.pth")
    seq = dismantle(G, method='mamba')   # 自动使用注册的权重

    # 方式 2: 显式指定检查点（无需注册，优先级最高）
    seq = dismantle(G, method='mamba', model_path="checkpoints/xxx/best_model.pth")

    # 恢复随机初始化
    unregister_model()
"""
import logging
from pathlib import Path
from typing import Dict, Optional

import torch

from .mamba_model import MambaDismantlingModel, create_mamba_model
from .mamba_dismantler import (
    register_trained_weights,
    unregister_trained_weights,
    is_trained_weights_registered,
    get_registered_weights_source,
)

logger = logging.getLogger(__name__)


def _default_device() -> str:
    """自动选择默认设备"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_model_weights(model: MambaDismantlingModel, path, extra: Optional[Dict] = None) -> str:
    """
    保存模型权重（兼容 trainer 检查点格式）

    保存内容: model_state_dict + model_config，
    加载后可按配置重建模型（任意 input_dim/d_model/n_layers 组合均可恢复）。

    Parameters
    ----------
    model : MambaDismantlingModel
        待保存的模型
    path : str
        保存路径
    extra : Dict, optional
        附加信息（如 epoch、best_val_loss 等）

    Returns
    -------
    path : str
        实际保存路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": model.input_dim,
            "d_model": model.d_model,
            "n_layers": model.n_layers,
        },
    }
    if extra:
        data.update(extra)

    torch.save(data, path)
    logger.info("Model weights saved: %s", path)
    return str(path)


def load_inference_model(path, device: Optional[str] = None) -> MambaDismantlingModel:
    """
    从检查点加载推理模型（eval 模式）

    兼容两种文件格式：
    - trainer.save_checkpoint 的完整检查点（含 model_state_dict / model_config）
    - save_model_weights 保存的权重文件
    - 纯 state_dict 文件

    Parameters
    ----------
    path : str
        检查点路径
    device : str, optional
        目标设备，默认自动选择

    Returns
    -------
    model : MambaDismantlingModel
        加载权重后的模型（eval 模式）
    """
    if device is None:
        device = _default_device()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"检查点文件不存在: {path}")

    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("model_config")
    else:
        state_dict = checkpoint  # 纯 state_dict 文件
        config = None

    # 按保存的配置重建模型，保证任意配置均可恢复
    if config:
        model = MambaDismantlingModel(**config)
    else:
        model = create_mamba_model(device=device)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    logger.info(
        "Inference model loaded from %s (d_model=%d, n_layers=%d)",
        path, model.d_model, model.n_layers,
    )
    return model


def register_model_for_inference(path, device: Optional[str] = None) -> MambaDismantlingModel:
    """
    加载检查点并注册到 Mamba 推理管线

    注册后，unified_interface 的 mamba 方法（dismantle(G, method='mamba')）
    自动使用该训练权重，调用方式完全不变。

    Parameters
    ----------
    path : str
        检查点路径（trainer.save_checkpoint 生成的格式）
    device : str, optional
        注册后推理使用的默认设备，默认自动选择

    Returns
    -------
    model : MambaDismantlingModel
        加载的推理模型（同时已完成全局注册）
    """
    if device is None:
        device = _default_device()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"检查点文件不存在: {path}")

    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("model_config")
    else:
        state_dict = checkpoint  # 纯 state_dict 文件
        config = None

    # 注册到 mamba_dismantle 的全局权重
    register_trained_weights(state_dict, config=config, source=str(path))

    # 同时返回一个可直接使用的推理模型
    model = load_inference_model(path, device=device)

    logger.info(
        "Trained weights registered for inference: %s (source: %s)",
        "dismantle(G, method='mamba')", path,
    )
    return model


def unregister_model():
    """清除已注册的训练权重，mamba 推理恢复随机初始化"""
    unregister_trained_weights()
    logger.info("Trained weights unregistered, mamba falls back to random init")


def is_model_registered() -> bool:
    """是否已注册训练权重"""
    return is_trained_weights_registered()


def registered_source() -> Optional[str]:
    """返回已注册权重的来源路径（未注册时返回 None）"""
    return get_registered_weights_source()
