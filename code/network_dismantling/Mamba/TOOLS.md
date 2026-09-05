# Mamba 外围工具脚本

围绕核心训练引擎（trainer.py）的三个配套工具，接口与现有模块严格对齐，可直接拼接运行。

## 工具清单

| 文件 | 功能 | 位置 |
|------|------|------|
| dataset_generator.py | 批量数据集生成（BA 网络 + CoreHD 标签） | network_dismantling/Mamba/ |
| model_io.py | 模型权重保存 / 加载 / 推理管线注册 | network_dismantling/Mamba/ |
| eval_after_train.py | 训练后效果验证（指标对比 + 曲线图） | 项目根目录 |

## 完整训练-验证工作流

```python
# ---------- Step 1: 生成训练数据集 ----------
from network_dismantling.Mamba.dataset_generator import (
    generate_and_save_datasets, build_dataloaders,
)

train_ds, val_ds = generate_and_save_datasets(
    out_dir="datasets/ba_corehd",
    num_samples=16,          # 样本总数可设置
    n_range=(500, 1500),     # BA 节点数范围
    m_range=(2, 5),          # BA 连边参数范围
    seed=42,
)
train_loader, val_loader = build_dataloaders(train_ds, val_ds, batch_size=2)

# ---------- Step 2: 训练模型 ----------
from network_dismantling.Mamba.trainer import create_trainer

trainer = create_trainer(device="cuda", checkpoint_dir="checkpoints/mamba_dismantling")
trainer.train(train_loader, val_loader, num_epochs=100, patience=10)
# 最佳模型保存在 checkpoints/mamba_dismantling/best_model.pth

# ---------- Step 3: 注册训练权重（调用方式不变） ----------
from network_dismantling.Mamba.model_io import register_model_for_inference
from network_dismantling.unified_interface import dismantle

register_model_for_inference("checkpoints/mamba_dismantling/best_model.pth")
seq = dismantle(G, method="mamba")   # 自动使用训练权重
```

## 各工具详细说明

### 1. dataset_generator.py

**核心函数**：

| 函数 | 说明 |
|------|------|
| `generate_ba_graphs(num_samples, n_range, m_range, seed)` | 批量生成 BA 图 |
| `build_datasets(graphs, split_ratio=0.8, label_method='CoreHD')` | 构建 8:2 训练/验证集 |
| `generate_and_save_datasets(out_dir, ...)` | 一键生成 + 保存 |
| `load_datasets(out_dir)` / `save_datasets(...)` | 本地文件加载 / 保存 |
| `build_dataloaders(train_ds, val_ds, batch_size)` | 构建 DataLoader（兼容 collate_fn） |

**命令行**：
```bash
python -m network_dismantling.Mamba.dataset_generator \
    --num-samples 16 --out-dir datasets/ba_corehd --seed 42
```

### 2. model_io.py

**核心函数**：

| 函数 | 说明 |
|------|------|
| `save_model_weights(model, path)` | 保存权重（state_dict + 模型配置） |
| `load_inference_model(path, device)` | 从检查点构建推理模型 |
| `register_model_for_inference(path)` | 注册到 mamba 推理管线 |
| `unregister_model()` | 清除注册，恢复随机初始化 |
| `is_model_registered()` | 查询注册状态 |

**两种使用训练权重的方式**：

```python
# 方式 1: 全局注册（之后所有 mamba 调用自动使用）
register_model_for_inference("checkpoints/xxx/best_model.pth")
seq = dismantle(G, method='mamba')

# 方式 2: 显式指定检查点（优先级最高，无需注册）
seq = dismantle(G, method='mamba', model_path="checkpoints/xxx/best_model.pth")
```

权重加载优先级：`model_path 参数` > `全局注册权重` > `随机初始化`

### 3. eval_after_train.py

**命令行**：
```bash
python eval_after_train.py --checkpoint checkpoints/mamba_dismantling/best_model.pth

# 自定义测试网络与输出
python eval_after_train.py --checkpoint xxx.pth \
    --n 1500 --m 3 --seed 2026 --output eval_result.png
```

**输出内容**：
1. 核心指标对比表（degree / pagerank / corehd / mamba 训练前后）
2. Mamba 训练前后提升对比（Stop步数、FC值改善率）
3. 鲁棒性对比曲线图（保存为本地 PNG）

## 实测验证结果

以 200 节点 BA 测试网络（模型仅在 5 个小图上快速训练 5 epoch）：

| 方法 | Stop步数 | FC值 |
|------|---------|------|
| mamba_random | 180 | 0.990 |
| mamba_trained | 65 | 0.805 |
| degree | 66 | 0.560 |
| pagerank | 66 | 0.500 |
| corehd | 64 | 0.495 |

训练后 Mamba 的 Stop 步数改善 63.9%，接近 CoreHD 水平。

## 数据文件格式

- 数据集: pickle 格式（`train_dataset.pkl` / `val_dataset.pkl`），
  包含图列表 + 预处理特征/标签缓存，加载后无需重新运行 CoreHD
- 检查点: torch.save 格式，兼容 trainer.save_checkpoint 的完整检查点、
  save_model_weights 的权重文件、纯 state_dict 三种格式
