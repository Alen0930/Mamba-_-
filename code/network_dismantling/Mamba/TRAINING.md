# Mamba 网络拆解模型 - 训练引擎文档

## 概述

完整的训练系统，支持基于 ListMLE 排序损失的端到端学习，使用 CoreHD 等高质量算法作为监督信号。

## 核心组件

### 1. ListMLE 排序损失 (`ListMLELoss`)

**原理**: List Maximum Likelihood Estimation，基于排列概率的学习排序损失函数。

**数学公式**:
```
L = -Σ[s_i - log(Σ_{j≥i} exp(s_j))]
```

其中 `s_i` 是按真实排序重排后的预测分数。

**特性**:
- 最大化正确排序的似然概率
- 梯度稳定（使用 log-sum-exp 技巧）
- 考虑整个序列的全局排序信息

**使用示例**:
```python
from network_dismantling.Mamba.trainer import ListMLELoss

criterion = ListMLELoss()
pred_scores = model(features)  # (batch_size, seq_len)
true_ranks = torch.tensor([[0, 1, 2, ...]])  # 真实排序
loss = criterion(pred_scores, true_ranks)
```

### 2. 数据集类 (`DismantlingDataset`)

**功能**: 将图数据转换为训练样本

**流程**:
1. 标准化图（节点重标记为 0..n-1）
2. 提取节点特征（4维拓扑特征）
3. 使用监督算法生成拆解序列
4. 将序列转换为排序标签（每个节点的移除位置）
5. 按特征序列顺序重排标签（与模型输入对齐）

**关键点**:
- 标签与特征序列节点一一对应
- 支持特征缓存（加速训练）
- 自动处理不同大小的图

**使用示例**:
```python
from network_dismantling.Mamba.trainer import DismantlingDataset
from network_dismantling.unified_interface import dismantle

# 定义监督算法
dismantler_fn = lambda G: dismantle(G, method='corehd')

# 创建数据集
dataset = DismantlingDataset(
    graphs=train_graphs,
    dismantler_fn=dismantler_fn,
    cache_features=True  # 推荐开启
)
```

### 3. 训练器 (`MambaTrainer`)

**功能**: 完整的训练流程管理

**特性**:
- AdamW 优化器（支持权重衰减）
- 自动梯度裁剪（防止梯度爆炸）
- 早停机制（基于验证损失）
- 模型检查点保存/加载
- 训练历史记录

**使用示例**:
```python
from network_dismantling.Mamba.trainer import create_trainer
from torch.utils.data import DataLoader

# 创建训练器
trainer = create_trainer(
    input_dim=4,
    d_model=64,
    n_layers=2,
    device='cuda',
    learning_rate=1e-3,
    weight_decay=1e-4,
    checkpoint_dir='checkpoints'
)

# 训练
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    patience=10,
    verbose=True
)
```

### 4. 数据加载 (`collate_fn`)

**功能**: 处理变长序列的批次组装

**策略**:
- Padding 对齐到 batch 中最大长度
- 生成 mask 标记有效位置
- 损失计算时自动过滤 padding

## 完整训练流程

### 步骤 1: 准备数据

```python
import networkx as nx
from network_dismantling.unified_interface import dismantle

# 生成训练图
train_graphs = [
    nx.barabasi_albert_graph(100, 3, seed=i) 
    for i in range(100)
]

# 定义监督算法（推荐使用 CoreHD）
dismantler_fn = lambda G: dismantle(G, method='corehd', stop_condition=1)
```

### 步骤 2: 创建数据集和加载器

```python
from network_dismantling.Mamba.trainer import DismantlingDataset, collate_fn
from torch.utils.data import DataLoader

# 创建数据集
train_dataset = DismantlingDataset(
    graphs=train_graphs,
    dismantler_fn=dismantler_fn,
    cache_features=True
)

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0  # Windows 建议设为 0
)
```

### 步骤 3: 创建训练器并训练

```python
from network_dismantling.Mamba.trainer import create_trainer

# 创建训练器
trainer = create_trainer(
    input_dim=4,
    d_model=64,
    n_layers=2,
    device='cuda',
    learning_rate=1e-3,
    weight_decay=1e-4,
    checkpoint_dir='checkpoints/mamba_dismantling'
)

# 训练
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    patience=10,
    verbose=True
)
```

### 步骤 4: 加载训练好的模型

```python
from network_dismantling.Mamba.trainer import load_trained_model

# 加载最佳模型
model = load_trained_model(
    checkpoint_path='checkpoints/mamba_dismantling/best_model.pth',
    device='cuda'
)

# 推理
model.eval()
with torch.no_grad():
    scores = model(features_tensor)
```

## 监督信号选择

推荐使用高质量的拆解算法作为监督信号：

| 算法 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **CoreHD** | 高质量，理论基础强 | 计算较慢 | ⭐⭐⭐⭐⭐ |
| **degree** | 速度快，效果稳定 | 质量一般 | ⭐⭐⭐⭐ |
| **pagerank** | 考虑全局结构 | 计算中等 | ⭐⭐⭐⭐ |
| **betweenness** | 精度高 | 计算极慢 | ⭐⭐⭐ |

**推荐策略**:
- 小数据集（<1000图）: 使用 CoreHD
- 大数据集（>1000图）: 使用 degree 或 pagerank
- 混合训练: 70% degree + 30% CoreHD

## 超参数调优

### 模型超参数

```python
# 默认配置（适用于大多数场景）
input_dim = 4        # 特征维度（固定）
d_model = 64        # 隐藏维度（可调：32, 64, 128）
n_layers = 2        # Mamba 层数（可调：1, 2, 3）
```

**调优建议**:
- 小图（<100节点）: `d_model=32, n_layers=1`
- 中图（100-500节点）: `d_model=64, n_layers=2` ⭐推荐
- 大图（>500节点）: `d_model=128, n_layers=3`

### 训练超参数

```python
# 默认配置
learning_rate = 1e-3     # 学习率（可调：1e-4 到 1e-2）
weight_decay = 1e-4      # L2 正则化
batch_size = 8           # 批次大小（可调：4, 8, 16）
num_epochs = 100         # 最大训练轮数
patience = 10            # 早停耐心值
```

**调优建议**:
- 学习率: 从 1e-3 开始，如果震荡降低到 1e-4
- 批次大小: GPU 内存允许的情况下尽量大
- 早停: patience 设为 epoch 数的 10%

## 训练监控

### 损失曲线

```python
import matplotlib.pyplot as plt

# 绘制训练历史
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_history.png')
```

### 评估指标

```python
# 在测试集上评估
from evaluate import calc_metrics

test_sequence = predict_dismantling(model, test_graph)
stop_step, fc_value, reach_stop, reach_fc = calc_metrics(
    test_graph, 
    test_sequence,
    stop_ratio=0.1,
    fc_threshold=0.01
)

print(f"Stop步数: {stop_step}, FC值: {fc_value:.4f}")
```

## 训练技巧

### 1. 数据增强

```python
# 使用多种图类型增加泛化性
def generate_diverse_graphs(num_graphs):
    graphs = []
    for i in range(num_graphs):
        graph_type = np.random.choice(['ba', 'er', 'ws'])
        if graph_type == 'ba':
            G = nx.barabasi_albert_graph(n, m, seed=i)
        elif graph_type == 'er':
            G = nx.erdos_renyi_graph(n, p, seed=i)
        else:
            G = nx.watts_strogatz_graph(n, k, p, seed=i)
        graphs.append(G)
    return graphs
```

### 2. 课程学习

```python
# 从小图开始，逐渐增加难度
def curriculum_learning():
    # Stage 1: 小图
    small_graphs = [nx.ba_graph(50, 2) for _ in range(50)]
    train_on(small_graphs, epochs=20)
    
    # Stage 2: 中图
    medium_graphs = [nx.ba_graph(100, 3) for _ in range(50)]
    train_on(medium_graphs, epochs=20)
    
    # Stage 3: 大图
    large_graphs = [nx.ba_graph(200, 4) for _ in range(50)]
    train_on(large_graphs, epochs=20)
```

### 3. 迁移学习

```python
# 先在 degree 上预训练，再在 CoreHD 上微调
# Stage 1: 快速预训练
trainer.train(degree_loader, epochs=50)

# Stage 2: 精细微调
trainer.optimizer.param_groups[0]['lr'] = 1e-4  # 降低学习率
trainer.train(corehd_loader, epochs=20)
```

## 常见问题

### Q1: 损失不下降

**可能原因**:
- 学习率过大或过小
- 批次大小太小
- 监督信号质量差

**解决方案**:
```python
# 降低学习率
trainer.optimizer.param_groups[0]['lr'] = 1e-4

# 增加批次大小
train_loader = DataLoader(dataset, batch_size=16, ...)

# 使用更好的监督算法
dismantler_fn = lambda G: dismantle(G, method='corehd')
```

### Q2: 过拟合

**症状**: 训练损失持续下降，验证损失上升

**解决方案**:
```python
# 增加 weight_decay
trainer = create_trainer(weight_decay=1e-3)

# 增加训练数据
train_graphs = generate_graphs(num_graphs=500)

# 使用 dropout（需修改模型）
```

### Q3: GPU 内存不足

**解决方案**:
```python
# 减小批次大小
batch_size = 4

# 减小模型大小
trainer = create_trainer(d_model=32, n_layers=1)

# 关闭特征缓存
dataset = DismantlingDataset(..., cache_features=False)
```

## 性能基准

### 训练时间（100 个图，100 个 epoch）

| 配置 | GPU | 时间 |
|------|-----|------|
| d_model=32, n_layers=1 | RTX 5060 | ~5 分钟 |
| d_model=64, n_layers=2 | RTX 5060 | ~10 分钟 |
| d_model=128, n_layers=3 | RTX 5060 | ~20 分钟 |

### 内存占用

| 批次大小 | 图大小 | GPU 内存 |
|---------|--------|----------|
| 4 | 100 节点 | ~200 MB |
| 8 | 100 节点 | ~400 MB |
| 16 | 100 节点 | ~800 MB |

## 完整示例

见 `train_mamba_example.py`:

```bash
# 快速测试
python train_mamba_example.py --quick

# 完整训练
python train_mamba_example.py
```

## 参考文献

1. **ListMLE**: Xia et al. "Listwise approach to learning to rank: theory and algorithm." ICML 2008.
2. **Mamba**: Gu & Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv 2023.
3. **CoreHD**: Zdeborová et al. "Fast and simple decycling and dismantling of networks." Scientific Reports 2016.
