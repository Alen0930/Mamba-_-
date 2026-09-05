# Mamba 网络拆解算法

## 概述

基于 Mamba 序列模型的网络拆解算法，使用状态空间模型（SSM）对节点拓扑特征进行序列编码，输出节点拆解优先级分数。

**主要功能**:
- ✅ 完整推理流程（随机初始化或加载训练权重）
- ✅ 端到端训练系统（ListMLE 排序损失）
- ✅ 统一接口集成（`dismantle(G, method='mamba')`）
- ✅ GPU 加速支持（RTX 5060 兼容）

## 快速开始

### 推理模式

```python
import networkx as nx
from network_dismantling.unified_interface import dismantle

# 创建网络
G = nx.barabasi_albert_graph(1000, 3, seed=42)

# 运行 Mamba 拆解（使用随机初始化权重）
sequence = dismantle(G, method='mamba', stop_condition=1)

print(f"拆解序列长度: {len(sequence)}")
```

### 训练模式

```python
from network_dismantling.Mamba import create_trainer, DismantlingDataset, collate_fn
from network_dismantling.unified_interface import dismantle
from torch.utils.data import DataLoader

# 准备训练数据
train_graphs = [nx.barabasi_albert_graph(100, 3, seed=i) for i in range(100)]
dismantler_fn = lambda G: dismantle(G, method='corehd')  # 监督信号

# 创建数据集
dataset = DismantlingDataset(train_graphs, dismantler_fn, cache_features=True)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# 创建训练器并训练
trainer = create_trainer(device='cuda', checkpoint_dir='checkpoints')
history = trainer.train(train_loader, val_loader, num_epochs=100, patience=10)
```

详见: [TRAINING.md](TRAINING.md)

## 模块结构

```
network_dismantling/Mamba/
├── __init__.py              # 模块导出
├── feature_encoding.py      # 节点特征编码（4维拓扑特征）
├── mamba_model.py          # Mamba 评分模型（2层SSM）
├── mamba_dismantler.py     # 拆解算法主逻辑
├── trainer.py              # 训练引擎（ListMLE损失）
├── README.md               # 本文档
└── TRAINING.md             # 训练文档
```

## 实现细节

### 1. 节点特征编码 (`feature_encoding.py`)

**功能**: 将图转换为有序序列，提取并归一化拓扑特征

**特征维度** (4维):
- 节点度 (Degree)
- k-core 值
- PageRank 值
- 接近中心性 (Closeness Centrality)

**排序策略**: 按节点度降序排列

**归一化**: Min-Max 归一化，将所有特征缩放到 [0, 1]

### 2. Mamba 评分模型 (`mamba_model.py`)

**架构**:
```
输入层 (4维) 
  ↓ 线性投影
隐藏层 (64维)
  ↓ Mamba Block × 2 (带残差连接)
输出层 (1维优先级分数)
```

**超参数**:
- `d_model`: 64 (隐藏层维度)
- `d_state`: 16 (SSM 状态维度)
- `d_conv`: 4 (卷积核大小)
- `expand`: 2 (扩展因子)
- `n_layers`: 2 (Mamba 层数)

**关键配置**:
- `use_fast_path=False`: 禁用 CUDA fast path，兼容 RTX 5060 (sm_120)
- 使用 `selective_scan_ref` 替代 CUDA kernel

### 3. 拆解算法 (`mamba_dismantler.py`)

**流程**:
1. 提取节点特征并归一化
2. 通过 Mamba 模型推理，获取所有节点的优先级分数
3. 按分数从高到低渐进式移除节点
4. 达到 `stop_condition` 后停止

**设备支持**:
- 自动检测 CUDA 可用性
- GPU 推理（通过 selective_scan_ref 回退）
- CPU 推理回退

## 统一接口集成

### 注册

已注册到 `network_dismantling.unified_interface.METHOD_REGISTRY`，键名为 `'mamba'`

### 调用方式

```python
from network_dismantling.unified_interface import dismantle

# 基本调用
sequence = dismantle(G, method='mamba')

# 指定停止条件
sequence = dismantle(G, method='mamba', stop_condition=10)

# 指定设备
sequence = dismantle(G, method='mamba', device='cuda')  # 或 'cpu'
```

### 参数

- `G`: networkx.Graph - 输入图
- `method`: 'mamba' - 方法名
- `stop_condition`: int - 停止条件（LCC 大小）
- `device`: str - 计算设备 ('cuda' 或 'cpu')

## 性能测试结果

### 测试网络
- BA 无标度网络
- 1000 节点，2991 条边
- 平均度: 5.98

### 拆解性能对比

| 方法 | 耗时(秒) | Stop步数 (LCC≤10%) | FC值 | LCC@20%移除 |
|------|----------|-------------------|------|-------------|
| degree | 0.205 | 270 | 0.3840 | 530 |
| pagerank | 0.122 | 249 | 0.3520 | 512 |
| betweenness | 1.061 | 287 | 0.4160 | 588 |
| **mamba** | 0.708 | 746 | 0.8390 | 799 |

### GPU 使用情况

- 设备: NVIDIA GeForce RTX 5060 Laptop GPU
- CUDA 版本: 12.8
- 内存使用: ~27 MB (500节点网络)

## 环境依赖

### Python 版本
- Python 3.12.10

### 核心依赖
- torch 2.9.1+cu128
- mamba_ssm 2.2.6.post3
- triton-windows 3.5.1.post22
- networkx
- numpy

### 重要说明

**RTX 5060 (sm_120) 兼容性**:
- 预编译的 `causal_conv1d` 和 `mamba_ssm` wheel 最高支持 sm_100
- PTX 不被驱动 JIT 到 sm_120，CUDA kernel 不可用
- **解决方案**: 卸载 `causal_conv1d`，触发 PyTorch Conv1d 回退
- 配置 `use_fast_path=False` 并使用 `selective_scan_ref`

详见: `.claude/projects/.../memory/rtx5060-mamba-env.md`

## 使用示例

### 基础使用

```python
import networkx as nx
from network_dismantling.unified_interface import dismantle

# 创建网络
G = nx.barabasi_albert_graph(1000, 3, seed=42)

# 运行 Mamba 拆解
sequence = dismantle(G, method='mamba', stop_condition=1)

print(f"拆解序列长度: {len(sequence)}")
print(f"前10个节点: {sequence[:10]}")
```

### 与 evaluate.py 集成

```python
from evaluate import calc_metrics, plot_robustness

# 计算指标
stop_step, fc_value, reach_stop, reach_fc = calc_metrics(
    G, sequence,
    stop_ratio=0.1,
    fc_threshold=0.01
)

# 多方法对比
methods = ['degree', 'pagerank', 'mamba']
sequences = {m: dismantle(G, method=m) for m in methods}

# 绘制对比曲线
plot_robustness(
    G, sequences,
    sample_step=10,
    save_path='robustness_comparison.png'
)
```

## 验证脚本

- `test_mamba.py`: 基础功能测试
- `verify_mamba_integration.py`: 完整集成验证

运行验证:
```bash
source venv312/Scripts/activate
python verify_mamba_integration.py
```

## 未来改进方向

1. **模型训练**: 当前使用随机初始化权重，可通过真实网络数据训练优化性能
2. **特征工程**: 添加更多拓扑特征（介数中心性、聚类系数等）
3. **超参数调优**: 优化 `d_model`、`n_layers` 等超参数
4. **批处理推理**: 对大规模网络使用分批推理降低内存占用
5. **动态评分**: 在拆解过程中动态更新节点特征和分数（类似 CoreHD）

## 引用

如果使用本实现，请引用：

- Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
- 项目地址: [此仓库地址]

## 许可

与项目主仓库保持一致
