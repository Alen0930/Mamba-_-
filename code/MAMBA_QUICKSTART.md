# Mamba 网络拆解算法 - 快速开始

## 安装环境

使用 Python 3.12 虚拟环境 (venv312):

```bash
source venv312/Scripts/activate  # Windows Git Bash
```

## 快速使用

```python
import networkx as nx
from network_dismantling.unified_interface import dismantle

# 创建测试网络
G = nx.barabasi_albert_graph(1000, 3, seed=42)

# 运行 Mamba 拆解
sequence = dismantle(G, method='mamba', stop_condition=1)

print(f"拆解序列: {sequence[:10]}...")
```

## 与基线方法对比

```python
from evaluate import calc_metrics, plot_robustness

# 运行多种方法
methods = ['degree', 'pagerank', 'betweenness', 'mamba']
sequences = {m: dismantle(G, method=m) for m in methods}

# 绘制对比曲线
plot_robustness(G, sequences, save_path='comparison.png')

# 计算量化指标
for method, seq in sequences.items():
    stop_step, fc_value, reach_stop, reach_fc = calc_metrics(G, seq)
    print(f"{method}: Stop步数={stop_step}, FC值={fc_value:.4f}")
```

## 完整验证

```bash
# 基础功能测试
python test_mamba.py

# 完整集成验证（推荐）
python verify_mamba_integration.py
```

## 输出示例

验证脚本会输出：
- ✅ 各项功能测试结果
- 📊 与基线方法的性能对比
- 📈 鲁棒性对比曲线图 (mamba_robustness_comparison.png)
- 💾 GPU 内存使用情况

## 性能参考 (1000节点BA网络)

| 方法 | 耗时 | Stop步数 | FC值 |
|------|------|---------|------|
| degree | 0.2s | 270 | 0.384 |
| pagerank | 0.1s | 249 | 0.352 |
| mamba | 0.7s | 746 | 0.839 |

**注意**: Mamba 当前使用随机初始化权重，性能低于训练优化的基线方法。通过真实数据训练可显著提升性能。

## 故障排除

### CUDA kernel error

如遇到 "no kernel image is available" 错误：

```bash
pip uninstall causal_conv1d -y
```

这会触发 PyTorch Conv1d 回退，兼容 RTX 5060 (sm_120)。

详见: [network_dismantling/Mamba/README.md](network_dismantling/Mamba/README.md)
