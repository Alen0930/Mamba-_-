# 网络拆解算法库使用指南

> 项目仓库：https://github.com/kuaiii/dismantling_topology

---

## 1. 项目概述

本库实现了 **21 种网络拆解（Network Dismantling）算法**，覆盖经典启发式、谱方法、信息论方法、图神经网络（GNN）和强化学习（RL）五大类。所有算法通过统一接口 `dismantle(G, method, stop_condition)` 调用，输入为 `networkx.Graph`，输出为拆解序列。

### 1.1 算法分类总览

| 类别 | 方法 | 原理简述 |
|------|------|---------|
| **静态启发式** | degree, pagerank, betweenness, eigenvector, random | 基于节点中心性一次性打分，按分数降序移除 |
| **精确解** | brute_force | 枚举小网络的所有节点组合，找最优解 |
| **Collective Influence** | CI_L1, CI_L2, CI_L3 | 考虑 L-hop 邻居影响的节点重要性度量 |
| **核分解** | CoreHD | 2-core 最大度移除 → 大树拆分 → 贪心回插 |
| **谱方法** | GND, EGND | 谱二分 + 贪心顶点覆盖，EGND 多次取最优 |
| **爆炸免疫** | EI_s1, EI_s2 | 有效度数 + DSU + 贪心策略 |
| **信息论/纠缠** | entanglement_small/mid/large, vertex_entanglement | 基于 Shannon 熵的谱计算 |
| **监督 GNN** | GDM, GDM+R | GAT 学习节点重要性 + 可选重插入优化 |
| **强化学习+GNN** | FINDER | DQN + GraphSAGE，逐步决策移除节点 |

---

## 2. 环境配置

### 2.1 主环境：`kanResilience`

用于运行除 FINDER 外的所有算法。

```bash
conda activate kanResilience
```

**Python 版本**：3.10.20  
**核心依赖**：

| 包 | 版本 | 用途 |
|----|------|------|
| torch | 2.3.0+cu121 | GDM 的 GAT 模型推理 |
| torch_geometric | 2.7.0 | 图神经网络数据结构和层 |
| networkx | 3.4.2 | 图数据结构和算法 |
| numpy | 1.26.4 | 数值计算 |
| scipy | 1.15.3 | 稀疏矩阵、谱分解 |
| pandas | 2.3.3 | 数据处理和分析 |

**安装命令**（如需要重建环境）：
```bash
conda create -n kanResilience python=3.10
conda activate kanResilience
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.7.0
pip install networkx==3.4.2 numpy==1.26.4 scipy==1.15.3 pandas==2.3.3
```

### 2.2 FINDER 专用环境：`finder_tf`

用于运行 FINDER 方法（TensorFlow 1.x 依赖）。

```bash
conda activate finder_tf
```

**Python 版本**：3.7.16  
**核心依赖**：

| 包 | 版本 | 用途 |
|----|------|------|
| tensorflow | 1.15.5 | FINDER 的 DQN + GraphSAGE 模型 |
| networkx | 2.6.3 | 图数据结构 |
| numpy | 1.18.5 | 数值计算 |
| protobuf | 3.19.6 | TF 1.x 兼容性（必须 < 3.20） |

**安装命令**：
```bash
conda create -n finder_tf python=3.7
conda activate finder_tf
pip install tensorflow==1.15.5 networkx==2.6.3 numpy==1.18.5
pip install "protobuf<3.20" --force-reinstall
```

### 2.3 外部依赖

| 依赖 | 说明 | 状态 |
|------|------|------|
| `CI.exe` | MinGW 编译的 Collective Influence 可执行文件 | 已编译，位于 `network_dismantling/CI/CI.exe` |
| `graph_tool` | 原 FINDER 代码依赖 | ❌ Windows 不可用，已用纯 Python 替代 |
| Cython/C++ | 原 FINDER 的 Cython 扩展 | ❌ Windows 编译失败，已用纯 Python 重写 |

---

## 3. 统一接口使用指南

### 3.1 基本调用

```python
from network_dismantling.unified_interface import dismantle, METHOD_REGISTRY
import networkx as nx

# 生成测试网络
G = nx.barabasi_albert_graph(100, 4, seed=42)

# 查看所有可用方法
print(METHOD_REGISTRY.keys())

# 调用拆解算法
seq = dismantle(G, method='CI_L2', stop_condition=10)
# seq: 节点移除顺序列表（原始节点 ID）
```

### 3.2 接口参数

```python
dismantle(G, method, stop_condition=None, **kwargs)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `G` | `networkx.Graph` | 输入图（有向图会自动转为无向图） |
| `method` | `str` | 方法名，必须在 `METHOD_REGISTRY` 中 |
| `stop_condition` | `int` | 当最大连通分量（LCC）≤ 此值时停止；若为 None 则拆完所有节点 |
| `**kwargs` | - | 方法特定参数（如 `seed`, `model_path` 等） |

### 3.3 图标准化

统一接口内部会自动执行：
1. 将图转为无向简单图（去除自环、重边）
2. 将节点重新编号为 `0, 1, ..., n-1`
3. 若方法返回的序列不完整，剩余节点按度降序补全
4. 将结果映射回原始节点 ID

---

## 4. 各算法详细说明

### 4.1 静态启发式（Heuristics）

基于节点中心性一次性计算分数，按分数降序移除。速度快，适合大规模网络。

#### `degree` — 度中心性
- **原理**：移除度数最高的节点
- **复杂度**：`O(n log n)`（排序）
- **参数**：无
- **适用场景**：大规模网络快速基准

#### `pagerank` — PageRank
- **原理**：基于 Google PageRank 算法的重要性排序
- **复杂度**：`O(n log n)`
- **参数**：无

#### `betweenness` — 介数中心性
- **原理**：经过该节点的最短路径数量
- **复杂度**：`O(n · m)`，大图极慢
- **参数**：无
- **注意**：BA-1000 上约需 100+ 秒，建议仅用于小规模网络

#### `eigenvector` — 特征向量中心性
- **原理**：邻接矩阵主特征向量对应的节点权重
- **复杂度**：`O(n log n)`
- **参数**：无

#### `random` — 随机移除
- **原理**：随机打乱节点顺序后移除
- **复杂度**：`O(n)`
- **参数**：`seed`（可选）

```python
seq = dismantle(G, 'degree', stop_condition=10)
seq = dismantle(G, 'random', stop_condition=10, seed=42)
```

---

### 4.2 精确解：`brute_force`

- **原理**：枚举所有大小为 k 的节点组合，找使 LCC 最小的组合
- **复杂度**：`O(C(n,k))`，仅适用于 n ≤ 30 的小网络
- **参数**：
  - `max_k=10`：最大搜索的节点组合大小
- **适用场景**：小网络的精确基准

```python
seq = dismantle(G, 'brute_force', stop_condition=5, max_k=8)
```

---

### 4.3 Collective Influence：`CI_L1`, `CI_L2`, `CI_L3`

- **原理**：`CI_ℓ(i) = (k_i - 1) Σ_{j∈∂B(i,ℓ)} (k_j - 1)`，考虑 ℓ-hop 邻居影响的节点重要性
- **实现**：调用预编译的 `CI.exe`（MinGW 编译，Windows 原生）
- **复杂度**：取决于 `ℓ` 和网络结构，通常 `O(n · d^ℓ)`
- **参数**：
  - `l`：已在方法名中固定（L1=1, L2=2, L3=3）
- **注意**：`CI.exe` 必须存在且可执行

```python
seq = dismantle(G, 'CI_L2', stop_condition=10)  # 最常用
```

---

### 4.4 CoreHD

- **原理**：三步策略：
  1. **Fix0**：从 2-core 中迭代移除最大度节点
  2. **大树拆分**：对超过阈值的大树组件，找使最大子树最小的分割点
  3. **贪心回插**：尝试将被移除节点加回，若不导致组件超标则保留
- **实现**：纯 Python/networkx，基于 C++ 源码重构
- **复杂度**：`O(n log n)` 到 `O(n^2)`，取决于网络结构
- **参数**：
  - `seed`：随机种子（用于 Fix0 阶段的节点平局打破）

```python
seq = dismantle(G, 'CoreHD', stop_condition=10, seed=42)
```

---

### 4.5 GND / EGND

#### `GND` — Generalized Network Dismantling
- **原理**：
  1. 提取最大连通分量（GCC）
  2. 用幂迭代计算 Fiedler 向量（图拉普拉斯第二小特征值对应的特征向量）
  3. 按 Fiedler 向量符号将 GCC 二分
  4. 构建跨分区的割边子图
  5. 对割边做贪心顶点覆盖
  6. 移除覆盖节点，重复直到 GCC ≤ stop_condition
- **实现**：纯 Python，使用 `scipy.sparse` 进行稀疏矩阵运算
- **复杂度**：每次迭代 `O(m · iter)`，其中 iter 是幂迭代次数（默认 `30·log(n)·√log(n)`）
- **参数**：
  - `remove_strategy=3`：3=无权（默认），1=加权
  - `maxiter`：幂迭代最大次数
  - `tol=1e-6`：收敛容差
  - `seed`：随机初始化种子

#### `EGND` — Ensemble GND
- **原理**：运行 GND 多次（默认 10 次），每次用不同随机种子，取移除节点数最少的结果
- **参数**：
  - `runs=10`：运行次数
  - `remove_strategy`：同 GND

```python
seq = dismantle(G, 'GND', stop_condition=10, seed=42)
seq = dismantle(G, 'EGND', stop_condition=10, runs=10, seed=42)
```

**注意**：GND 在 BA-1000 上约需 15 秒，是库中最慢的方法之一。

---

### 4.6 EI：`EI_s1`, `EI_s2`

- **原理**：Explosive Immunization，基于"有效度数"（effective degree）的贪心策略
  - 将节点按有效度数排序
  - 用并查集（DSU）高效追踪连通分量变化
  - 贪心移除使 LCC 下降最快的节点
- **实现**：纯 Python/networkx，去除了原 graph_tool 依赖
- **复杂度**：`O(n α(n))`，DSU 优化后接近线性
- **参数**：
  - `sigma`：已在方法名中固定（s1→σ=1, s2→σ=2）
  - `kk=1000`：迭代次数参数
  - `seed`：随机种子

```python
seq = dismantle(G, 'EI_s1', stop_condition=10, seed=42)
seq = dismantle(G, 'EI_s2', stop_condition=10, seed=42)
```

---

### 4.7 多尺度纠缠：`entanglement_small`, `entanglement_mid`, `entanglement_large`

- **原理**：基于信息论（Shannon 熵）的节点重要性度量。节点纠缠度越高，移除后对网络连通性的破坏越大。
- **实现**：networkx 版本，去除了原 graph_tool 依赖
- **复杂度**：取决于网络规模，通常 `O(n^2)` 到 `O(n^3)`
- **参数**：无

```python
seq = dismantle(G, 'entanglement_small', stop_condition=10)
```

---

### 4.8 顶点纠缠：`vertex_entanglement`

- **原理**：纯 numpy 谱计算，通过图的谱特性度量节点对网络连通性的贡献
- **实现**：`VertexEnt_nx`，纯 numpy + networkx，无需 graph_tool
- **复杂度**：谱计算主导，约 `O(n^3)` 或 `O(m·k)`（k 为迭代次数）
- **参数**：无

```python
seq = dismantle(G, 'vertex_entanglement', stop_condition=10)
```

---

### 4.9 GDM / GDM+R

#### `GDM` — Graph Dismantling Machine
- **原理**：
  1. 用 GAT（Graph Attention Network）一次性预测所有节点的拆解重要性分数
  2. 按分数降序逐步移除节点，检查 LCC 是否达标
- **模型**：3 层 GATConv，参数量约 12KB，训练于 4138 个真实网络（带 CoreHD 伪标签）
- **实现**：PyTorch + PyTorch Geometric
- **复杂度**：推理 `O(m·h)`（h=注意力头数），单次前向传播即可得全图分数
- **参数**：
  - `model_path`：模型权重路径（默认 `network_dismantling/GDM/models_newpg/gdm_nx_best.pth`）
  - `device`：推理设备（自动检测 CUDA，否则 CPU）

#### `GDM+R` — GDM + Reinsertion
- **原理**：在 GDM 初始序列基础上，尝试按逆序回插已移除节点。若回插后 LCC 仍 ≤ stop_condition，则保留该节点在图中。
- **效果**：通常使 rem_num 减少 10-30%
- **参数**：同 GDM

```python
seq = dismantle(G, 'GDM', stop_condition=10)
seq = dismantle(G, 'GDM+R', stop_condition=10)
# 或指定自定义模型
seq = dismantle(G, 'GDM+R', stop_condition=10, 
                model_path='path/to/custom_model.pth',
                device='cuda')
```

**注意**：GDM 模型权重仅 ~12KB，推理速度快（BA-1000 约 4.5 秒）。

---

### 4.10 FINDER

- **原理**：强化学习（DQN）+ GraphSAGE
  1. 每步用 GraphSAGE 计算节点 embedding
  2. action embedding × global embedding → Q-value
  3. 选 Q-value 最大的节点移除
  4. 重复直到所有边被覆盖
- **模型**：预训练于 n=30~50 的合成 BA 网络（`nrange_30_50_iter_78000.ckpt`）
- **实现**：TensorFlow 1.15 + 纯 Python 数据准备（已替代原 Cython 扩展）
- **复杂度**：每步需一次 TF 前向传播，BA-1000 约 1.6 秒
- **环境**：必须在 `finder_tf` conda 环境中运行
- **调用方式**：不通过统一接口，直接使用 `finder_pure.py`

```python
# 必须在 finder_tf 环境中运行
import sys
sys.path.insert(0, 'network_dismantling/FINDER_ND')
from finder_pure import FINDER_Pure
import networkx as nx

dqn = FINDER_Pure()
dqn.LoadModel('network_dismantling/FINDER_ND/models/nrange_30_50_iter_78000.ckpt')

G = nx.barabasi_albert_graph(100, 4, seed=42)
solution, time_cost = dqn.EvaluateRealData(G, stepRatio=0.01)
```

**跨环境限制**：FINDER 无法直接注册到 `kanResilience` 环境的统一接口中（Python 3.7 vs 3.10 不兼容）。若需在统一对比中使用，建议：
1. 在 `finder_tf` 中独立运行 FINDER
2. 将结果保存为 CSV/JSON
3. 在 `kanResilience` 中读取合并

---

## 5. 结果文件说明

### 5.1 重复实验结果

| 文件 | 内容 | 行数 |
|------|------|------|
| `results/results_repeated_main.csv` | 6 方法 × 3 网络 × 7 规模 × 20 次 | 2521（含表头） |
| `results/results_repeated_finder.csv` | FINDER × 3 网络 × 7 规模 × 20 次 | 421（含表头） |
| `results/dismantling_results.xlsx` | 合并后的均值/标准差/透视表 | 6 个 Sheet |

### 5.2 Excel 文件结构

| Sheet | 说明 |
|-------|------|
| `Summary_Mean_Std` | 均值 ± 标准差（完整表格，4 位小数） |
| `Mean_Only` | 仅均值（论文用简洁版） |
| `Stddev_Only` | 仅标准差 |
| `Raw_Data` | 2940 行原始实验数据 |
| `AUC_Pivot` | 方法 × 网络的 AUC 均值透视表 |
| `RemNum_Pivot` | 方法 × 网络的 rem_num 均值透视表 |

---

## 6. 已知限制与注意事项

### 6.1 环境隔离

| 方法 | 所需环境 | 原因 |
|------|---------|------|
| degree, CI_L2, CoreHD, GND, EI, entanglement, vertex_entanglement, GDM, GDM+R | `kanResilience` | Python 3.10 + PyTorch 2.3 |
| FINDER | `finder_tf` | Python 3.7 + TensorFlow 1.15 |

**无法跨环境 import**。FINDER 的 `.ckpt` 权重格式为 TF 1.x 专有，无法加载到 PyTorch 中。

### 6.2 方法适用性

| 方法 | 小图 (n<100) | 中图 (100-500) | 大图 (500+) | 备注 |
|------|-------------|---------------|------------|------|
| brute_force | ✅ 精确 | ❌ 不可行 | ❌ 不可行 | 仅 n ≤ 30 |
| betweenness | ✅ | ⚠️ 慢 | ❌ 极慢 | BA-1000 需 100+s |
| GND | ✅ | ⚠️ 慢 | ❌ 极慢 | BA-1000 需 15+s |
| CI_L2 | ✅ | ✅ | ✅ | 综合最优 |
| CoreHD | ✅ | ✅ | ✅ | ER 上优于 BA |
| GDM/GDM+R | ✅ | ✅ | ✅ | 需 PyTorch |
| FINDER | ✅ | ✅ | ✅ | 需 TF 1.x |
| degree | ✅ | ✅ | ✅ | 最快但效果一般 |

### 6.3 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `CI.exe not found` | MinGW 未编译或文件被移动 | 确认 `network_dismantling/CI/CI.exe` 存在 |
| `ModuleNotFoundError: torch` | 未激活 `kanResilience` | `conda activate kanResilience` |
| `tf.sparse_placeholder` 警告 | TF 1.x API 已弃用 | 可忽略，不影响运行 |
| FINDER `Key Variable_X not found` | 多次创建 `FINDER_Pure` 实例 | 只创建一个实例并复用 |
| GDM 模型路径错误 | `gdm_nx_best.pth` 不存在 | 确认路径 `network_dismantling/GDM/models_newpg/` |

### 6.4 图输入要求

- 统一接口接受任意 `networkx.Graph`
- 有向图会自动转为无向图
- 自环和重边会自动去除
- 节点 ID 可为任意可哈希类型（字符串、整数等），内部会自动重编号

---

## 7. 快速参考：一行代码调用所有方法

```python
import networkx as nx
from network_dismantling.unified_interface import dismantle, METHOD_REGISTRY

G = nx.barabasi_albert_graph(100, 4, seed=42)

# 所有 kanResilience 环境方法
methods = list(METHOD_REGISTRY.keys())  # 20 个方法
for method in methods:
    seq = dismantle(G, method, stop_condition=10)
    print(f"{method}: removed {len(seq)} nodes")

# FINDER（需在 finder_tf 环境中单独运行）
# 见 4.10 节示例代码
```

---

## 8. 引用与致谢

| 方法 | 原始论文 |
|------|---------|
| CI | Morone & Makse, *Nature* 2015 |
| CoreHD | Zdeborová et al., *Sci. Rep.* 2016 |
| GND/EGND | Wandelt et al., *Nature Commun.* 2018 |
| EI | Radicchi & Bianconi, *Phys. Rev. E* 2017 |
| GDM | Fan et al., *PNAS* 2020 |
| FINDER | Fan et al., *NeurIPS* 2020 |
| Entanglement | 信息论网络鲁棒性系列工作 |

---

*文档生成时间：2026-05-21*  
*项目版本：v3-dismantling*
