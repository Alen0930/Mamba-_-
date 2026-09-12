# 算法支持状态汇总

## 环境
- **OS**: Windows 10/11 (原生)
- **Python**: 3.10 (conda 环境 `kanResilience`)
- **缺失包**: `graph_tool` (Windows 原生不可安装)、`Cython`、`TensorFlow 1.x`

---

## ✅ 已支持的算法（共 18 个）

| 类别 | 方法名 | 实现方式 |
|------|--------|----------|
| **启发式** | `degree`, `pagerank`, `betweenness`, `eigenvector`, `random` | networkx 中心性 + 静态排序 |
| **暴力搜索** | `brute_force` | 小网络枚举 |
| **纠缠类** | `entanglement_small`, `entanglement_mid`, `entanglement_large` | networkx 谱计算 |
| **顶点纠缠** | `vertex_entanglement` | numpy 谱计算 |
| **集体影响** | `CI_L1`, `CI_L2`, `CI_L3` | mingw64 编译 `CI.exe` + subprocess |
| **CoreHD** | `CoreHD` | ✅ **Python 重构**（2-core + tree breaking + 贪心回插） |
| **GND** | `GND` | ✅ **Python 重构**（谱二分 + 割边顶点覆盖） |
| **EGND** | `EGND` | ✅ **Python 重构**（多次 GND 取最优） |
| **EI** | `EI_s1`, `EI_s2` | ✅ **Python 重构**（Newman-Ziff DSU + 贪心选择） |

---

## ❌ 暂不支持

| 算法 | 原因 |
|------|------|
| **decycler** (Min-Sum + BP) | 算法基于 Reinforced Max-Sum Belief Propagation，涉及大量迭代消息传递。纯 Python 实现**功能可行但性能极差**（比 C++/OpenMP 慢 2~3 个数量级）。如需使用，建议：① 安装 Boost 后用 mingw64 编译原 C++ 代码；② 或仅用于 <1000 节点的小网络并配合 Numba 加速。 |
| **GDM / CoreGDM** | ① 深度绑定 `graph_tool`；② 仓库中**缺少预训练 PyTorch 权重**（`models_newpg/` 为空）。 |
| **FINDER_ND** | 依赖 TensorFlow 1.x，Python 3.10 无法安装。建议单独创建 Python 3.7 + TF1.x 环境。 |

---

## 使用方式

```python
from network_dismantling.unified_interface import dismantle
import networkx as nx

G = nx.barabasi_albert_graph(100, 3, seed=42)

# 任意已支持的方法
seq = dismantle(G, method="CoreHD", stop_condition=5)
seq = dismantle(G, method="GND", stop_condition=5)
seq = dismantle(G, method="EGND", stop_condition=5, runs=10)
seq = dismantle(G, method="EI_s2", stop_condition=5)
```

所有方法统一返回 `List[int]`，长度为 |V|，按拆解优先级排序。
