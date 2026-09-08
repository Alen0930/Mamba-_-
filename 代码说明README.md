# Mamba 网络拆解 —— 阶段性成果交付说明

基于状态空间模型（Mamba）的网络拆解算法完整实现：从 GNN+双向 Mamba 动态重算，到 RL(PPO) 强化学习微调，突破「拟合 CoreHD 标签」的行为克隆上界。

**核心成果**（n=500/1000/1500，各 3 seeds，9 图平均）：

| 方法 | 平均 stop_step | 平均 fc_value |
|---|---|---|
| degree | 270.0 | 0.3799 |
| CoreHD（动态上界） | 224.1 | 0.3995 |
| gnn-dyn-b25（监督） | 252.3 | 0.3406 |
| **rl-greedy（RL 微调）** | **239.8** | **0.3079** |

- **fc_value（LCC≤1% 彻底瓦解，更严格指标）：RL 大幅超越 CoreHD 23%**（0.3079 vs 0.3995），也超监督模型 9.6%。
- stop_step（LCC≤10%）：RL 逼近 CoreHD（差 7%），明显优于监督模型与 degree。

> 完整四阶段研究过程（消融 → GNN 重构 → 动态重算 → RL）见同目录 `阶段性成果.md`。

---

## 一、交付包结构

```
Mamba 网络拆解_阶段性成果/
├── 阶段性成果.md                  ← 四阶段完整研究过程与结果（消融→GNN→动态重算→RL）
├── 代码说明README.md              ← 本文件
├── code/                          ← 项目根目录（所有命令在此目录下执行）
│   ├── network_dismantling/       ← 完整代码包
│   │   ├── unified_interface.py   ← 统一拆解接口（含【Mamba 对接-修改部分】标注）
│   │   ├── Mamba/                 ← Mamba 模块（本项目核心新增）
│   │   │   ├── feature_encoding.py      ← 节点特征编码（degree 输入）
│   │   │   ├── mamba_model.py           ← 纯 Mamba 评分模型（2 层 SSM）
│   │   │   ├── gnn_mamba_model.py       ← GCN 编码器 + 双向 Mamba（encode/forward 分离）
│   │   │   ├── gnn_mamba_actor_critic.py ← Actor-Critic（复用编码器 + value_head）
│   │   │   ├── rl_env.py                ← RL 环境与 rollout 采样（批动作 + potential reward）
│   │   │   ├── ppo_trainer.py           ← PPO 训练器（GAE / REINFORCE / per-step potential）
│   │   │   ├── mamba_dismantler.py      ← 拆解主逻辑（含动态重算 batch_size 参数）
│   │   │   ├── trainer.py               ← 监督训练引擎（ListMLE 损失）
│   │   │   ├── dataset_generator.py     ← BA 数据集批量生成器
│   │   │   ├── model_io.py              ← 模型权重管理工具
│   │   │   └── *.md                     ← 模块文档
│   │   └── （heuristics / CoreHD / GND / EI / CI 等原有方法）
│   ├── run_ablation_train.py     ← 消融实验训练
│   ├── run_full_train.py         ← 纯 Mamba 训练（复现命令②）
│   ├── run_gnn_train.py          ← GNN+双向 Mamba 训练
│   ├── run_rl_train.py           ← RL(PPO) 训练（复现命令④）
│   ├── prepare_gnn_dataset.py    ← GNN 数据集准备
│   ├── eval_after_train.py       ← 训练后效果评测
│   ├── eval_dynamic.py           ← 动态重算评测
│   ├── eval_rl.py                ← RL 正式评测（复现命令⑤）
│   ├── train_mamba_example.py    ← 训练示例脚本
│   └── evaluate.py               ← 评测工具（calc_metrics / plot_robustness）
├── results/
│   ├── rl_eval.csv               ← RL 正式评测数据（n=500/1000/1500）
│   ├── dynamic_eval*.csv         ← 动态重算评测数据
│   ├── ablation*.csv             ← 消融实验数据
│   └── 指标对比表.txt / 鲁棒性对比曲线.png / 训练日志.txt
└── checkpoints/
    ├── full_train/best_model.pth           ← 纯 Mamba 训练权重
    ├── gnn_mamba/best_model.pth            ← 监督预训练权重（RL 初始策略）
    └── gnn_mamba_rl500v4/best_model.pth    ← RL 微调最终权重
```

> 说明：`code/` 目录即项目根目录，Python 包名保持 `network_dismantling` 不变，
> 所有命令均在 `code/` 目录下执行，保证 import 路径与开发环境一致、代码零修改可运行。

---

## 二、环境依赖

### 2.1 核心依赖版本

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | **3.12.10** | 必须 3.12（预编译 wheel 为 cp312） |
| PyTorch | **2.9.1+cu128** | RTX 5060 (sm_120) 必须 ≥2.7+cu128 |
| mamba-ssm | **2.2.6.post3** | Windows 预编译 wheel（本地安装） |
| triton-windows | **3.5.1.post22** | 实测在 sm_120 上正常 |
| transformers | **<5**（实测 4.57.6） | 5.x 移除 GreedySearchDecoderOnlyOutput 导致 mamba_ssm import 失败 |
| causal_conv1d | **不安装** | 见 2.3 兼容性说明（触发 torch conv1d 回退） |
| einops | 0.8.2 | mamba_ssm 运行依赖 |
| networkx | 3.6.1 | 图处理 |
| numpy | 2.5.2 | 特征计算 |
| matplotlib | 3.11.1 | 曲线绘图 |
| scipy / pandas / tqdm | 1.18.1 / 3.0.5 / 4.70.0 | 评测与日志 |
| fastnumbers / humanize | 5.2.0 / 4.16.0 | 原有方法依赖 |

### 2.2 安装方式

```bash
# 1. 创建 Python 3.12 虚拟环境
python3.12 -m venv venv312
source venv312/Scripts/activate        # Windows (Git Bash)
# 或: venv312\Scripts\activate.bat    # Windows (CMD)

# 2. 安装 PyTorch（阿里云 pytorch-wheels 镜像下载 wheel 后本地安装）
pip install torch-2.9.1+cu128-cp312-cp312-win_amd64.whl

# 3. 安装 triton
pip install triton-windows==3.5.1.post22

# 4. 安装 mamba_ssm（--no-deps 跳过依赖；不安装 causal_conv1d）
pip install mamba_ssm-2.2.6.post3-cp312-cp312-win_amd64.whl --no-deps

# 5. 安装其余依赖
pip install "transformers<5" einops networkx numpy matplotlib scipy pandas tqdm fastnumbers humanize
```

> 三个 wheel 文件（torch / mamba_ssm）位于开发环境项目根目录，交付时随附或从原获取渠道下载。

### 2.3 RTX 5060 显卡兼容性配置（重要）

RTX 5060 为 Blackwell **sm_120** 架构，预编译的 mamba_ssm wheel 最高支持 sm_100，
其 CUDA kernel 无法在 sm_120 上运行（报错 `no kernel image is available`）。
本项目的兼容性方案（已在 `mamba_model.py` 中内置，**无需手动配置**）：

1. **禁用 Mamba fast path**：所有 Mamba 层构造时传 `use_fast_path=False`；
2. **selective_scan 回退**：模块加载时自动将 `selective_scan_fn` 替换为
   `selective_scan_ref`（PyTorch 原生算子实现的参考版本，GPU/CPU 均可运行）；
3. **不安装 causal_conv1d**：卸载/不安装后，mamba_ssm 自动回退到 `torch.nn.functional.conv1d`。

以上三项保证 Mamba 前向/反向在 RTX 5060 上全程可用（本次训练 65 分钟即为其上实测）。

---

## 三、复现步骤

以下三条命令在 `code/` 目录下顺序执行，即可完整复现本次实验
（生成数据集 → 训练模型 → 评测效果）。

### 命令①：生成 BA 训练数据集

```bash
python -m network_dismantling.Mamba.dataset_generator \
    --num-samples 100 --n-min 500 --n-max 1500 --m-min 2 --m-max 5 \
    --split-ratio 0.8 --out-dir datasets/ba_corehd_full
```

- 生成 100 个 BA 无标度网络（节点数 500-1500 随机、连边参数 2-5 随机）
- 内部调用 **CoreHD** 生成拆解标签，按 8:2 自动划分为训练集 80 个 / 验证集 20 个
- 输出 `datasets/ba_corehd_full/train_dataset.pkl` 与 `val_dataset.pkl`

### 命令②：启动完整模型训练

```bash
python run_full_train.py \
    --dataset-dir datasets/ba_corehd_full \
    --epochs 100 --patience 15 --batch-size 8 \
    --checkpoint-dir checkpoints/full_train
```

- 模型：2 层 Mamba（d_model=64），AdamW（lr=1e-3，weight_decay=1e-4），ListMLE 排序损失
- 每 epoch 约 1 分钟，完整训练约 65 分钟（本次实测 1.08 小时，epoch 66 触发早停）
- 最佳模型自动保存到 `checkpoints/full_train/best_model.pth`
- 训练日志参考 `../results/训练日志.txt`

### 命令③：运行训练后效果评测

```bash
python eval_after_train.py \
    --checkpoint checkpoints/full_train/best_model.pth \
    --n 1000 --m 3 --seed 2026 \
    --output results/鲁棒性对比曲线.png
```

- 在独立 1000 节点 BA 测试网络（seed=2026，与训练集无交集）上评测
- 输出五种方法（随机 Mamba / degree / pagerank / CoreHD / 训练后 Mamba）的
  Stop 步数与 FC 值对比表，并生成鲁棒性对比曲线图
  （`--output` 为相对 `code/` 的路径，即保存到 `code/results/` 下）
- 本次实验的完整评测数据见 `../results/指标对比表.txt`

### 命令④：GNN+双向 Mamba 监督训练

```bash
python run_gnn_train.py \
    --dataset-dir datasets/ba_corehd_full \
    --epochs 100 --batch-size 8 \
    --checkpoint-dir checkpoints/gnn_mamba
```

- GCN 编码器（dense 邻接矩阵消息传递）+ 双向 Mamba，输入只用 degree
- 最佳模型保存到 `checkpoints/gnn_mamba/best_model.pth`（同时作为 RL 初始策略）

### 命令⑤：RL(PPO) 微调训练（突破行为克隆上界）

```bash
python run_rl_train.py \
    --ckpt checkpoints/gnn_mamba/best_model.pth \
    --n 500 --action-k 10 --num-envs 4 --iterations 100 \
    --eval-interval 20 --eval-sizes 500 --eval-seeds 3 \
    --lr 2e-4 --ent-coef 0.001 \
    --out-dir checkpoints/gnn_mamba_rl500v4
```

- 在监督预训练模型之上，用 per-step potential advantage（`(LCC_before-LCC_after)/n0`）
  做 PPO 策略梯度微调，reward 换成「每步 LCC 下降 / 拆解集大小」
- 最佳模型保存到 `checkpoints/gnn_mamba_rl500v4/best_model.pth`
- 训练全程 fc 单调下降、无退化（advantage 设计的三次迭代见 `阶段性成果.md` 第五节）

### 命令⑥：RL 正式评测（n=500/1000/1500）

```bash
python eval_rl.py \
    --ckpt checkpoints/gnn_mamba_rl500v4/best_model.pth \
    --gnn-ckpt checkpoints/gnn_mamba/best_model.pth \
    --n-sizes 500,1000,1500 --seeds-per-size 3
```

- 对比 degree / CoreHD / gnn-dyn-b25（监督）/ rl-greedy（RL）四种方法
- 结果保存到 `results/rl_eval.csv`，核心结论见 README 顶部表格

---

## 四、接口使用方式

训练好的权重加载后，Mamba 方法通过统一接口 `dismantle()` 调用，
与 degree、CoreHD 等方法的调用方式**完全一致**：

```python
import networkx as nx
from network_dismantling.unified_interface import dismantle
from network_dismantling.Mamba.model_io import register_model_for_inference

# ---------- 1. 加载训练好的权重 ----------
# 方式 A：全局注册（之后所有 mamba 调用自动使用训练权重，推荐）
register_model_for_inference("checkpoints/full_train/best_model.pth")

# 方式 B：显式指定检查点（无需注册，优先级最高）
# dismantle(G, method='mamba', model_path="checkpoints/full_train/best_model.pth")

# ---------- 2. 通过统一接口调用 ----------
G = nx.barabasi_albert_graph(1000, 3, seed=42)

seq_mamba   = dismantle(G, method='mamba')     # 训练后 Mamba（与 degree 等调用方式一致）
seq_degree  = dismantle(G, method='degree')
seq_corehd  = dismantle(G, method='CoreHD')

# ---------- 3. 评测 ----------
from evaluate import calc_metrics, plot_robustness

stop_step, fc_value, reach_stop, reach_fc = calc_metrics(G, seq_mamba)

plot_robustness(
    G,
    {'degree': seq_degree, 'CoreHD': seq_corehd, 'mamba (trained)': seq_mamba},
    sample_step=10,
    save_path="robustness.png",
)
```

**可用的拆解方法**：`degree`、`pagerank`、`betweenness`、`eigenvector`、`random`、
`CoreHD`、`GND`、`EI_s1`、`CI_L1`、`vertex_entanglement`、`mamba` 等
（全部通过 `dismantle(G, method=...)` 统一调用）。

**权重加载优先级**：`model_path` 参数 > 全局注册权重 > 随机初始化。
取消注册可用 `unregister_model()` 恢复随机初始化。

---

## 五、本次实验总结

### 5.1 训练配置

| 项目 | 配置 |
|------|------|
| 训练数据 | 100 个 BA 无标度网络（500-1500 节点，m∈[2,5]），8:2 划分（80/20） |
| 监督信号 | CoreHD 拆解序列（ListMLE 排序标签） |
| 节点特征 | 度、k-core、PageRank、接近中心性（4 维，min-max 归一化，按度降序排列） |
| 模型结构 | 2 层 Mamba（d_model=64，d_state=16，d_conv=4，expand=2）+ 线性输出层，共 65,665 参数 |
| 损失函数 | ListMLE 排序损失（向量化实现，logcumsumexp 后缀累积） |
| 优化器 | AdamW（lr=1e-3，weight_decay=1e-4），梯度裁剪 max_norm=1.0 |
| 训练设置 | 100 epochs 上限，早停 patience=15，batch_size=8 |
| 硬件 | NVIDIA GeForce RTX 5060 Laptop GPU（8.5 GB 显存），峰值占用约 3.3 GB |

### 5.2 训练过程

- 总耗时 **65 分钟**（epoch 66 触发早停，最佳验证损失出现在 **epoch 51**）
- 训练损失 5518 → 4653（持续下降）；验证损失 6800 → **5768.8**（epoch 51 后回升触发早停）

### 5.3 模型效果（1000 节点独立测试网络，seed=2026）

| 方法 | Stop 步数 | FC 值 | 相对随机 Mamba 提升 |
|------|----------|-------|--------------------|
| mamba（随机初始化） | 720 | 0.843 | — |
| degree | 276 | 0.390 | — |
| pagerank | 267 | 0.382 | — |
| **mamba（训练后）** | **261** | **0.377** | Stop +63.7%，FC +55.3% |
| CoreHD | 238 | 0.438 | — |

### 5.4 核心结论

1. **端到端管线全部打通**：BA 数据集生成 → CoreHD 标签 → ListMLE 训练 → 权重注册 →
   统一接口调用 → 效果评测，全流程可复现（见第三节三条命令）。
2. **训练后的 Mamba 超越经典启发式基线**：Stop 步数优于 degree（-5.4%）与
   pagerank（-2.2%），逼近监督信号 CoreHD（差 9.7%）。
3. **瓦解阈值指标全面最优**：FC 值 0.377 优于所有对比方法（含 CoreHD 的 0.438），
   说明模型学到的策略在"快速瓦解网络"维度上泛化良好。
4. **模型具备跨网络泛化能力**：测试网络（seed=2026）与训练集（随机生成、无固定种子）
   完全独立，效果无退化，证明学习到的是通用拓扑拆解规律而非记忆特定网络。

### 5.5 后续进展：GNN 重构 → 动态重算 → RL 突破

纯 Mamba（阶段 A/B）之后，项目进一步推进了三个阶段（详见 `阶段性成果.md`）：

1. **GNN + 双向 Mamba 重构**：新增 GCN 编码器（真正用上邻接矩阵），输入只用 degree。
2. **动态重算**：每移除 b 个节点在剩余子图上重算分数，stop_step 269.9→252.3（改善 6.5%），
   fc_value 反超 CoreHD。
3. **RL(PPO) 突破行为克隆上界**：监督预训练模型作初始策略，PPO 用真实拆解 reward 微调。
   经三次 advantage 设计迭代（critic/GAE 与 REINFORCE 均退化），最终采用
   per-step potential advantage。成果：**fc_value 大幅超越 CoreHD 23%**，
   stop_step 逼近 CoreHD（差 7%），完整数据见 README 顶部表格与 `results/rl_eval.csv`。

---

*交付日期：2026-09-08｜ 项目：Mamba 网络拆解（网络拆解算法研究课题组）*
