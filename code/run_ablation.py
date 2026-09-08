"""
Mamba 网络拆解 —— 消融实验脚本

回答核心问题：模型到底学到了 degree，还是学到了中心性特征（k-core/PageRank/closeness）？

四组对比（唯一变量是特征维度）：
    A. degree                —— 纯启发式基线
    B. mamba (input_dim=1)   —— 仅 degree 特征（需训练，新增）
    C. mamba (input_dim=4)   —— 度 + k-core + PageRank + closeness（已有 best_model.pth）
    D. mamba (随机初始化)     —— 4 维，说明"训练学到了什么"

关键严谨性设计：
    B 与 C 使用【完全相同的图、相同的 CoreHD 标签、相同的节点排序】，
    唯一区别是特征维度（1 vs 4）。做法：直接从已有 4 维数据集的缓存里
    切片出第 0 列（degree，已 min-max 归一化）作为 1 维特征，不重新生成数据集、
    不重新跑 CoreHD，保证可比性。

用法:
    # 训练 1 维模型 + 四组评测
    python run_ablation.py

    # 已训练好 1 维模型，只评测
    python run_ablation.py --skip-train
"""
import argparse
import io
import sys
from typing import List

import numpy as np
import networkx as nx

from network_dismantling.Mamba.dataset_generator import load_datasets, build_dataloaders
from network_dismantling.Mamba.trainer import create_trainer, DismantlingDataset
from network_dismantling.Mamba.mamba_dismantler import mamba_dismantle
from network_dismantling.unified_interface import dismantle
from eval_after_train import build_test_graph
from evaluate import calc_metrics, plot_robustness

# ---------------------------------------------------------------------------
# 默认配置（与 run_full_train.py 完全一致，保证唯一变量是特征维度）
# ---------------------------------------------------------------------------
DATASET_DIR = "datasets/ba_corehd_full"
CHECKPOINT_4D = "checkpoints/full_train/best_model.pth"   # 已有 4 维模型
CHECKPOINT_1D = "checkpoints/ablation_degree/best_model.pth"  # 新增 1 维模型
NUM_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
D_MODEL = 64
N_LAYERS = 2


# ---------------------------------------------------------------------------
# 1 维特征切片
# ---------------------------------------------------------------------------
def slice_to_degree(dataset: DismantlingDataset) -> DismantlingDataset:
    """
    从 4 维特征数据集切片出 1 维（degree）特征数据集，复用相同标签与排序。

    原理：4 维特征的第 0 列即 degree（feature_encoding 中
    features = stack([degrees, cores, pageranks, closeness])），且统一按度降序
    排序、min-max 归一化。故第 0 列与 feature_set='degree' 单独提取的结果
    数学等价，可直接切片复用，无需重新跑 CoreHD。
    """
    if dataset.cached_data is None:
        raise ValueError("数据集未缓存，无法切片（需 cache_features=True）")

    new_cached = []
    for sample in dataset.cached_data:
        new_cached.append({
            'features': sample['features'][:, :1].copy(),  # (n, 1) 仅 degree 列
            'node_ids': sample['node_ids'],
            'ranks': sample['ranks'],
        })

    # 复用原 graphs，仅替换 cached_data（不触发预处理、不重跑标签）
    new_dataset = DismantlingDataset(
        dataset.graphs, dismantler_fn=None, cache_features=False, feature_set='degree'
    )
    new_dataset.cached_data = new_cached
    new_dataset.cache_features = True
    return new_dataset


# ---------------------------------------------------------------------------
# 训练 1 维模型
# ---------------------------------------------------------------------------
def train_degree_model(args, device: str):
    """训练 input_dim=1（仅 degree）的 Mamba 模型，配置与 4 维训练完全一致"""
    print("\n" + "=" * 76)
    print("阶段 1/2：训练 1 维（degree-only）Mamba 模型")
    print("=" * 76)

    train_ds_4d, val_ds_4d = load_datasets(args.dataset_dir)
    print(f"加载 4 维数据集: train={len(train_ds_4d)}, val={len(val_ds_4d)}")

    print("切片出 1 维 degree 特征（复用相同图/标签/排序）...")
    train_ds = slice_to_degree(train_ds_4d)
    val_ds = slice_to_degree(val_ds_4d)
    # 校验维度
    assert train_ds.cached_data[0]['features'].shape[1] == 1, "切片后特征维度应=1"

    train_loader, val_loader = build_dataloaders(
        train_ds, val_ds, batch_size=args.batch_size
    )

    trainer = create_trainer(
        input_dim=1,  # 唯一与 4 维训练不同之处
        d_model=args.d_model,
        n_layers=args.n_layers,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir_1d,
    )
    print(f"模型参数量: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print("-" * 76)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        patience=args.patience,
        verbose=True,
    )
    print(f"\n1 维模型训练完成，最佳验证损失: {trainer.best_val_loss:.4f}")
    print(f"检查点: {args.checkpoint_dir_1d}/best_model.pth")


# ---------------------------------------------------------------------------
# 四组评测
# ---------------------------------------------------------------------------
def run_ablation_eval(args, device: str):
    """在独立测试网络上评测四组，输出对比表与曲线"""
    print("\n" + "=" * 76)
    print("阶段 2/2：四组消融评测")
    print("=" * 76)

    G = build_test_graph("ba", args.n, args.m, args.seed)
    print(f"测试网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 条边")

    sequences = {}
    metrics = {}

    # A. degree 基线
    print("\n[A] degree 基线...")
    sequences['degree'] = dismantle(G, method='degree')
    metrics['degree'] = calc_metrics(G, sequences['degree'])

    # B. 1 维 mamba（训练后）
    print("[B] mamba (input_dim=1, 训练后)...")
    sequences['mamba_1d'] = mamba_dismantle(
        G, model_path=args.checkpoint_1d, feature_set='degree', device=device
    )
    metrics['mamba_1d'] = calc_metrics(G, sequences['mamba_1d'])

    # C. 4 维 mamba（训练后）
    print("[C] mamba (input_dim=4, 训练后)...")
    sequences['mamba_4d'] = mamba_dismantle(
        G, model_path=args.checkpoint_4d, feature_set='full', device=device
    )
    metrics['mamba_4d'] = calc_metrics(G, sequences['mamba_4d'])

    # D. 随机初始化 mamba（4 维）
    print("[D] mamba (随机初始化)...")
    sequences['mamba_random'] = mamba_dismantle(G, device=device)
    metrics['mamba_random'] = calc_metrics(G, sequences['mamba_random'])

    # 输出对比表
    print("\n" + "=" * 76)
    print("消融对比表 (stop_ratio=0.1, fc_threshold=0.01)")
    print("=" * 76)
    print(f"{'方法':<24} {'Stop步数':<12} {'FC值':<12} {'达到Stop':<10} {'达到FC':<10}")
    print("-" * 76)
    for name, (stop, fc, rs, rf) in metrics.items():
        print(f"{name:<24} {stop:<12} {fc:<12.4f} {str(rs):<10} {str(rf):<10}")
    print("=" * 76)

    # 关键结论：1 维 vs 4 维
    stop_1d, fc_1d = metrics['mamba_1d'][0], metrics['mamba_1d'][1]
    stop_4d, fc_4d = metrics['mamba_4d'][0], metrics['mamba_4d'][1]
    print("\n>>> 关键结论（1 维 vs 4 维）:")
    print(f"    Stop 步数: 1维={stop_1d} vs 4维={stop_4d} "
          f"({(stop_1d - stop_4d) / stop_4d * 100:+.1f}% 相对差异)")
    print(f"    FC 值:     1维={fc_1d:.4f} vs 4维={fc_4d:.4f}")
    if abs(stop_1d - stop_4d) <= max(3, stop_4d * 0.03):
        print("    => 两者接近：增益几乎全部来自 degree，中心性特征贡献很小（negative result）")
    else:
        print("    => 4 维显著更优：中心性特征（k-core/PR/closeness）提供了额外信息")

    # 鲁棒性曲线
    plot_names = {
        'degree': 'degree',
        'mamba_1d': 'mamba (degree-only)',
        'mamba_4d': 'mamba (4 features)',
        'mamba_random': 'mamba (random)',
    }
    plot_dict = {plot_names[k]: v for k, v in sequences.items()}
    plot_robustness(G, plot_dict, sample_step=10, save_path=args.output, dpi=300)
    print(f"\n鲁棒性对比曲线已保存: {args.output}")

    return sequences, metrics


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True
        )

    parser = argparse.ArgumentParser(description="Mamba 网络拆解消融实验")
    parser.add_argument("--dataset-dir", type=str, default=DATASET_DIR, help="4 维数据集目录")
    parser.add_argument("--checkpoint-4d", type=str, default=CHECKPOINT_4D, help="4 维模型检查点")
    parser.add_argument("--checkpoint-1d", type=str, default=CHECKPOINT_1D, help="1 维模型检查点")
    parser.add_argument("--checkpoint-dir-1d", type=str, default="checkpoints/ablation_degree",
                        help="1 维模型训练检查点目录")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="最大训练轮数")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="早停耐心值")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="批次大小")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="权重衰减")
    parser.add_argument("--d-model", type=int, default=D_MODEL, help="隐藏维度")
    parser.add_argument("--n-layers", type=int, default=N_LAYERS, help="Mamba 层数")
    parser.add_argument("--n", type=int, default=1000, help="测试网络节点数")
    parser.add_argument("--m", type=int, default=3, help="测试网络连边参数")
    parser.add_argument("--seed", type=int, default=2026, help="测试网络随机种子")
    parser.add_argument("--device", type=str, default=None, help="设备 (默认自动)")
    parser.add_argument("--output", type=str, default="results/ablation_comparison.png",
                        help="对比曲线输出路径")
    parser.add_argument("--skip-train", action="store_true", help="跳过 1 维模型训练，直接评测")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 76)
    print("Mamba 网络拆解 —— 消融实验")
    print("=" * 76)
    print(f"设备: {device}")
    print(f"对比组: degree | mamba(1维) | mamba(4维) | mamba(随机)")

    if not args.skip_train:
        train_degree_model(args, device)
    else:
        print(f"\n跳过训练，使用已有 1 维模型: {args.checkpoint_1d}")

    run_ablation_eval(args, device)

    print("\n消融实验完成。")


if __name__ == "__main__":
    main()
