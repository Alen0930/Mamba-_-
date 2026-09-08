"""
消融实验：训练不同特征集（degree-only / all）的 Mamba 模型

用于回答核心问题「模型到底学到了排序本身，还是只读出了中心性特征」：

    --feature-set degree  -> 仅用节点度作为输入（input_dim=1），
                             与 4 特征模型对比，隔离中心性特征的贡献
    --feature-set all     -> 度 + k-core + PageRank + 接近中心性（input_dim=4），
                             即当前完整模型

复用 datasets/ba_corehd_full 的图与 CoreHD 标签（feature_set='degree' 时在取样本阶段
对缓存特征取度子集，无需重新生成标签）。

用法:
    python run_ablation_train.py --feature-set degree
    python run_ablation_train.py --feature-set all --checkpoint-dir checkpoints/ablation_4feat
"""
import argparse
import io
import sys
import time
from datetime import datetime


def main():
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True
        )

    parser = argparse.ArgumentParser(description="消融实验：训练 degree-only / all 特征的 Mamba")
    parser.add_argument("--feature-set", type=str, default="degree", choices=["degree", "all"],
                        help="特征集（degree: 仅度；all: 度+kcore+PageRank+closeness）")
    parser.add_argument("--dataset-dir", type=str, default="datasets/ba_corehd_full",
                        help="数据集目录（复用 CoreHD 标签）")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="检查点目录（默认 checkpoints/ablation_<feature_set>）")
    parser.add_argument("--epochs", type=int, default=100, help="最大训练轮数")
    parser.add_argument("--patience", type=int, default=15, help="早停耐心值")
    parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--d-model", type=int, default=64, help="隐藏维度")
    parser.add_argument("--n-layers", type=int, default=2, help="Mamba 层数")
    parser.add_argument("--device", type=str, default=None, help="设备 (默认自动)")
    args = parser.parse_args()

    import torch
    from network_dismantling.Mamba.dataset_generator import load_datasets, build_dataloaders
    from network_dismantling.Mamba.trainer import create_trainer

    feature_set = args.feature_set
    input_dim = 1 if feature_set == "degree" else 4
    checkpoint_dir = args.checkpoint_dir or f"checkpoints/ablation_{feature_set}"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 76)
    print("Mamba 消融训练（特征集隔离实验）")
    print("=" * 76)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"特征集: {feature_set} (input_dim={input_dim})")
    print(f"设备: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"数据集: {args.dataset_dir}")
    print(f"检查点: {checkpoint_dir}")
    print(f"超参数: epochs={args.epochs}, patience={args.patience}, "
          f"batch_size={args.batch_size}, lr={args.lr}, d_model={args.d_model}, n_layers={args.n_layers}")
    print("=" * 76)

    # ========== 1. 加载数据集 ==========
    print("\n[1/3] Loading dataset...")
    t0 = time.time()
    train_ds, val_ds = load_datasets(args.dataset_dir, feature_set=feature_set)
    print(f"      train={len(train_ds)} samples, val={len(val_ds)} samples "
          f"({time.time() - t0:.1f}s)")

    # ========== 2. 构建 DataLoader ==========
    print("\n[2/3] Building data loaders...")
    train_loader, val_loader = build_dataloaders(
        train_ds, val_ds, batch_size=args.batch_size
    )
    print(f"      train batches={len(train_loader)}, val batches={len(val_loader)}")

    # ========== 3. 创建训练器并训练 ==========
    print("\n[3/3] Training...\n")
    trainer = create_trainer(
        input_dim=input_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=checkpoint_dir,
    )
    print(f"模型参数量: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print("-" * 76)

    train_start = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        patience=args.patience,
        verbose=True,
    )
    train_time = time.time() - train_start

    # ========== 4. 训练总结 ==========
    print("\n" + "=" * 76)
    print("训练完成")
    print("=" * 76)
    print(f"总耗时: {train_time / 60:.1f} 分钟")
    print(f"完成轮数: {len(history['train_loss'])}")
    print(f"最佳验证损失: {trainer.best_val_loss:.4f}")
    print(f"最终训练损失: {history['train_loss'][-1]:.4f}")
    print(f"最佳模型: {checkpoint_dir}/best_model.pth")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 76)


if __name__ == "__main__":
    main()
