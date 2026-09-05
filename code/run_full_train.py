"""
Mamba 模型完整训练脚本

使用 ba_corehd_full 数据集（BA 网络 500-1500 节点，CoreHD 标签）训练：
- 100 个 epoch，早停 patience=15
- 批次大小 8
- 检查点保存到 checkpoints/full_train

训练完成后可用 eval_after_train.py 验证效果:
    python eval_after_train.py --checkpoint checkpoints/full_train/best_model.pth

用法:
    python run_full_train.py                # 按默认参数训练
    python run_full_train.py --epochs 50    # 自定义 epoch 数
"""
import argparse
import io
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# 默认配置（与任务要求一致）
# ---------------------------------------------------------------------------
DATASET_DIR = "datasets/ba_corehd_full"
CHECKPOINT_DIR = "checkpoints/full_train"
NUM_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
D_MODEL = 64
N_LAYERS = 2
SEED = None  # 不固定种子，保留随机性


def main():
    # Windows 控制台编码安全 + 无缓冲输出（后台运行时日志立即可见）
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True
        )

    parser = argparse.ArgumentParser(description="Mamba 模型完整训练")
    parser.add_argument("--dataset-dir", type=str, default=DATASET_DIR, help="数据集目录")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR, help="检查点目录")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="最大训练轮数")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="早停耐心值")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="批次大小")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="权重衰减")
    parser.add_argument("--d-model", type=int, default=D_MODEL, help="隐藏维度")
    parser.add_argument("--n-layers", type=int, default=N_LAYERS, help="Mamba 层数")
    parser.add_argument("--device", type=str, default=None, help="设备 (默认自动)")
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复训练（可选）")
    args = parser.parse_args()

    import torch
    from network_dismantling.Mamba.dataset_generator import load_datasets, build_dataloaders
    from network_dismantling.Mamba.trainer import create_trainer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ========== 1. 加载数据集 ==========
    print("=" * 76)
    print("Mamba 完整训练")
    print("=" * 76)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"数据集: {args.dataset_dir}")
    print(f"超参数: epochs={args.epochs}, patience={args.patience}, "
          f"batch_size={args.batch_size}, lr={args.lr}, "
          f"d_model={args.d_model}, n_layers={args.n_layers}")
    print("=" * 76)

    print("\n[1/3] Loading dataset...")
    t0 = time.time()
    train_ds, val_ds = load_datasets(args.dataset_dir)
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
        input_dim=4,
        d_model=args.d_model,
        n_layers=args.n_layers,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(f"模型参数量: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print("-" * 76)

    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"从检查点恢复: {args.resume} (epoch {trainer.current_epoch + 1})")

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
    print(f"总耗时: {train_time / 3600:.2f} 小时 ({train_time / 60:.1f} 分钟)")
    print(f"完成轮数: {len(history['train_loss'])}")
    print(f"最佳验证损失: {trainer.best_val_loss:.4f}")
    print(f"最终训练损失: {history['train_loss'][-1]:.4f}")
    print(f"最终验证损失: {history['val_loss'][-1]:.4f}")
    print(f"最佳模型: {args.checkpoint_dir}/best_model.pth")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 76)
    print("\n验证训练效果:")
    print(f"  python eval_after_train.py --checkpoint {args.checkpoint_dir}/best_model.pth")


if __name__ == "__main__":
    main()
