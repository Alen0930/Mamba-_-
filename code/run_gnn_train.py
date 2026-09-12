"""
GNN + 双向 Mamba 网络拆解模型训练脚本（阶段 B）

架构：GCN 消息传递编码（用邻接矩阵）+ 双向 Mamba 序列编码，
输入特征只用节点度（degree），标签复用 CoreHD 生成。

用法:
    python run_gnn_train.py                              # 默认超参
    python run_gnn_train.py --epochs 60 --batch-size 4
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

    parser = argparse.ArgumentParser(description="GNN + 双向 Mamba 网络拆解训练")
    parser.add_argument("--dataset-dir", type=str, default="datasets/ba_gnn",
                        help="含邻接矩阵的数据集目录")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/gnn_mamba",
                        help="检查点目录")
    parser.add_argument("--epochs", type=int, default=100, help="最大训练轮数")
    parser.add_argument("--patience", type=int, default=15, help="早停耐心值")
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--hidden-dim", type=int, default=64, help="GCN 隐藏维度")
    parser.add_argument("--d-model", type=int, default=64, help="Mamba 隐藏维度")
    parser.add_argument("--n-gnn-layers", type=int, default=2, help="GCN 层数")
    parser.add_argument("--n-mamba-layers", type=int, default=2,
                        help="序列模型层数；0 = 纯 GCN（无序列模型，消融用）")
    parser.add_argument("--seq-model", type=str, default="mamba",
                        choices=["mamba", "attention"],
                        help="序列模型类型：mamba（双向 SSM，O(N)，本方法）/"
                             "attention（Transformer 自注意力，O(N²)，扩展性对照）")
    parser.add_argument("--device", type=str, default=None, help="设备 (默认自动)")
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复训练（传入检查点路径）")
    parser.add_argument("--select-by", type=str, default="loss", choices=["loss", "auc"],
                        help="best_model 选择准则：loss=ListMLE 验证损失（默认，向后兼容）"
                             " / auc=验证图静态拆解 AUC（主指标）")
    parser.add_argument("--auc-subset", type=int, default=5,
                        help="select-by=auc 时每 epoch 评测的验证图数量（控制开销）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（固定初始化与数据顺序，保证训练可复现）")
    parser.add_argument("--order", type=str, default="degree",
                        choices=["degree", "bfs", "dfs", "core"],
                        help="节点序列排序方式（只影响 Mamba 分支，GCN 对置换等变）。"
                             "数据集必须是用同一 order 构建的")
    args = parser.parse_args()

    import torch
    from network_dismantling.Mamba.dataset_generator import load_datasets, build_dataloaders
    from network_dismantling.Mamba.trainer import MambaTrainer
    from network_dismantling.Mamba.gnn_mamba_model import GNNMambaModel

    # 固定随机性：此前 run_gnn_train 未设种子，模型初始化与数据顺序
    # 跨运行不可复现（可重现性清单要求的「随机性控制」）。
    import numpy as _np
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    _np.random.seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 76)
    print("GNN + 双向 Mamba 网络拆解训练")
    print("=" * 76)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"数据集: {args.dataset_dir}")
    print(f"检查点: {args.checkpoint_dir}")
    print(f"超参数: epochs={args.epochs}, patience={args.patience}, batch_size={args.batch_size}, "
          f"lr={args.lr}, hidden_dim={args.hidden_dim}, d_model={args.d_model}, "
          f"n_gnn_layers={args.n_gnn_layers}, n_mamba_layers={args.n_mamba_layers}")
    print("=" * 76)

    # ========== 1. 加载数据集 ==========
    print("\n[1/3] Loading dataset...")
    t0 = time.time()
    train_ds, val_ds = load_datasets(args.dataset_dir, feature_set="degree")
    print(f"      train={len(train_ds)} samples, val={len(val_ds)} samples "
          f"({time.time() - t0:.1f}s)")

    # ========== 2. 构建 DataLoader ==========
    print("\n[2/3] Building data loaders...")
    train_loader, val_loader = build_dataloaders(
        train_ds, val_ds, batch_size=args.batch_size
    )
    print(f"      train batches={len(train_loader)}, val batches={len(val_loader)}")

    # ========== 3. 创建模型与训练器 ==========
    print("\n[3/3] Training...\n")
    model = GNNMambaModel(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        d_model=args.d_model,
        n_gnn_layers=args.n_gnn_layers,
        n_mamba_layers=args.n_mamba_layers,
        seq_model=args.seq_model,
    )
    trainer = MambaTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        select_by=args.select_by,
        auc_graphs=val_ds.graphs if args.select_by == "auc" else None,
        auc_subset=args.auc_subset,
        order=args.order,
    )
    print(f"模型参数量: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"节点序列排序: {args.order}")
    print(f"选模型准则: {args.select_by}"
          + (f" (每轮评测 {args.auc_subset} 张验证图)" if args.select_by == "auc" else ""))
    print("-" * 76)

    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"从检查点恢复: {args.resume} (epoch {trainer.current_epoch + 1}, "
              f"best_val_loss={trainer.best_val_loss:.4f})")

    train_start = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        patience=args.patience,
        verbose=True,
        start_epoch=(trainer.current_epoch + 1) if args.resume else None,
    )
    train_time = time.time() - train_start

    # ========== 4. 训练总结 ==========
    print("\n" + "=" * 76)
    print("训练完成")
    print("=" * 76)
    print(f"总耗时: {train_time / 3600:.2f} 小时 ({train_time / 60:.1f} 分钟)")
    print(f"完成轮数: {history['n_epochs_run']}")
    if args.select_by == "auc":
        # 按 AUC 选模型时 best_val_loss 从不更新，仍是 inf，不要打印它误导人
        print(f"最佳验证 AUC: {trainer.best_val_auc:.4f}（best_model 按此选择）")
        print(f"最终验证损失: {history['final_val_loss']:.4f}（仅记录，未用于选模型）")
    else:
        print(f"最佳验证损失: {trainer.best_val_loss:.4f}")
    print(f"最终训练损失: {history['final_train_loss']:.4f}")

    # 训练结束后做一次全量 AUC 复核（训练中每轮只用子集，开销所限）
    if args.select_by == "auc":
        from network_dismantling.Mamba.auc_eval import mean_auc
        full_auc = mean_auc(trainer.model, val_ds.graphs, device, max_graphs=None,
                            order=args.order)
        print(f"全量验证 AUC（{len(val_ds.graphs)} 张图复核）: {full_auc:.4f}")

    print(f"最佳模型: {args.checkpoint_dir}/best_model.pth")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 76)


if __name__ == "__main__":
    main()
