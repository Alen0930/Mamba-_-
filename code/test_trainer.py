"""
Mamba 训练引擎单元测试

测试 ListMLE 损失函数、数据集、训练器的核心功能
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import numpy as np
import networkx as nx
from network_dismantling.Mamba.trainer import (
    ListMLELoss,
    DismantlingDataset,
    MambaTrainer,
    create_trainer,
    collate_fn
)
from network_dismantling.unified_interface import dismantle


def test_listmle_loss():
    """测试 ListMLE 损失函数"""
    print("=" * 70)
    print("测试 1: ListMLE 损失函数")
    print("=" * 70)

    criterion = ListMLELoss()

    # 测试案例 1: 完美排序（损失应该很小）
    pred_scores = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    true_ranks = torch.tensor([[0, 1, 2, 3, 4]])  # 完美匹配
    loss1 = criterion(pred_scores, true_ranks)
    print(f"  完美排序损失: {loss1.item():.4f} (期望接近0)")

    # 测试案例 2: 完全逆序（损失应该很大）
    pred_scores = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    true_ranks = torch.tensor([[0, 1, 2, 3, 4]])  # 预测与真实相反
    loss2 = criterion(pred_scores, true_ranks)
    print(f"  逆序排序损失: {loss2.item():.4f} (期望较大)")

    # 测试案例 3: 批量输入
    pred_scores = torch.randn(4, 10)
    true_ranks = torch.randint(0, 10, (4, 10))
    loss3 = criterion(pred_scores, true_ranks)
    print(f"  批量输入损失: {loss3.item():.4f}")

    # 测试梯度
    pred_scores = torch.randn(2, 5, requires_grad=True)
    true_ranks = torch.randint(0, 5, (2, 5))
    loss = criterion(pred_scores, true_ranks)
    loss.backward()
    print(f"  梯度检查: {pred_scores.grad is not None} (期望 True)")
    print(f"  梯度范数: {pred_scores.grad.norm().item():.4f}")

    assert loss1 < loss2, "完美排序损失应小于逆序损失"
    assert pred_scores.grad is not None, "梯度应该被计算"

    print("✅ ListMLE 损失测试通过\n")


def test_dataset():
    """测试数据集类"""
    print("=" * 70)
    print("测试 2: DismantlingDataset")
    print("=" * 70)

    # 创建测试图
    graphs = [
        nx.barabasi_albert_graph(50, 2, seed=i)
        for i in range(5)
    ]

    # 定义拆解函数
    dismantler_fn = lambda G: dismantle(G, method='degree')

    # 创建数据集
    dataset = DismantlingDataset(graphs, dismantler_fn, cache_features=True)

    print(f"  数据集大小: {len(dataset)}")

    # 获取样本
    sample = dataset[0]
    print(f"  样本键: {sample.keys()}")
    print(f"  特征形状: {sample['features'].shape}")
    print(f"  节点ID形状: {sample['node_ids'].shape}")
    print(f"  排序标签形状: {sample['ranks'].shape}")

    # 验证排序标签
    ranks = sample['ranks'].numpy()
    print(f"  排序标签范围: [{ranks.min()}, {ranks.max()}]")
    print(f"  排序标签唯一值数量: {len(np.unique(ranks))}")

    # 测试 collate_fn
    batch = [dataset[i] for i in range(3)]
    batched = collate_fn(batch)
    print(f"\n  批次特征形状: {batched['features'].shape}")
    print(f"  批次标签形状: {batched['ranks'].shape}")
    print(f"  批次mask形状: {batched['mask'].shape}")

    assert len(dataset) == 5, "数据集大小不正确"
    assert sample['features'].dim() == 2, "特征应该是2维"
    assert sample['features'].shape[1] == 4, "特征维度应该是4"

    print("✅ 数据集测试通过\n")


def test_trainer():
    """测试训练器"""
    print("=" * 70)
    print("测试 3: MambaTrainer")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  使用设备: {device}")

    # 创建训练器
    trainer = create_trainer(
        input_dim=4,
        d_model=32,  # 使用较小的模型加速测试
        n_layers=1,
        device=device,
        checkpoint_dir='checkpoints/test'
    )

    print(f"  模型参数量: {sum(p.numel() for p in trainer.model.parameters()):,}")

    # 创建极小数据集
    graphs = [nx.barabasi_albert_graph(30, 2, seed=i) for i in range(3)]
    dismantler_fn = lambda G: dismantle(G, method='degree')
    dataset = DismantlingDataset(graphs, dismantler_fn, cache_features=True)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    # 训练一个 epoch
    print("\n  训练 1 个 epoch...")
    loss = trainer.train_epoch(train_loader)
    print(f"  训练损失: {loss:.4f}")

    # 测试保存和加载
    print("\n  测试模型保存...")
    trainer.save_checkpoint('test_model.pth')
    print("  模型已保存")

    print("  测试模型加载...")
    trainer.load_checkpoint('test_model.pth')
    print("  模型已加载")

    assert loss > 0, "损失应该大于0"
    assert trainer.current_epoch == 0, "当前epoch应该是0"

    print("✅ 训练器测试通过\n")


def test_training_convergence():
    """测试训练收敛性"""
    print("=" * 70)
    print("测试 4: 训练收敛性")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建过拟合测试：单个图训练多轮
    G = nx.barabasi_albert_graph(40, 2, seed=42)
    graphs = [G] * 10  # 重复同一个图

    dismantler_fn = lambda G: dismantle(G, method='degree')
    dataset = DismantlingDataset(graphs, dismantler_fn, cache_features=True)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    trainer = create_trainer(
        d_model=32,
        n_layers=1,
        device=device,
        learning_rate=1e-2,  # 较大学习率加速收敛
        checkpoint_dir='checkpoints/convergence_test'
    )

    print("  训练 10 个 epoch，验证损失下降...")
    losses = []
    for epoch in range(10):
        loss = trainer.train_epoch(train_loader)
        losses.append(loss)
        if epoch % 3 == 0:
            print(f"    Epoch {epoch+1}: Loss = {loss:.4f}")

    print(f"\n  初始损失: {losses[0]:.4f}")
    print(f"  最终损失: {losses[-1]:.4f}")
    print(f"  损失下降: {losses[0] - losses[-1]:.4f}")

    # 验证损失下降
    assert losses[-1] < losses[0], "训练应该使损失下降"

    print("✅ 收敛性测试通过\n")


def test_model_inference():
    """测试训练后的模型推理"""
    print("=" * 70)
    print("测试 5: 训练后模型推理")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 快速训练一个模型
    graphs = [nx.barabasi_albert_graph(40, 2, seed=i) for i in range(10)]
    dismantler_fn = lambda G: dismantle(G, method='degree')
    dataset = DismantlingDataset(graphs, dismantler_fn, cache_features=True)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    trainer = create_trainer(d_model=32, n_layers=1, device=device,
                            checkpoint_dir='checkpoints/inference_test')

    print("  快速训练 5 个 epoch...")
    for _ in range(5):
        trainer.train_epoch(train_loader)

    # 测试推理
    test_graph = nx.barabasi_albert_graph(50, 2, seed=999)
    print(f"\n  测试图: {test_graph.number_of_nodes()} 节点")

    from network_dismantling.Mamba.feature_encoding import extract_node_features
    features, node_ids = extract_node_features(test_graph)
    features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)

    trainer.model.eval()
    with torch.no_grad():
        scores = trainer.model(features_tensor)

    print(f"  输出分数形状: {scores.shape}")
    print(f"  分数范围: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
    print(f"  分数前5个: {scores[0, :5].cpu().numpy()}")

    assert scores.shape == (1, len(node_ids)), "输出形状不正确"
    assert not torch.isnan(scores).any(), "输出包含 NaN"

    print("✅ 推理测试通过\n")


def run_all_tests():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "Mamba 训练引擎单元测试" + " " * 20 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")

    tests = [
        test_listmle_loss,
        test_dataset,
        test_trainer,
        test_training_convergence,
        test_model_inference
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"❌ 测试失败: {test_fn.__name__}")
            print(f"   错误: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    print("=" * 70)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
