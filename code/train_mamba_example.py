"""
Mamba 网络拆解模型训练示例

演示完整的训练流程：数据准备、模型训练、验证评估
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import logging
import networkx as nx
import torch
from torch.utils.data import DataLoader, random_split

from network_dismantling.Mamba.trainer import (
    DismantlingDataset,
    MambaTrainer,
    create_trainer,
    collate_fn
)
from network_dismantling.unified_interface import dismantle

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_training_graphs(num_graphs: int = 100, size_range=(50, 150)) -> list:
    """
    生成训练用的图数据集

    Parameters
    ----------
    num_graphs : int
        图的数量
    size_range : tuple
        节点数范围 (min_nodes, max_nodes)

    Returns
    -------
    graphs : list
        生成的图列表
    """
    import random
    graphs = []

    for i in range(num_graphs):
        # 随机选择图类型和参数
        graph_type = random.choice(['ba', 'er', 'ws'])
        n = random.randint(*size_range)

        try:
            if graph_type == 'ba':
                # Barabasi-Albert 无标度网络
                m = random.randint(2, min(5, n-1))
                G = nx.barabasi_albert_graph(n, m, seed=i)

            elif graph_type == 'er':
                # Erdos-Renyi 随机图
                p = random.uniform(0.05, 0.15)
                G = nx.erdos_renyi_graph(n, p, seed=i)

            else:  # ws
                # Watts-Strogatz 小世界网络
                k = random.randint(4, min(10, n-1))
                k = k if k % 2 == 0 else k + 1  # k 必须是偶数
                p = random.uniform(0.1, 0.3)
                G = nx.watts_strogatz_graph(n, k, p, seed=i)

            # 确保图是连通的
            if not nx.is_connected(G):
                # 取最大连通分量
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()

            if G.number_of_nodes() >= 20:  # 至少 20 个节点
                graphs.append(G)

        except Exception as e:
            logger.warning(f"生成图 {i} 失败: {e}")
            continue

    logger.info(f"成功生成 {len(graphs)} 个训练图")
    return graphs


def train_model_example():
    """完整训练示例"""
    print("=" * 70)
    print("Mamba 网络拆解模型训练示例")
    print("=" * 70)

    # ========== 1. 准备数据 ==========
    print("\n[1] 生成训练数据...")
    train_graphs = generate_training_graphs(num_graphs=80, size_range=(50, 100))
    val_graphs = generate_training_graphs(num_graphs=20, size_range=(50, 100))

    print(f"    训练集: {len(train_graphs)} 个图")
    print(f"    验证集: {len(val_graphs)} 个图")

    # 定义拆解函数（使用 CoreHD 作为监督信号）
    def dismantler_fn(G):
        return dismantle(G, method='degree', stop_condition=1)

    # 创建数据集
    print("\n[2] 创建数据集...")
    train_dataset = DismantlingDataset(
        graphs=train_graphs,
        dismantler_fn=dismantler_fn,
        cache_features=True
    )

    val_dataset = DismantlingDataset(
        graphs=val_graphs,
        dismantler_fn=dismantler_fn,
        cache_features=True
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Windows 上建议设为 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"    训练批次数: {len(train_loader)}")
    print(f"    验证批次数: {len(val_loader)}")

    # ========== 3. 创建训练器 ==========
    print("\n[3] 创建训练器...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"    使用设备: {device}")

    trainer = create_trainer(
        input_dim=4,
        d_model=64,
        n_layers=2,
        device=device,
        learning_rate=1e-3,
        weight_decay=1e-4,
        checkpoint_dir='checkpoints/mamba_dismantling'
    )

    print(f"    模型参数量: {sum(p.numel() for p in trainer.model.parameters()):,}")

    # ========== 4. 训练模型 ==========
    print("\n[4] 开始训练...")
    print("    提示: 这是一个演示，使用小数据集和少量 epoch")
    print("-" * 70)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        patience=5,
        verbose=True
    )

    # ========== 5. 训练结果 ==========
    print("\n" + "=" * 70)
    print("[5] 训练完成")
    print("=" * 70)

    print(f"\n训练历史:")
    print(f"  最终训练损失: {history['train_loss'][-1]:.4f}")
    print(f"  最终验证损失: {history['val_loss'][-1]:.4f}")
    print(f"  最佳验证损失: {trainer.best_val_loss:.4f}")

    # ========== 6. 测试推理 ==========
    print("\n[6] 测试训练后的模型...")
    test_graph = nx.barabasi_albert_graph(100, 3, seed=999)
    print(f"    测试图: {test_graph.number_of_nodes()} 节点, {test_graph.number_of_edges()} 条边")

    # 使用训练后的模型进行推理
    from network_dismantling.Mamba.feature_encoding import extract_node_features

    features, node_ids = extract_node_features(test_graph)
    features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)

    trainer.model.eval()
    with torch.no_grad():
        scores = trainer.model(features_tensor)
        scores = scores.squeeze(0).cpu().numpy()

    # 按分数排序生成拆解序列
    sorted_indices = scores.argsort()[::-1]
    predicted_sequence = [node_ids[i] for i in sorted_indices]

    print(f"    预测拆解序列前10个节点: {predicted_sequence[:10]}")

    # 与 degree 方法对比
    degree_sequence = dismantle(test_graph, method='degree')
    print(f"    Degree 拆解序列前10个节点: {degree_sequence[:10]}")

    # 计算 Spearman 相关性
    from scipy.stats import spearmanr
    correlation, _ = spearmanr(predicted_sequence, degree_sequence)
    print(f"    与 Degree 方法的 Spearman 相关性: {correlation:.4f}")

    # ========== 7. 保存最终模型 ==========
    print("\n[7] 保存最终模型...")
    trainer.save_checkpoint('final_model.pth')
    print(f"    模型已保存到: checkpoints/mamba_dismantling/final_model.pth")

    print("\n" + "=" * 70)
    print("训练示例完成！")
    print("=" * 70)


def quick_test():
    """快速测试训练流程（极小数据集）"""
    print("\n快速测试模式（5个图，5个epoch）\n")

    # 生成极小数据集
    train_graphs = [nx.barabasi_albert_graph(50, 2, seed=i) for i in range(5)]
    val_graphs = [nx.barabasi_albert_graph(50, 2, seed=i+100) for i in range(2)]

    dismantler_fn = lambda G: dismantle(G, method='degree')

    train_dataset = DismantlingDataset(train_graphs, dismantler_fn, cache_features=True)
    val_dataset = DismantlingDataset(val_graphs, dismantler_fn, cache_features=True)

    train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = create_trainer(device=device, checkpoint_dir='checkpoints/quick_test')

    print(f"设备: {device}")
    print(f"训练图: {len(train_graphs)}, 验证图: {len(val_graphs)}\n")

    history = trainer.train(train_loader, val_loader, num_epochs=5, patience=3, verbose=True)

    print(f"\n快速测试完成！")
    print(f"最终训练损失: {history['train_loss'][-1]:.4f}")
    print(f"最终验证损失: {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Mamba 网络拆解模型训练')
    parser.add_argument('--quick', action='store_true', help='快速测试模式')
    args = parser.parse_args()

    if args.quick:
        quick_test()
    else:
        train_model_example()
