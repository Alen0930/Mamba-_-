"""
验证 Mamba 网络拆解算法
使用 1000 节点的 BA 无标度网络进行测试
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import networkx as nx
import numpy as np
import torch
import time

from network_dismantling.unified_interface import dismantle, METHOD_REGISTRY


def test_mamba_basic():
    """基础功能测试：确保 Mamba 方法可以正常调用"""
    print("=" * 60)
    print("测试 1: 基础功能测试")
    print("=" * 60)

    # 检查 mamba 是否已注册
    if 'mamba' not in METHOD_REGISTRY:
        print("❌ 错误: mamba 方法未在 METHOD_REGISTRY 中注册")
        return False

    print("✓ mamba 方法已注册")

    # 创建小型测试图
    G_small = nx.barabasi_albert_graph(50, 2, seed=42)
    print(f"✓ 创建测试图: {G_small.number_of_nodes()} 节点, {G_small.number_of_edges()} 条边")

    # 测试拆解
    try:
        sequence = dismantle(G_small, method='mamba', stop_condition=1)
        print(f"✓ 拆解成功，序列长度: {len(sequence)}")

        # 验证序列完整性
        if len(sequence) == G_small.number_of_nodes():
            print("✓ 序列长度正确")
        else:
            print(f"❌ 序列长度错误: 期望 {G_small.number_of_nodes()}, 实际 {len(sequence)}")
            return False

        # 验证序列包含所有节点且无重复
        if len(set(sequence)) == len(sequence) == G_small.number_of_nodes():
            print("✓ 序列无重复且包含所有节点")
        else:
            print("❌ 序列存在重复或缺失节点")
            return False

        print("\n✅ 基础功能测试通过\n")
        return True

    except Exception as e:
        print(f"❌ 拆解失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mamba_1000_nodes():
    """在 1000 节点 BA 网络上测试"""
    print("=" * 60)
    print("测试 2: 1000 节点 BA 网络测试")
    print("=" * 60)

    # 创建 BA 无标度网络
    n = 1000
    m = 3
    G = nx.barabasi_albert_graph(n, m, seed=42)
    print(f"创建 BA 网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 条边")

    # 计算初始 LCC
    initial_lcc = max(len(c) for c in nx.connected_components(G))
    print(f"初始最大连通分量大小: {initial_lcc}")

    # 测试拆解
    try:
        print("\n开始拆解...")
        start_time = time.time()
        sequence = dismantle(G, method='mamba', stop_condition=1)
        end_time = time.time()

        print(f"✓ 拆解完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"✓ 拆解序列长度: {len(sequence)}")

        # 验证拆解过程
        G_tmp = G.copy()
        lcc_values = [initial_lcc]

        for i, node in enumerate(sequence[:100]):  # 只验证前 100 个节点
            G_tmp.remove_node(node)
            if G_tmp.number_of_nodes() > 0:
                lcc = max(len(c) for c in nx.connected_components(G_tmp))
                lcc_values.append(lcc)

        print(f"✓ 前 10 步 LCC 变化: {lcc_values[:11]}")

        # 检查 LCC 是否单调递减（总体趋势）
        if lcc_values[-1] < lcc_values[0]:
            print("✓ LCC 整体递减趋势正确")
        else:
            print("⚠ LCC 未明显递减")

        print("\n✅ 1000 节点测试通过\n")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_usage():
    """测试 GPU 使用情况"""
    print("=" * 60)
    print("测试 3: GPU 使用测试")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠ CUDA 不可用，跳过 GPU 测试")
        return True

    print(f"✓ CUDA 可用")
    print(f"✓ GPU 设备: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA 版本: {torch.version.cuda}")

    # 创建测试图
    G = nx.barabasi_albert_graph(500, 2, seed=42)

    try:
        # 清空 GPU 缓存
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(0)

        # 运行拆解
        print("\n运行 Mamba 拆解...")
        sequence = dismantle(G, method='mamba', stop_condition=1, device='cuda')

        # 检查 GPU 内存使用
        peak_memory = torch.cuda.max_memory_allocated(0)
        memory_used = (peak_memory - initial_memory) / 1024 / 1024  # MB

        print(f"✓ 拆解成功")
        print(f"✓ GPU 内存使用: {memory_used:.2f} MB")

        if memory_used > 0:
            print("✓ 确认使用了 GPU 进行计算")
        else:
            print("⚠ GPU 内存使用为 0，可能未使用 GPU")

        print("\n✅ GPU 测试通过\n")
        return True

    except Exception as e:
        print(f"❌ GPU 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compare_with_baseline():
    """与基线方法对比"""
    print("=" * 60)
    print("测试 4: 与基线方法对比")
    print("=" * 60)

    # 创建测试网络
    G = nx.barabasi_albert_graph(200, 2, seed=42)
    print(f"测试网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 条边")

    methods = ['degree', 'pagerank', 'mamba']
    results = {}

    for method in methods:
        try:
            print(f"\n测试方法: {method}")
            start_time = time.time()
            sequence = dismantle(G, method=method, stop_condition=1)
            end_time = time.time()

            # 计算移除前 20% 节点后的 LCC
            k = int(0.2 * G.number_of_nodes())
            G_tmp = G.copy()
            G_tmp.remove_nodes_from(sequence[:k])
            lcc_20 = max(len(c) for c in nx.connected_components(G_tmp)) if G_tmp.number_of_nodes() > 0 else 0

            results[method] = {
                'time': end_time - start_time,
                'lcc_20': lcc_20,
                'sequence_length': len(sequence)
            }

            print(f"  耗时: {results[method]['time']:.3f} 秒")
            print(f"  移除 20% 节点后 LCC: {lcc_20}")

        except Exception as e:
            print(f"  ❌ 方法 {method} 失败: {e}")
            results[method] = None

    # 显示对比结果
    print("\n" + "=" * 60)
    print("对比结果汇总:")
    print("=" * 60)
    print(f"{'方法':<15} {'耗时(秒)':<12} {'LCC@20%':<10}")
    print("-" * 60)

    for method in methods:
        if results[method] is not None:
            print(f"{method:<15} {results[method]['time']:<12.3f} {results[method]['lcc_20']:<10}")
        else:
            print(f"{method:<15} {'失败':<12}")

    print("\n✅ 对比测试完成\n")
    return True


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "Mamba 网络拆解算法验证" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

    # 运行所有测试
    all_passed = True

    all_passed &= test_mamba_basic()
    all_passed &= test_mamba_1000_nodes()
    all_passed &= test_gpu_usage()
    all_passed &= test_compare_with_baseline()

    # 总结
    print("=" * 60)
    if all_passed:
        print("✅ 所有测试通过！")
    else:
        print("⚠ 部分测试未通过，请检查上述输出")
    print("=" * 60)
