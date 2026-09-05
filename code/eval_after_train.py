"""
训练后效果验证脚本

加载训练完成的 Mamba 模型权重，在独立测试网络上运行拆解，
与基线方法（degree / pagerank / CoreHD）对比，
并输出训练前后 Mamba 的效果提升对比。

使用示例:
    # 基本用法
    python eval_after_train.py --checkpoint checkpoints/mamba_dismantling/best_model.pth

    # 自定义测试网络与输出
    python eval_after_train.py --checkpoint checkpoints/xxx/best_model.pth \
        --n 1500 --m 3 --seed 2026 --output eval_result.png

    # 在代码中调用
    from eval_after_train import run_evaluation, print_metrics_table
    sequences, metrics = run_evaluation(G, "checkpoints/xxx/best_model.pth")
    print_metrics_table(metrics)
"""
import argparse
import io
import sys
from typing import Dict, List, Optional, Tuple

import networkx as nx

from network_dismantling.unified_interface import dismantle
from network_dismantling.Mamba.model_io import (
    register_model_for_inference,
    unregister_model,
    is_model_registered,
    registered_source,
)
from evaluate import calc_metrics, plot_robustness

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
# 对比的基线方法: (显示名, unified_interface 注册名)
BASELINE_METHODS = [
    ("degree", "degree"),
    ("pagerank", "pagerank"),
    ("corehd", "CoreHD"),
]

# 训练前/后 Mamba 的显示名（用于指标表和对比曲线）
MAMBA_RANDOM_NAME = "mamba_random"
MAMBA_TRAINED_NAME = "mamba_trained"


# ---------------------------------------------------------------------------
# 测试网络生成
# ---------------------------------------------------------------------------
def build_test_graph(
    graph_type: str = "ba",
    n: int = 1000,
    m: int = 3,
    seed: int = 2026,
) -> nx.Graph:
    """
    生成独立测试网络（与训练集不同的随机种子，保证泛化性评估）

    Parameters
    ----------
    graph_type : str
        网络类型: 'ba' | 'er' | 'ws'
    n : int
        节点数
    m : int
        BA 连边参数（ER/WS 时为相应参数）
    seed : int
        随机种子
    """
    if graph_type == "ba":
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    elif graph_type == "er":
        G = nx.erdos_renyi_graph(n, 2 * m / n, seed=seed)
    elif graph_type == "ws":
        k = max(2, m * 2)
        if k % 2:
            k += 1
        G = nx.watts_strogatz_graph(n, k, 0.1, seed=seed)
    else:
        raise ValueError(f"不支持的图类型: {graph_type}")

    # 取最大连通分量，保证图连通
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    return G


# ---------------------------------------------------------------------------
# 评估主流程
# ---------------------------------------------------------------------------
def run_evaluation(
    G: nx.Graph,
    checkpoint_path: str,
    device: Optional[str] = None,
    stop_ratio: float = 0.1,
    fc_threshold: float = 0.01,
    baseline_methods: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[Dict[str, List[int]], Dict[str, Tuple]]:
    """
    运行完整评估：基线方法 + 训练前/后 Mamba

    Parameters
    ----------
    G : nx.Graph
        测试网络
    checkpoint_path : str
        训练完成的检查点路径
    device : str, optional
        计算设备
    stop_ratio / fc_threshold : float
        传给 calc_metrics 的评估参数
    baseline_methods : List[Tuple[str, str]], optional
        基线方法列表 [(显示名, 注册名)]，默认 degree/pagerank/CoreHD

    Returns
    -------
    sequences : Dict[str, List[int]]
        各方法的完整拆解序列
    metrics : Dict[str, Tuple]
        各方法的 (stop_step, fc_value, reach_stop, reach_fc)
    """
    baseline_methods = baseline_methods if baseline_methods is not None else BASELINE_METHODS

    sequences: Dict[str, List[int]] = {}
    metrics: Dict[str, Tuple] = {}

    # ---------- 1. 训练前 Mamba（随机初始化） ----------
    unregister_model()  # 确保无残留注册
    print(f"[1/4] Running mamba (random init)...")
    sequences[MAMBA_RANDOM_NAME] = dismantle(G, method="mamba")
    metrics[MAMBA_RANDOM_NAME] = calc_metrics(
        G, sequences[MAMBA_RANDOM_NAME], stop_ratio=stop_ratio, fc_threshold=fc_threshold
    )

    # ---------- 2. 训练后 Mamba（加载训练权重） ----------
    print(f"[2/4] Registering trained weights from {checkpoint_path} ...")
    register_model_for_inference(checkpoint_path, device=device)
    assert is_model_registered(), "训练权重注册失败"
    print(f"      registered source: {registered_source()}")
    sequences[MAMBA_TRAINED_NAME] = dismantle(G, method="mamba")
    metrics[MAMBA_TRAINED_NAME] = calc_metrics(
        G, sequences[MAMBA_TRAINED_NAME], stop_ratio=stop_ratio, fc_threshold=fc_threshold
    )

    # ---------- 3. 基线方法 ----------
    for display_name, method_name in baseline_methods:
        print(f"[3/4] Running baseline: {display_name} ...")
        sequences[display_name] = dismantle(G, method=method_name)
        metrics[display_name] = calc_metrics(
            G, sequences[display_name], stop_ratio=stop_ratio, fc_threshold=fc_threshold
        )

    # ---------- 4. 汇总 ----------
    print("[4/4] Evaluation done.")
    return sequences, metrics


# ---------------------------------------------------------------------------
# 结果输出
# ---------------------------------------------------------------------------
def print_metrics_table(metrics: Dict[str, Tuple]):
    """打印核心指标对比表"""
    print("\n" + "=" * 76)
    print("核心指标对比 (stop_ratio=0.1, fc_threshold=0.01)")
    print("=" * 76)
    print(f"{'方法':<20} {'Stop步数':<12} {'FC值':<12} {'达到Stop':<10} {'达到FC':<10}")
    print("-" * 76)

    for name, (stop_step, fc_value, reach_stop, reach_fc) in metrics.items():
        print(
            f"{name:<20} {stop_step:<12} {fc_value:<12.4f} "
            f"{str(reach_stop):<10} {str(reach_fc):<10}"
        )
    print("=" * 76)
    print("说明: Stop步数/FC值 越小越好（更快瓦解网络）")


def print_improvement(metrics: Dict[str, Tuple]):
    """打印训练前后 Mamba 的效果提升对比"""
    if MAMBA_RANDOM_NAME not in metrics or MAMBA_TRAINED_NAME not in metrics:
        return

    r_stop, r_fc, _, _ = metrics[MAMBA_RANDOM_NAME]
    t_stop, t_fc, _, _ = metrics[MAMBA_TRAINED_NAME]

    def _improve(before: float, after: float) -> str:
        """改善率: 正数表示更好（越小越好）"""
        if before == 0:
            return "-"
        pct = (before - after) / before * 100
        return f"{pct:+.1f}%"

    print("\n" + "=" * 76)
    print("Mamba 训练前后提升对比")
    print("=" * 76)
    print(f"{'指标':<14} {'训练前(随机)':<14} {'训练后':<14} {'改善':<10}")
    print("-" * 76)
    print(f"{'Stop步数':<14} {r_stop:<14} {t_stop:<14} {_improve(r_stop, t_stop):<10}")
    print(f"{'FC值':<14} {r_fc:<14.4f} {t_fc:<14.4f} {_improve(r_fc, t_fc):<10}")
    print("=" * 76)


def save_robustness_plot(
    G: nx.Graph,
    sequences: Dict[str, List[int]],
    save_path: str,
    sample_step: int = 10,
    dpi: int = 300,
):
    """生成并保存鲁棒性对比曲线图"""
    # 图例使用英文名（matplotlib 默认字体不含中文）
    plot_names = {
        MAMBA_RANDOM_NAME: "mamba (random)",
        MAMBA_TRAINED_NAME: "mamba (trained)",
    }
    plot_dict = {plot_names.get(k, k): v for k, v in sequences.items()}

    plot_robustness(
        G,
        plot_dict,
        sample_step=sample_step,
        save_path=save_path,
        dpi=dpi,
    )
    print(f"\nRobustness comparison plot saved: {save_path}")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def main():
    """命令行入口"""
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Mamba 训练后效果验证")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练完成的检查点路径")
    parser.add_argument("--graph-type", type=str, default="ba", choices=["ba", "er", "ws"], help="测试网络类型 (默认 ba)")
    parser.add_argument("--n", type=int, default=1000, help="测试网络节点数 (默认 1000)")
    parser.add_argument("--m", type=int, default=3, help="测试网络连边参数 (默认 3)")
    parser.add_argument("--seed", type=int, default=2026, help="测试网络随机种子 (默认 2026)")
    parser.add_argument("--device", type=str, default=None, help="计算设备 (默认自动)")
    parser.add_argument("--output", type=str, default="eval_after_train.png", help="对比曲线输出路径")
    parser.add_argument("--sample-step", type=int, default=10, help="曲线采样间隔 (默认 10)")
    parser.add_argument("--dpi", type=int, default=300, help="图片分辨率 (默认 300)")
    args = parser.parse_args()

    # 生成独立测试网络
    print("=" * 76)
    print("Mamba 训练后效果验证")
    print("=" * 76)
    print(f"检查点: {args.checkpoint}")
    print(f"生成测试网络: {args.graph_type}, n={args.n}, m={args.m}, seed={args.seed}")
    G = build_test_graph(args.graph_type, args.n, args.m, args.seed)
    print(f"测试网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 条边")
    print("=" * 76)

    # 运行评估
    sequences, metrics = run_evaluation(
        G,
        args.checkpoint,
        device=args.device,
    )

    # 输出结果
    print_metrics_table(metrics)
    print_improvement(metrics)
    save_robustness_plot(
        G, sequences, args.output, sample_step=args.sample_step, dpi=args.dpi
    )

    print("\n验证完成。")


if __name__ == "__main__":
    main()
