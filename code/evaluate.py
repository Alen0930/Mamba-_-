from typing import Optional

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def calc_metrics(
    graph,
    dismantle_seq,
    stop_ratio=0.1,
    fc_threshold=0.01,
    strict=False
):
    """
    计算网络拆解算法的核心量化指标
    ----------
    参数：
        graph : nx.Graph
            原始待拆解网络图
        dismantle_seq : list
            拆解节点序列（按移除优先级从高到低排列）
        stop_ratio : float, 可选
            停止条件：最大连通分量占原始网络的比例，取值(0, 1]，默认0.1
        fc_threshold : float, 可选
            临界阈值判定标准：最大连通分量占比低于该值视为网络瓦解，默认0.01
        strict : bool, 可选
            严格模式：True时序列中出现无效节点直接报错；False时自动跳过，默认False
    ----------
    返回：
        stop_step : int
            达到停止条件所需的有效移除节点数
        fc_value : float
            临界阈值：网络瓦解时移除节点占总节点的比例
        reach_stop : bool
            是否成功达到停止条件
        reach_fc : bool
            是否成功达到临界瓦解阈值
    """
    # ========== 输入校验 ==========
    if graph.number_of_nodes() == 0:
        raise ValueError("输入图不能为空")
    if not 0 < stop_ratio <= 1:
        raise ValueError(f"stop_ratio 必须在(0, 1]范围内，当前值：{stop_ratio}")
    if not 0 < fc_threshold <= 1:
        raise ValueError(f"fc_threshold 必须在(0, 1]范围内，当前值：{fc_threshold}")
    if len(dismantle_seq) == 0:
        raise ValueError("拆解序列不能为空")

    total_nodes = graph.number_of_nodes()
    current_graph = graph.copy()

    # 状态初始化
    stop_step = None
    fc_step = None
    reach_stop = False
    reach_fc = False

    # ========== 迭代拆解过程 ==========
    for step_idx, node in enumerate(dismantle_seq, start=1):
        # 处理无效/重复节点
        if node not in current_graph:
            if strict:
                raise ValueError(f"节点 {node} 不存在于当前图中（可能重复或无效）")
            continue

        current_graph.remove_node(node)

        # 图已完全拆空，直接终止
        if current_graph.number_of_nodes() == 0:
            if not reach_stop:
                stop_step = step_idx
                reach_stop = True
            if not reach_fc:
                fc_step = step_idx
                reach_fc = True
            break

        # 计算当前最大连通分量占比
        largest_cc = max(nx.connected_components(current_graph), key=len)
        lcc_ratio = len(largest_cc) / total_nodes

        # 判定停止条件
        if not reach_stop and lcc_ratio <= stop_ratio:
            stop_step = step_idx
            reach_stop = True

        # 判定临界瓦解阈值
        if not reach_fc and lcc_ratio <= fc_threshold:
            fc_step = step_idx
            reach_fc = True

        # 两个条件都已达到，可提前终止
        if reach_stop and reach_fc:
            break

    # ========== 兜底处理：序列遍历完仍未达标 ==========
    if not reach_stop:
        stop_step = step_idx
    if not reach_fc:
        fc_step = step_idx

    fc_value = fc_step / total_nodes
    return stop_step, fc_value, reach_stop, reach_fc


# ===========================================================================
# AUC（主指标）与统一 trace
# ===========================================================================
# 口径（全仓库唯一）：
#   trace[i] = 第 i+1 次「有效移除」后最大连通分量占原始节点数的比例 LCC/n
#   q_vals   = arange(1, len(trace)+1) / n
#   auc      = np.trapezoid(trace, q_vals)      # 即 ∫₀¹ S(q)dq
#
# 这就是 Schneider 等人提出的 robustness R，值域 [0, 1]，越小越好
# （0 = 一步瓦解，1 = 完全拆不动）。
#
# 注意两点：
#   1. 必须走**完整序列**再积分。calc_metrics 为了省时间在 LCC<=1% 处就 break，
#      只覆盖曲线前 ~35%，不同方法的截断点还不同，直接拿来积 AUC 会有系统性偏差。
#   2. numpy >= 2.0 已移除 np.trapz，只能用 np.trapezoid。
#
# 与 stop_step / fc_value 的关系：三者由**同一条 trace** 导出，保证自洽，
# 且一次遍历即可同时算出，比分别计算快约 3 倍。
def lcc_trace(graph: nx.Graph, dismantle_seq) -> np.ndarray:
    """
    一次性走完整条拆解序列，记录每一步的 LCC 占比。

    Parameters
    ----------
    graph : nx.Graph
        原始待拆解网络
    dismantle_seq : list
        拆解节点序列（按移除优先级从高到低）

    Returns
    -------
    trace : np.ndarray, dtype float64
        trace[i] = 第 i+1 次**有效移除**后的 LCC/n。
        长度 = 序列中的有效移除数；**仅当图被真正拆空时**才补 0 到 n。

    关于补齐的语义（重要）：只有「图确实被拆空」才补 0——那是真实的 S(q)=0。
    若序列被**提前截断**（长度 < n 且图未拆空），不补 0，trace 长度就等于
    有效移除数。否则补 0 会凭空造出一次「阈值穿越」，让 stop_step 与
    calc_metrics 不一致，也会让 AUC 被虚假地压低（截断的方法白赚便宜）。
    因此做 AUC 对比时，务必先把各方法的序列补齐到 n（见 complete_sequence）。
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("输入图不能为空")
    if len(dismantle_seq) == 0:
        raise ValueError("拆解序列不能为空")

    total_nodes = graph.number_of_nodes()
    current_graph = graph.copy()
    trace = []
    emptied = False

    for node in dismantle_seq:
        if node not in current_graph:
            continue  # 无效/重复节点：不计入有效移除数
        current_graph.remove_node(node)

        if current_graph.number_of_nodes() == 0:
            trace.append(0.0)
            emptied = True
            break

        largest_cc = max(nx.connected_components(current_graph), key=len)
        trace.append(len(largest_cc) / total_nodes)

    trace = np.asarray(trace, dtype=np.float64)
    if emptied and trace.shape[0] < total_nodes:
        trace = np.concatenate([trace, np.zeros(total_nodes - trace.shape[0])])
    return trace


def complete_sequence(graph: nx.Graph, dismantle_seq) -> list:
    """
    把（可能被提前截断的）拆解序列补齐到长度 n。

    复用 unified_interface._fill_remaining 的语义：按**原图**度数降序补齐。
    这样所有方法的序列长度一致、尾部形状一致，AUC 与 stop_step 才可比——
    dismantle() 返回的序列本来就已经这样补齐过，本函数是给不走 dismantle()
    的路径（如 RL 的 greedy_sequence）用的。
    """
    total_nodes = graph.number_of_nodes()
    seq = list(dismantle_seq)
    if len(seq) >= total_nodes:
        return seq[:total_nodes]
    removed = set(seq)
    remaining = [v for v in graph.nodes() if v not in removed]
    remaining.sort(key=lambda v: graph.degree(v), reverse=True)
    return seq + remaining


def calc_auc(trace: np.ndarray, n: Optional[int] = None) -> float:
    """
    trace -> AUC（曲线下面积）。

    Parameters
    ----------
    trace : np.ndarray
        lcc_trace(...) 的输出
    n : int, optional
        原始图节点数，用作归一化分母。默认取 len(trace)（即假定序列已完整）。

    Returns
    -------
    auc : float
        ∫₀¹ S(q)dq，越小越好。**只有传入完整（长度 n）的 trace 才得到完整 AUC**；
        对截断的 trace 得到的是「已观测视界内」的部分面积，方法间不可比。
    """
    trace = np.asarray(trace, dtype=np.float64)
    if trace.shape[0] == 0:
        raise ValueError("trace 不能为空")
    n = n or trace.shape[0]
    q_vals = np.arange(1, trace.shape[0] + 1) / n
    return float(np.trapezoid(trace, q_vals))


def calc_metrics_from_trace(
    trace: np.ndarray,
    n: Optional[int] = None,
    stop_ratio: float = 0.1,
    fc_threshold: float = 0.01,
):
    """
    从 trace 反推 stop_step 与 fc_value，与 calc_metrics 的语义保持一致。

    之所以能共享，是因为 trace[i] 与 calc_metrics 内部的 lcc_ratio 一一对应
    （两者都按「有效移除」计数）。

    Parameters
    ----------
    trace : np.ndarray
        lcc_trace(...) 的输出
    n : int, optional
        原始图节点数（fc_value 的分母）。默认取 len(trace)。

    Returns
    -------
    (stop_step, fc_value) : Tuple[int, float]
    """
    trace = np.asarray(trace, dtype=np.float64)
    if trace.shape[0] == 0:
        raise ValueError("trace 不能为空")
    n = n or trace.shape[0]

    stop_hit = np.flatnonzero(trace <= stop_ratio)
    fc_hit = np.flatnonzero(trace <= fc_threshold)

    # 未触发时与 calc_metrics 的兜底一致：取已走完的有效移除数
    stop_step = int(stop_hit[0]) + 1 if stop_hit.size else trace.shape[0]
    fc_step = int(fc_hit[0]) + 1 if fc_hit.size else trace.shape[0]
    return stop_step, fc_step / n


def calc_all_metrics(graph: nx.Graph, dismantle_seq) -> dict:
    """
    一次遍历同时得到三个指标（AUC 主指标 + stop_step + fc_value）。

    避免分别调用 calc_metrics 与 calc_auc 造成重复遍历（AUC 需要走满 n 步，
    而 calc_metrics 只走 ~0.35n，各算一次总开销反而更大）。

    返回 {'auc', 'stop_step', 'fc_value'}。AUC 的完整性取决于序列是否完整——
    传入前请确保已用 complete_sequence 补齐（dismantle() 的返回值天然满足）。
    """
    n = graph.number_of_nodes()
    trace = lcc_trace(graph, dismantle_seq)
    stop_step, fc_value = calc_metrics_from_trace(trace, n=n)
    return {
        "auc": calc_auc(trace, n=n),
        "stop_step": stop_step,
        "fc_value": fc_value,
    }


def plot_robustness(
    graph,
    methods_seq_dict,
    sample_step=10,
    save_path=None,
    dpi=300,
    figsize=(8, 5)
):
    """
    绘制多算法拆解鲁棒性对比曲线
    ----------
    参数：
        graph : nx.Graph
            原始待拆解网络图
        methods_seq_dict : dict
            算法字典，key为算法名称，value为对应的拆解节点序列
        sample_step : int, 可选
            采样间隔（每移除多少个节点记录一次），默认10
        save_path : str, 可选
            图片保存路径，不填则直接弹出显示
        dpi : int, 可选
            保存图片的分辨率，默认300
        figsize : tuple, 可选
            画布尺寸，默认(8, 5)
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("输入图不能为空")
    if not methods_seq_dict:
        raise ValueError("算法序列字典不能为空")
    if sample_step < 1:
        raise ValueError("采样间隔必须大于等于1")

    total_nodes = graph.number_of_nodes()
    plt.figure(figsize=figsize)

    for method_name, seq in methods_seq_dict.items():
        if len(seq) == 0:
            continue

        current_graph = graph.copy()
        x_axis = [0.0]
        y_axis = [1.0]

        for idx, node in enumerate(seq, start=1):
            if node not in current_graph:
                continue
            current_graph.remove_node(node)

            # 按间隔采样
            if idx % sample_step == 0:
                if current_graph.number_of_nodes() == 0:
                    x_axis.append(idx / total_nodes)
                    y_axis.append(0.0)
                    break

                largest_cc = max(nx.connected_components(current_graph), key=len)
                lcc_ratio = len(largest_cc) / total_nodes
                x_axis.append(idx / total_nodes)
                y_axis.append(lcc_ratio)

        plt.plot(x_axis, y_axis, label=method_name, linewidth=1.5)

    # ========== 图表格式 ==========
    plt.xlabel('Removed nodes fraction', fontsize=11)
    plt.ylabel('Largest component fraction', fontsize=11)
    plt.title('Dismantling Robustness Comparison', fontsize=12, pad=10)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
