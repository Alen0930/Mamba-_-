import networkx as nx
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
