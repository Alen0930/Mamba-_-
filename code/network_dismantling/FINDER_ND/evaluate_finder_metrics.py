"""
Evaluate FINDER with unified metrics on BA-50 and BA-100.
Run this in the finder_tf environment.
"""
import sys
import time
import numpy as np
import networkx as nx

sys.path.insert(0, '.')
from finder_pure import FINDER_Pure

MODEL_PATH = 'models/nrange_30_50_iter_78000.ckpt'


def evaluate_sequence(G, seq, stop_condition=0.1):
    n = G.number_of_nodes()
    stop = max(1, int(stop_condition * n))

    G_tmp = G.copy()
    trace_lcc = []
    trace_slcc = []
    rem_num = n
    removed = set()

    for i, node in enumerate(seq):
        if node in G_tmp:
            G_tmp.remove_node(node)
            removed.add(node)

        if G_tmp.number_of_nodes() == 0:
            trace_lcc.append(0.0)
            trace_slcc.append(0)
            if rem_num == n:
                rem_num = i + 1
            break

        components = sorted((len(c) for c in nx.connected_components(G_tmp)), reverse=True)
        lcc = components[0] if len(components) > 0 else 0
        slcc = components[1] if len(components) > 1 else 0

        trace_lcc.append(lcc / n)
        trace_slcc.append(slcc)

        if lcc <= stop and rem_num == n:
            rem_num = i + 1

    # Complete with remaining nodes
    remaining = [v for v in G.nodes() if v not in removed]
    if remaining:
        deg_map = dict(G.degree(remaining))
        remaining.sort(key=lambda v: -deg_map[v])
        for node in remaining:
            if node in G_tmp:
                G_tmp.remove_node(node)
            if G_tmp.number_of_nodes() == 0:
                trace_lcc.append(0.0)
                trace_slcc.append(0)
                break
            components = sorted((len(c) for c in nx.connected_components(G_tmp)), reverse=True)
            lcc = components[0] if len(components) > 0 else 0
            slcc = components[1] if len(components) > 1 else 0
            trace_lcc.append(lcc / n)
            trace_slcc.append(slcc)

    while len(trace_lcc) < n:
        trace_lcc.append(0.0)
        trace_slcc.append(0)

    q_vals = np.arange(1, len(trace_lcc) + 1) / n
    auc = np.trapz(trace_lcc, q_vals)

    q_c_opt = 1.0
    for i, lcc in enumerate(trace_lcc):
        if lcc == 0:
            q_c_opt = (i + 1) / n
            break

    slcc_peak = max(trace_slcc) if trace_slcc else 0
    q_at_slcc_peak = (trace_slcc.index(slcc_peak) + 1) / n if trace_slcc else 0

    return {
        'rem_num': rem_num,
        'auc': auc,
        'q_c_opt': q_c_opt,
        'slcc_peak': slcc_peak,
        'q_at_slcc_peak': q_at_slcc_peak,
        'trace_lcc': trace_lcc,
        'trace_slcc': trace_slcc,
    }


def main():
    networks = {
        'BA_50': nx.barabasi_albert_graph(50, 4, seed=42),
        'BA_100': nx.barabasi_albert_graph(100, 4, seed=42),
    }

    dqn = FINDER_Pure()
    dqn.LoadModel(MODEL_PATH)

    rows = []
    for net_name, G in networks.items():
        print(f"\n=== {net_name} (n={G.number_of_nodes()}, m={G.number_of_edges()}) ===")
        t0 = time.time()
        seq, _ = dqn.EvaluateRealData(G, stepRatio=0.01)
        t1 = time.time()

        metrics = evaluate_sequence(G, seq)
        metrics.update({
            'network': net_name,
            'method': 'FINDER',
            'time_sec': t1 - t0,
        })
        rows.append(metrics)
        print(f"  FINDER  : rem={metrics['rem_num']:3d}, AUC={metrics['auc']:.4f}, "
              f"q_c={metrics['q_c_opt']:.4f}, SLCC_peak={metrics['slcc_peak']:2d}, "
              f"time={metrics['time_sec']:.3f}s")

    import pandas as pd
    df = pd.DataFrame(rows)
    out_path = '../../results/results_finder_metrics.csv'
    df.to_csv(out_path, index=False)
    print(f"\nFINDER results saved to {out_path}")


if __name__ == '__main__':
    main()
