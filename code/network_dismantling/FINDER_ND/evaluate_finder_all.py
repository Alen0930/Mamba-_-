"""
Evaluate FINDER on BA/ER networks of various scales + statistical test.
Run in finder_tf environment.
"""
import sys
import time
import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import wilcoxon

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
    return {
        'rem_num': rem_num,
        'auc': auc,
        'q_c_opt': q_c_opt,
        'slcc_peak': slcc_peak,
    }


def run_single(dqn, G):
    t0 = time.time()
    seq, _ = dqn.EvaluateRealData(G, stepRatio=0.005)
    t1 = time.time()
    metrics = evaluate_sequence(G, seq)
    metrics['time_sec'] = t1 - t0
    return metrics


def part1_single_run(dqn):
    configs = [
        ('BA_50', lambda: nx.barabasi_albert_graph(50, 4, seed=42)),
        ('BA_100', lambda: nx.barabasi_albert_graph(100, 4, seed=42)),
        ('BA_500', lambda: nx.barabasi_albert_graph(500, 4, seed=42)),
        ('BA_1000', lambda: nx.barabasi_albert_graph(1000, 4, seed=42)),
        ('ER_50', lambda: nx.erdos_renyi_graph(50, 0.08, seed=42)),
        ('ER_500', lambda: nx.erdos_renyi_graph(500, 0.016, seed=42)),
        ('ER_1000', lambda: nx.erdos_renyi_graph(1000, 0.008, seed=42)),
    ]

    rows = []
    for net_name, gen_fn in configs:
        G = gen_fn()
        print(f"\n=== {net_name} (n={G.number_of_nodes()}, m={G.number_of_edges()}) ===")
        metrics = run_single(dqn, G)
        metrics.update({'network': net_name, 'method': 'FINDER', 'trial': 0})
        rows.append(metrics)
        print(f"  FINDER: rem={metrics['rem_num']:4d}, AUC={metrics['auc']:.4f}, "
              f"SLCC={metrics['slcc_peak']:3d}, time={metrics['time_sec']:.3f}s")

    df = pd.DataFrame(rows)
    df.to_csv('../../results/results_finder_all.csv', index=False)
    print(f"\nFINDER single-run saved to ../../results/results_finder_all.csv")
    return df


def part2_statistical_test(dqn, n_trials=10):
    configs = [
        ('BA_100', lambda s: nx.barabasi_albert_graph(100, 4, seed=s)),
        ('ER_100', lambda s: nx.erdos_renyi_graph(100, 0.08, seed=s)),
    ]

    rows = []
    for net_name, gen_fn in configs:
        print(f"\n=== FINDER statistical test: {net_name} ({n_trials} trials) ===")
        for trial in range(n_trials):
            G = gen_fn(2000 + trial)
            metrics = run_single(dqn, G)
            metrics.update({'network': net_name, 'method': 'FINDER', 'trial': trial})
            rows.append(metrics)

    df = pd.DataFrame(rows)
    df.to_csv('../../results/results_finder_stat.csv', index=False)
    print(f"\nFINDER statistical test saved to ../../results/results_finder_stat.csv")
    return df


def main():
    print("Creating FINDER model...")
    dqn = FINDER_Pure()
    dqn.LoadModel(MODEL_PATH)
    print("Model loaded.\n")

    print("PART 1: FINDER single-run on all networks")
    print("=" * 60)
    part1_single_run(dqn)

    print("\n\nPART 2: FINDER statistical test (10 trials)")
    print("=" * 60)
    part2_statistical_test(dqn, n_trials=10)


if __name__ == '__main__':
    main()
