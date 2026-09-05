"""
Repeated FINDER experiment: 3 network types × 9 scales × 20 trials.
Incremental save to prevent data loss.
Output: ../../results/results_repeated_finder.csv
"""
import sys
import os
import time
import numpy as np
import networkx as nx
import pandas as pd

sys.path.insert(0, '.')
from finder_pure import FINDER_Pure

MODEL_PATH = 'models/nrange_30_50_iter_78000.ckpt'
NETWORKS = ['BA', 'ER', 'WS']
SIZES = [50, 100, 200, 300, 400, 500, 1000]
N_TRIALS = 20
STOP = 0.1
OUT_FILE = '../../results/results_repeated_finder.csv'
CHECKPOINT_EVERY = 50


def generate_network(net_type, n, seed):
    if net_type == 'BA':
        return nx.barabasi_albert_graph(n, 4, seed=seed)
    elif net_type == 'ER':
        p = 8.0 / (n - 1)
        return nx.erdos_renyi_graph(n, p, seed=seed)
    elif net_type == 'WS':
        return nx.watts_strogatz_graph(n, 8, 0.3, seed=seed)
    else:
        raise ValueError(f"Unknown network type: {net_type}")


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
    slcc_peak = max(trace_slcc) if trace_slcc else 0
    return auc, rem_num, slcc_peak


def main():
    total_tasks = len(NETWORKS) * len(SIZES) * N_TRIALS
    completed = 0
    rows = []

    if os.path.exists(OUT_FILE):
        df_existing = pd.read_csv(OUT_FILE)
        rows = df_existing.to_dict('records')
        completed = len(rows)
        print(f"Resumed from {OUT_FILE}: {completed}/{total_tasks} already done.")

    done_keys = set()
    for r in rows:
        key = (r['network_type'], r['n'], r['trial'])
        done_keys.add(key)

    print(f"Running FINDER {total_tasks} experiments.")
    print(f"Networks: {NETWORKS}, Sizes: {SIZES}, Trials: {N_TRIALS}")
    print("=" * 70)

    print("Loading FINDER model...")
    dqn = FINDER_Pure()
    dqn.LoadModel(MODEL_PATH)
    print("Model loaded.\n")

    t_start = time.time()
    for net_type in NETWORKS:
        for n in SIZES:
            for trial in range(N_TRIALS):
                key = (net_type, n, trial)
                if key in done_keys:
                    continue

                G = generate_network(net_type, n, 20000 + trial)
                t0 = time.time()
                seq, _ = dqn.EvaluateRealData(G, stepRatio=0.005)
                t1 = time.time()

                auc, rem_num, slcc_peak = evaluate_sequence(G, seq)
                rows.append({
                    'network_type': net_type,
                    'n': n,
                    'trial': trial,
                    'method': 'FINDER',
                    'auc': auc,
                    'rem_num': rem_num,
                    'slcc_peak': slcc_peak,
                    'time_sec': t1 - t0,
                })
                completed += 1

                if completed % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
                    elapsed = time.time() - t_start
                    eta = elapsed / completed * (total_tasks - completed)
                    print(f"  Progress: {completed}/{total_tasks} ({100*completed/total_tasks:.1f}%) | "
                          f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")

    pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
    print(f"\nAll done! Results saved to {OUT_FILE}")
    print(f"Total time: {(time.time() - t_start)/60:.1f} minutes")


if __name__ == '__main__':
    main()
