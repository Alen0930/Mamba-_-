"""
FINDER – networkx interface (no graph_tool dependency).

Reference:
> Fan, C., Zeng, L., Sun, Y., Liu, Y.-Y. Finding key players in complex networks through deep reinforcement learning. Nature Machine Intelligence 2, 317–324 (2020). DOI: 10.1038/s42256-020-0177-2
"""
"""
FINDER ND interface using networkx (no graph_tool dependency).
"""
import logging
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np

# Add FINDER_ND to path
local_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(local_dir))

from finder_pure import FINDER_Pure

model_file_path = local_dir / 'models'

if not model_file_path.exists():
    raise FileNotFoundError(f"Model file path {model_file_path} does not exist")
elif not model_file_path.is_dir():
    raise NotADirectoryError(f"Model file path {model_file_path} is not a directory")


def finder_nd_dismantle(G, stop_condition=0.1, reinsertion=True, strategy_id=0,
                        model_file_ckpt='nrange_30_50_iter_78000.ckpt',
                        step_ratio=0.01, reinsert_step=0.001,
                        logger=logging.getLogger("dummy"), **kwargs):
    """
    Pure-Python FINDER ND dismantler.
    Accepts networkx Graph, returns removal sequence.
    """
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    n = G.number_of_nodes()

    # Relabel to consecutive integers
    mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}
    G = nx.relabel_nodes(G, mapping)
    reverse_mapping = {v: k for k, v in mapping.items()}

    model_file = model_file_path / model_file_ckpt
    print('The best model is :%s' % (model_file))

    dqn = FINDER_Pure()
    dqn.LoadModel(str(model_file))

    print("Getting solution")
    solution, solution_time = dqn.EvaluateRealData(g=G, stepRatio=step_ratio)

    if reinsertion:
        print("Reinserting nodes")
        t1 = __import__('time').time()
        reinsert_solution, Robustness, MaxCCList = dqn.EvaluateSol(
            g=G, solution=solution, strategyID=strategy_id, reInsertStep=reinsert_step)
        t2 = __import__('time').time()
        solution_time = t2 - t1
        solution = reinsert_solution

    # Map back to original node IDs
    output = [reverse_mapping[node] for node in solution]

    # Truncate to stop_condition
    stop = max(1, int(stop_condition * n))
    G_tmp = G.copy()
    truncated = []
    for node in output:
        if node in G_tmp:
            G_tmp.remove_node(node)
        truncated.append(node)
        lcc = max((len(c) for c in nx.connected_components(G_tmp)), default=0) if G_tmp.number_of_nodes() > 0 else 0
        if lcc <= stop:
            break

    # Fill remaining nodes if needed
    remaining = [v for v in G.nodes() if v not in truncated]
    remaining_deg = [(v, G.degree(v)) for v in remaining]
    remaining_deg.sort(key=lambda x: -x[1])
    truncated.extend([v for v, _ in remaining_deg])

    # Map back to original IDs
    return [reverse_mapping[node] for node in truncated]
