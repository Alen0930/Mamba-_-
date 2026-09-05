"""
Feature extraction and target label generation for GDM (networkx version).
Uses DSU (Disjoint Set Union) for fast connected-components computation.
"""
import logging
from itertools import combinations, chain
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def _chi(o, e):
    if e == 0:
        return 0.0
    return (o - e) ** 2 / e


class _DSU:
    __slots__ = ("parent", "size")
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.size[rx] < self.size[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]


def compute_features(
    G: nx.Graph,
    features: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    if features is None:
        features = ["degree", "clustering_coefficient", "kcore", "chi_degree"]

    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    out_features = {}

    if "degree" in features or "chi_degree" in features:
        deg = np.array([G.degree(node) for node in nodes], dtype=float)
        max_deg = deg.max()
        norm_deg = deg / max_deg if max_deg > 0 else deg
        if "degree" in features:
            out_features["degree"] = norm_deg
        if "chi_degree" in features:
            avg_deg = norm_deg.mean()
            chi_deg = np.array([_chi(norm_deg[i], avg_deg) for i in range(n)], dtype=float)
            out_features["chi_degree"] = chi_deg

    if "clustering_coefficient" in features or "chi_lcc" in features:
        cc = nx.clustering(G, nodes=nodes)
        cc_arr = np.array([cc[node] for node in nodes], dtype=float)
        avg_cc = cc_arr.mean()
        chi_lcc = np.array([_chi(cc_arr[i], avg_cc) for i in range(n)], dtype=float)
        if "clustering_coefficient" in features:
            out_features["clustering_coefficient"] = cc_arr
        if "chi_lcc" in features:
            out_features["chi_lcc"] = chi_lcc

    if "pagerank_out" in features:
        pr = nx.pagerank(G)
        out_features["pagerank_out"] = np.array([pr[node] for node in nodes], dtype=float)

    if "betweenness_centrality" in features:
        bc = nx.betweenness_centrality(G)
        out_features["betweenness_centrality"] = np.array([bc[node] for node in nodes], dtype=float)

    if "eigenvectors" in features:
        try:
            ec = nx.eigenvector_centrality(G, max_iter=1000)
            ec_arr = np.array([ec[node] for node in nodes], dtype=float)
        except Exception:
            ec_arr = np.zeros(n, dtype=float)
        out_features["eigenvectors"] = ec_arr

    if "kcore" in features:
        kcore = nx.core_number(G)
        kcore_arr = np.array([kcore[node] for node in nodes], dtype=float)
        max_kcore = kcore_arr.max()
        if max_kcore > 0:
            kcore_arr = kcore_arr / max_kcore
        out_features["kcore"] = kcore_arr

    return out_features


def _lcc_after_removal_dsu(n: int, edge_u: np.ndarray, edge_v: np.ndarray, removed: set) -> int:
    """Compute LCC size after removing a set of nodes using DSU."""
    dsu = _DSU(n)
    for i in range(edge_u.shape[0]):
        u = int(edge_u[i])
        v = int(edge_v[i])
        if u not in removed and v not in removed:
            dsu.union(u, v)
    max_sz = 0
    for i in range(n):
        if i not in removed:
            sz = dsu.size[dsu.find(i)]
            if sz > max_sz:
                max_sz = sz
    return max_sz


def compute_targets_bruteforce(
    G: nx.Graph,
    threshold: float = 0.18,
    max_k: int = 4,
    use_random_sampling: bool = False,
    sample_limit: int = 10000,
    logger: logging.Logger = logging.getLogger("dummy"),
) -> np.ndarray:
    """
    Compute target labels via brute-force dismantling.
    Uses DSU for fast connected-components.
    """
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    stop_condition = int(np.ceil(n * threshold))

    # Precompute edge arrays
    edge_u = np.array([node_to_idx[u] for u, v in G.edges()], dtype=np.int32)
    edge_v = np.array([node_to_idx[v] for u, v in G.edges()], dtype=np.int32)

    best_combinations = set()
    best_score = n

    for k in range(1, min(max_k + 1, n) + 1):
        all_combos = list(combinations(range(n), k))

        if use_random_sampling and len(all_combos) > sample_limit:
            rng = np.random.default_rng(42)
            sampled_indices = rng.choice(len(all_combos), size=sample_limit, replace=False)
            combo_iter = (all_combos[i] for i in sampled_indices)
        else:
            combo_iter = iter(all_combos)

        current_best_score = n
        current_best_combos = set()

        for combo in combo_iter:
            removed = set(combo)
            lcc_size = _lcc_after_removal_dsu(n, edge_u, edge_v, removed)

            if lcc_size < current_best_score:
                current_best_score = lcc_size
                current_best_combos = {combo}
            elif lcc_size == current_best_score:
                current_best_combos.add(combo)

        if current_best_score <= stop_condition:
            best_score = current_best_score
            best_combinations = current_best_combos
            break
        elif current_best_score < best_score:
            best_score = current_best_score
            best_combinations = current_best_combos

    targets = np.zeros(n, dtype=float)
    if len(best_combinations) == 0:
        return targets

    all_occurrences = list(chain.from_iterable(best_combinations))
    unique, counts = np.unique(all_occurrences, return_counts=True)
    num_combinations = len(best_combinations)

    for idx, count in zip(unique, counts):
        targets[idx] = count / num_combinations

    return targets


def extract_training_data(
    G: nx.Graph,
    features: Optional[List[str]] = None,
    threshold: float = 0.18,
    max_k: int = 4,
    compute_targets: bool = True,
    use_random_sampling: bool = False,
    sample_limit: int = 10000,
    logger: logging.Logger = logging.getLogger("dummy"),
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if features is None:
        features = ["degree", "clustering_coefficient", "kcore", "chi_degree"]

    out_features = compute_features(G, features=features)
    nodes = sorted(G.nodes())
    x = np.column_stack([out_features[f] for f in features])

    y = None
    if compute_targets:
        y = compute_targets_bruteforce(
            G, threshold=threshold, max_k=max_k,
            use_random_sampling=use_random_sampling,
            sample_limit=sample_limit, logger=logger,
        )

    return x, y
