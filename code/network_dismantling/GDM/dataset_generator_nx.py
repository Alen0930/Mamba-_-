"""
Extended synthetic network generator for GDM training (networkx version).
Supports both serial and parallel generation.
"""
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

import networkx as nx
import numpy as np

from network_dismantling.GDM.training_data_extractor_nx import extract_training_data


def _generate_one(args):
    """Worker: generate network + compute labels, save to npz."""
    net_type, idx, n_range, seed, out_dir, features, threshold, max_k = args
    rng = np.random.default_rng(seed + idx)
    
    if net_type == "ba":
        n = rng.integers(n_range[0], n_range[1] + 1)
        m = rng.integers(2, min(5, n - 1) + 1)
        G = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(0, 2**31)))
    elif net_type == "er":
        n = rng.integers(n_range[0], n_range[1] + 1)
        p = rng.uniform(0.06, 0.25)
        G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(0, 2**31)))
        if G.number_of_nodes() > 0 and nx.number_of_isolates(G) > 0:
            p = min(p * 2, 0.5)
            G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(0, 2**31)))
    elif net_type == "ws":
        n = rng.integers(n_range[0], n_range[1] + 1)
        k = rng.integers(4, min(10, n - 1) + 1)
        if k % 2 == 1:
            k += 1
        p = rng.uniform(0.1, 0.5)
        G = nx.watts_strogatz_graph(n, k, p, seed=int(rng.integers(0, 2**31)))
    else:
        raise ValueError(f"Unknown net_type: {net_type}")
    
    n_actual = G.number_of_nodes()
    use_sampling = n_actual > 35
    sample_limit = 8000 if n_actual > 50 else 12000
    
    x, y = extract_training_data(
        G, features=features, threshold=threshold, max_k=max_k,
        compute_targets=True, use_random_sampling=use_sampling,
        sample_limit=sample_limit,
    )
    
    edges = np.array(list(G.edges()), dtype=np.int64)
    if edges.shape[0] > 0:
        edges_undirected = np.vstack([edges, edges[:, [1, 0]]])
    else:
        edges_undirected = edges
    
    fname = Path(out_dir) / f"{net_type}_{idx:04d}.npz"
    np.savez_compressed(
        fname,
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        edges=edges_undirected.astype(np.int64),
        n=n_actual,
    )
    return f"{net_type}_{idx:04d}.npz", n_actual


def generate_training_dataset(
    out_dir: str = "network_dismantling/GDM/dataset/synth_train_NEW",
    per_type: int = 100,
    n_range: Tuple[int, int] = (20, 60),
    seed: int = 42,
    features: List[str] = None,
    threshold: float = 0.18,
    max_k: int = 4,
    n_jobs: int = None,
):
    if features is None:
        features = ["degree", "clustering_coefficient", "kcore", "chi_degree"]
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    for net_type in ["ba", "er", "ws"]:
        for i in range(per_type):
            tasks.append((net_type, i, n_range, seed, out_dir, features, threshold, max_k))
    
    print(f"Generating {len(tasks)} networks using {n_jobs} processes...")
    
    with Pool(processes=n_jobs) as pool:
        results = pool.map(_generate_one, tasks)
    
    for fname, n in results:
        print(f"Saved {fname} (n={n})")
    
    print(f"Total: {len(results)} networks")


if __name__ == "__main__":
    generate_training_dataset()
