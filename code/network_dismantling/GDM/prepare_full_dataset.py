"""
Prepare full training dataset from dataset/data/raw/train/.
Reads .gml/.graphml files, computes features, and generates labels using CoreHD heuristic.
Saves as .npz for fast loading during training.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

import networkx as nx
import numpy as np

from network_dismantling.GDM.training_data_extractor_nx import compute_features
from network_dismantling.CoreHD.corehd_nx import corehd_dismantle


def _process_network(args) -> Tuple[str, bool]:
    """Worker: read network, compute features, CoreHD labels, save .npz."""
    filepath, out_dir, threshold = args
    fname = Path(filepath).stem
    out_path = Path(out_dir) / f"{fname}.npz"
    
    if out_path.exists():
        return fname, True  # Already processed
    
    try:
        # Read network
        if str(filepath).endswith('.gml'):
            G = nx.read_gml(str(filepath))
        elif str(filepath).endswith('.graphml'):
            G = nx.read_graphml(str(filepath))
        else:
            return fname, False
        
        # Convert to simple undirected graph
        if G.is_directed():
            G = G.to_undirected()
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        
        n = G.number_of_nodes()
        if n < 10:
            return fname, False  # Too small
        
        # Relabel to 0..n-1
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        
        # Compute features
        features = compute_features(G, features=["degree", "clustering_coefficient", "kcore", "chi_degree"])
        x = np.column_stack([features[f] for f in ["degree", "clustering_coefficient", "kcore", "chi_degree"]])
        
        # Generate labels using CoreHD heuristic
        stop_condition = max(1, int(np.ceil(n * threshold)))
        removal_seq = corehd_dismantle(G, stop_condition=stop_condition, seed=42)
        
        # Label: nodes in the CoreHD removal set get high score
        # Others get 0. We also use removal order: earlier = more important
        y = np.zeros(n, dtype=np.float32)
        for i, node in enumerate(removal_seq):
            if node < n:
                # Earlier removals are more important
                y[node] = 1.0 - (i / max(len(removal_seq), 1)) * 0.5
        
        edges = np.array(list(G.edges()), dtype=np.int64)
        if edges.shape[0] > 0:
            edges_undirected = np.vstack([edges, edges[:, [1, 0]]])
        else:
            edges_undirected = edges
        
        np.savez_compressed(
            out_path,
            x=x.astype(np.float32),
            y=y.astype(np.float32),
            edges=edges_undirected.astype(np.int64),
            n=n,
        )
        return fname, True
    except Exception as e:
        return fname, False


def prepare_dataset(
    in_dir: str = "dataset/data/raw/train",
    out_dir: str = "network_dismantling/GDM/dataset/full_train",
    threshold: float = 0.18,
    n_jobs: int = None,
):
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all network files
    files = []
    for ext in ["*.gml", "*.graphml"]:
        files.extend(Path(in_dir).glob(ext))
    
    print(f"Found {len(files)} networks. Processing with {n_jobs} workers...")
    
    tasks = [(f, out_dir, threshold) for f in files]
    
    success_count = 0
    with Pool(processes=n_jobs) as pool:
        for fname, success in pool.imap_unordered(_process_network, tasks):
            if success:
                success_count += 1
                if success_count % 100 == 0:
                    print(f"  Processed {success_count}/{len(files)} networks...")
            else:
                print(f"  FAILED: {fname}")
    
    print(f"Done! Successfully processed {success_count}/{len(files)} networks.")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    prepare_dataset()
