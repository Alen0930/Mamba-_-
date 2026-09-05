"""
Convert networkx graphs or precomputed .npz files to PyG Data objects.
(networkx version, replaces graph_tool dependency.)
"""
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch_geometric.data import Data

import networkx as nx


def prepare_graph_nx(
    G: nx.Graph,
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
) -> Data:
    """
    Convert a networkx graph with precomputed features to a PyG Data object.
    """
    n = G.number_of_nodes()
    assert x.shape[0] == n, f"Feature matrix rows {x.shape[0]} != nodes {n}"

    x_tensor = torch.from_numpy(x).to(torch.float)
    y_tensor = torch.from_numpy(y).to(torch.float) if y is not None else None

    edges = list(G.edges())
    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        src = [e[0] for e in edges] + [e[1] for e in edges]
        dst = [e[1] for e in edges] + [e[0] for e in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    return Data(x=x_tensor, edge_index=edge_index, y=y_tensor)


def load_data_from_npz(npz_path: str) -> Data:
    """
    Load a precomputed .npz file into a PyG Data object.
    Expects keys: x, y, edges (undirected, both directions).
    """
    data = np.load(npz_path)
    x = torch.from_numpy(data["x"]).to(torch.float)
    y = torch.from_numpy(data["y"]).to(torch.float)
    edge_index = torch.from_numpy(data["edges"].T).to(torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def init_network_provider_nx(
    location: str,
    features: Optional[List[str]] = None,
    threshold: float = 0.18,
    max_k: int = 3,
) -> List[Tuple[str, nx.Graph, Data]]:
    """
    Load pickled networkx graphs from directory, compute features/targets,
    and return list of (name, graph, data) tuples.
    """
    from network_dismantling.GDM.dataset_generator_nx import load_networks
    from network_dismantling.GDM.training_data_extractor_nx import extract_training_data

    networks = load_networks(location)
    provider = []

    for name, G in networks:
        x, y = extract_training_data(
            G,
            features=features,
            threshold=threshold,
            max_k=max_k,
            compute_targets=True,
        )
        data = prepare_graph_nx(G, x, y)
        provider.append((name, G, data))

    return provider


def load_npz_dataset(
    data_dir: str,
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Data], List[Data]]:
    """
    Load all .npz files from directory and split into train/val.
    Returns (train_data, val_data) lists of PyG Data objects.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.npz"))
    rng = np.random.default_rng(seed)
    rng.shuffle(files)

    split_idx = int(len(files) * (1 - val_split))
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    train_data = [load_data_from_npz(str(f)) for f in train_files]
    val_data = [load_data_from_npz(str(f)) for f in val_files]

    return train_data, val_data
