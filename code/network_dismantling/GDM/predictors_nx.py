"""
GDM predictor and dismantler (networkx version).

Reference:
> Grassia, M., De Domenico, M., Mangioni, G. Machine learning dismantling and early-warning signals of disintegration in complex systems. Nature Communications 12, 5190 (2021). DOI: 10.1038/s41467-021-25485-8
"""
"""
GDM predictor and dismantler (networkx version).
Provides static prediction, progressive dismantling, and reinsertion using a trained GAT model.
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

from network_dismantling.GDM.models.GAT import GAT_Model
from network_dismantling.GDM.dataset_providers_nx import prepare_graph_nx
from network_dismantling.GDM.training_data_extractor_nx import extract_training_data
from network_dismantling.GDM.reinsert_nx import reinsert_nodes

logger = logging.getLogger(__name__)


def load_model(
    model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[GAT_Model, List[str], dict]:
    """
    Load a trained GAT model from checkpoint.
    Returns (model, features_list, config).
    """
    checkpoint = torch.load(model_path, map_location=device)
    features = checkpoint["features"]
    config = checkpoint["config"]

    class Args:
        def __init__(self):
            self.features = features
            self.num_features = len(features)
            self.conv_layers = config["conv_layers"]
            self.heads = config["heads"]
            self.fc_layers = config["fc_layers"]
            self.concat = config.get("concat", [True] * len(self.conv_layers))
            self.negative_slope = config.get("negative_slope", [0.2] * len(self.conv_layers))
            self.dropout = config.get("dropout", [0.3] * len(self.conv_layers))
            self.bias = config.get("bias", [True] * len(self.conv_layers))
            self.seed_train = config.get("seed_train", 0)

    args = Args()
    model = GAT_Model(args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, features, config


def predict_scores(
    G: nx.Graph,
    model: GAT_Model,
    features: List[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> np.ndarray:
    """
    Predict dismantling scores for all nodes in G.
    Higher score = higher priority to remove.
    Returns (n,) numpy array aligned with sorted node IDs.
    """
    x, _ = extract_training_data(G, features=features, compute_targets=False)
    data = prepare_graph_nx(G, x, y=None)
    data = data.to(device)

    with torch.no_grad():
        scores = model(data.x, data.edge_index)
    scores = scores.cpu().numpy()
    return scores


def gdm_dismantle(
    G: nx.Graph,
    model: GAT_Model,
    features: List[str],
    stop_condition: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[int]:
    """
    Static GDM dismantler.
    1. Predict all node scores once.
    2. Progressively remove highest-score nodes until LCC <= stop_condition.
    3. Return removal sequence.
    """
    scores = predict_scores(G, model, features, device=device)
    nodes = sorted(G.nodes())

    G_tmp = G.copy()
    removed = []

    while G_tmp.number_of_nodes() > 0:
        remaining = list(G_tmp.nodes())
        if not remaining:
            break

        best_node = max(remaining, key=lambda v: scores[nodes.index(v)])
        removed.append(best_node)
        G_tmp.remove_node(best_node)

        if G_tmp.number_of_nodes() > 0:
            components = list(nx.connected_components(G_tmp))
            lcc_size = max(len(c) for c in components) if components else 0
            if lcc_size <= stop_condition:
                break
        else:
            break

    return removed


def gdm_dismantle_with_reinsertion(
    G: nx.Graph,
    model: GAT_Model,
    features: List[str],
    stop_condition: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[int]:
    """
    GDM + Reinsertion dismantler.
    1. Get initial GDM sequence.
    2. Run greedy reinsertion to minimize removals.
    3. Return optimized sequence.
    """
    initial_removals = gdm_dismantle(G, model, features, stop_condition, device=device)
    optimized = reinsert_nodes(G, initial_removals, stop_condition)
    return optimized


def gdm_dismantle_from_path(
    G: nx.Graph,
    model_path: str,
    stop_condition: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_reinsertion: bool = False,
) -> List[int]:
    """
    Convenience function: load model from path and dismantle.
    """
    model, features, _ = load_model(model_path, device=device)
    if use_reinsertion:
        return gdm_dismantle_with_reinsertion(G, model, features, stop_condition, device=device)
    else:
        return gdm_dismantle(G, model, features, stop_condition, device=device)
