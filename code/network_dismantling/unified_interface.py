"""
Unified network dismantling interface.
All algorithms accept networkx.Graph and return a dismantling sequence.
"""
import logging
from typing import List, Optional, Callable, Dict
from functools import wraps

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

METHOD_REGISTRY: Dict[str, Callable] = {}


def register_method(name: str):
    """Decorator to register a dismantling method."""
    def decorator(func: Callable):
        METHOD_REGISTRY[name] = func
        return func
    return decorator


def _standardize_graph(G: nx.Graph) -> nx.Graph:
    """
    Standardize a networkx graph for dismantling:
    - Convert to undirected simple graph (no self-loops, no parallel edges)
    - Relabel nodes to consecutive integers starting from 0
    - Returns the standardized graph and the mapping (new -> old)
    """
    if G.is_directed():
        G = G.to_undirected()
    G = nx.Graph(G)  # remove parallel edges
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # Relabel to consecutive integers 0..n-1
    mapping = {node: i for i, node in enumerate(G.nodes())}
    reverse_mapping = {i: node for node, i in mapping.items()}
    G = nx.relabel_nodes(G, mapping)
    
    G.graph["_reverse_mapping"] = reverse_mapping
    return G


def _fill_remaining(G: nx.Graph, sequence: List[int]) -> List[int]:
    """Fill remaining nodes (after stop_condition) by degree descending."""
    removed = set(sequence)
    remaining = [v for v in G.nodes() if v not in removed]
    remaining.sort(key=lambda v: G.degree(v), reverse=True)
    return list(sequence) + remaining


def dismantle(G: nx.Graph, method: str, stop_condition: Optional[int] = None, **kwargs) -> List[int]:
    """
    Unified dismantling interface.
    
    Parameters
    ----------
    G : networkx.Graph
        Input network (will be converted to undirected simple graph)
    method : str
        Dismantling method name (e.g., 'degree', 'pagerank', 'betweenness', 
        'eigenvector', 'random', 'brute_force', 'entanglement_small', 
        'entanglement_mid', 'entanglement_large', 'vertex_entanglement')
    stop_condition : int, optional
        Stop dismantling when LCC <= stop_condition. 
        If None, dismantle until all nodes are removed.
    **kwargs : additional method-specific parameters
    
    Returns
    -------
    List[int]
        Dismantling sequence of all nodes (original node IDs)
    """
    if method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method '{method}'. Available: {list(METHOD_REGISTRY.keys())}")
    
    # Standardize graph
    G_std = _standardize_graph(G)
    reverse_mapping = G_std.graph["_reverse_mapping"]
    n = G_std.number_of_nodes()
    
    if stop_condition is None:
        stop_condition = 1  # dismantle all
    
    # Call registered method
    func = METHOD_REGISTRY[method]
    seq_std = func(G_std, stop_condition=stop_condition, **kwargs)
    
    # Ensure complete sequence
    if len(seq_std) < n:
        seq_std = _fill_remaining(G_std, seq_std)
    
    # Map back to original node IDs
    seq_orig = [reverse_mapping[v] for v in seq_std]
    return seq_orig


# ---------------------------------------------------------------------------
# Helper: progressive dismantling for static scores
# ---------------------------------------------------------------------------
def _dismantle_by_scores(G: nx.Graph, scores: np.ndarray, stop_condition: int) -> List[int]:
    """
    Given a static score for each node (higher = more important to remove),
    progressively remove nodes and return removal sequence until stop_condition.
    scores must be aligned with node indices 0..n-1.
    """
    n = G.number_of_nodes()
    assert len(scores) == n, f"scores length {len(scores)} != nodes {n}"
    
    G_tmp = G.copy()
    removed = []
    scores = scores.copy()
    
    while G_tmp.number_of_nodes() > 0:
        # Node with highest score among remaining
        remaining = list(G_tmp.nodes())
        if not remaining:
            break
        
        idx = max(remaining, key=lambda v: scores[v])
        removed.append(idx)
        G_tmp.remove_node(idx)
        
        # Check stop condition
        if G_tmp.number_of_nodes() > 0:
            components = list(nx.connected_components(G_tmp))
            lcc_size = max(len(c) for c in components) if components else 0
            if lcc_size <= stop_condition:
                break
        else:
            break
    
    return removed


# ---------------------------------------------------------------------------
# Heuristics (static scores)
# ---------------------------------------------------------------------------
from network_dismantling.heuristics.sorters_nx import (
    degree_scores,
    pagerank_scores,
    betweenness_scores,
    eigenvector_scores,
    random_scores,
)


@register_method("degree")
def _degree_dismantler(G: nx.Graph, stop_condition: int, **kwargs) -> List[int]:
    scores = degree_scores(G)
    return _dismantle_by_scores(G, scores, stop_condition)


@register_method("pagerank")
def _pagerank_dismantler(G: nx.Graph, stop_condition: int, **kwargs) -> List[int]:
    scores = pagerank_scores(G)
    return _dismantle_by_scores(G, scores, stop_condition)


@register_method("betweenness")
def _betweenness_dismantler(G: nx.Graph, stop_condition: int, **kwargs) -> List[int]:
    scores = betweenness_scores(G)
    return _dismantle_by_scores(G, scores, stop_condition)


@register_method("eigenvector")
def _eigenvector_dismantler(G: nx.Graph, stop_condition: int, **kwargs) -> List[int]:
    scores = eigenvector_scores(G)
    return _dismantle_by_scores(G, scores, stop_condition)


@register_method("random")
def _random_dismantler(G: nx.Graph, stop_condition: int, seed: int = None, **kwargs) -> List[int]:
    scores = random_scores(G, seed=seed)
    return _dismantle_by_scores(G, scores, stop_condition)


# ---------------------------------------------------------------------------
# Brute Force (for small networks)
# ---------------------------------------------------------------------------
@register_method("brute_force")
def _brute_force_dismantler(G: nx.Graph, stop_condition: int, max_k: int = None, **kwargs) -> List[int]:
    """
    Brute-force dismantler for small networks.
    Finds the smallest set of nodes whose removal reduces LCC to <= stop_condition.
    Remaining nodes are sorted by degree descending.
    """
    from itertools import combinations
    
    n = G.number_of_nodes()
    if max_k is None:
        max_k = min(n, 10)
    
    nodes = list(G.nodes())
    best_sequence = None
    
    for k in range(1, max_k + 1):
        best_score = n
        best_combo = None
        
        for combo in combinations(nodes, k):
            temp = G.copy()
            temp.remove_nodes_from(combo)
            if temp.number_of_nodes() > 0:
                components = list(nx.connected_components(temp))
                lcc = max(len(c) for c in components) if components else 0
            else:
                lcc = 0
            
            if lcc < best_score:
                best_score = lcc
                best_combo = combo
            
            if best_score <= stop_condition:
                break
        
        if best_score <= stop_condition and best_combo is not None:
            best_sequence = list(best_combo)
            break
    
    if best_sequence is None:
        # Fallback to degree
        best_sequence = sorted(nodes, key=lambda v: G.degree(v), reverse=True)[:max_k]
    
    remaining = [v for v in nodes if v not in best_sequence]
    remaining.sort(key=lambda v: G.degree(v), reverse=True)
    return best_sequence + remaining


# ---------------------------------------------------------------------------
# Multiscale Entanglement (networkx version)
# ---------------------------------------------------------------------------
from network_dismantling.multiscale_entanglement.original_entanglement_functions import (
    entanglement_small as _entanglement_small,
    entanglement_mid as _entanglement_mid,
    entanglement_large as _entanglement_large,
)


@register_method("entanglement_small")
def _entanglement_small_dismantler(G: nx.Graph, stop_condition: int, **kwargs) -> List[int]:
    ent = _entanglement_small(G)
    # ent is dict {node: entanglement_value}
    # Higher entanglement = more important node, so higher score for removal
    scores = np.zeros(G.number_of_nodes())
    for v, val in ent.items():
        scores[v] = val
    return _dismantle_by_scores(G, scores, stop_condition)


@register_method("entanglement_mid")
def _entanglement_mid_dismantler(G: nx.Graph, stop_condition: int, **kwargs) -> List[int]:
    ent = _entanglement_mid(G)
    scores = np.zeros(G.number_of_nodes())
    for v, val in ent.items():
        scores[v] = val
    return _dismantle_by_scores(G, scores, stop_condition)


@register_method("entanglement_large")
def _entanglement_large_dismantler(G: nx.Graph, stop_condition: int, **kwargs) -> List[int]:
    ent = _entanglement_large(G)
    scores = np.zeros(G.number_of_nodes())
    for v, val in ent.items():
        scores[v] = val
    return _dismantle_by_scores(G, scores, stop_condition)


# ---------------------------------------------------------------------------
# CI (Collective Influence) - C executable wrapper
# ---------------------------------------------------------------------------
from network_dismantling.CI.ci_wrapper_nx import ci_dismantle_nx


@register_method("CI_L1")
def _ci_l1_dismantler(G: nx.Graph, stop_condition: int, **kwargs) -> List[int]:
    return ci_dismantle_nx(G, l=1, stop_condition=stop_condition)


@register_method("CI_L2")
def _ci_l2_dismantler(G: nx.Graph, stop_condition: int, **kwargs) -> List[int]:
    return ci_dismantle_nx(G, l=2, stop_condition=stop_condition)


@register_method("CI_L3")
def _ci_l3_dismantler(G: nx.Graph, stop_condition: int, **kwargs) -> List[int]:
    return ci_dismantle_nx(G, l=3, stop_condition=stop_condition)


# ---------------------------------------------------------------------------
# CoreHD (Python reimplementation)
# ---------------------------------------------------------------------------
from network_dismantling.CoreHD.corehd_nx import corehd_dismantle


@register_method("CoreHD")
def _corehd_dismantler(G: nx.Graph, stop_condition: int, seed: int = None, **kwargs) -> List[int]:
    seq = corehd_dismantle(G, stop_condition=stop_condition, seed=seed)
    # Fill remaining nodes by degree descending
    return _fill_remaining(G, seq)


# ---------------------------------------------------------------------------
# GND / EGND (Python reimplementation)
# ---------------------------------------------------------------------------
from network_dismantling.GND.gnd_nx import gnd_dismantle


@register_method("GND")
def _gnd_dismantler(G: nx.Graph, stop_condition: int, remove_strategy: int = 3, seed: int = None, **kwargs) -> List[int]:
    seq = gnd_dismantle(G, stop_condition=stop_condition, remove_strategy=remove_strategy, seed=seed)
    return _fill_remaining(G, seq)


# ---------------------------------------------------------------------------
# EI (Python reimplementation)
# ---------------------------------------------------------------------------
from network_dismantling.EI.ei_nx import ei_dismantle


@register_method("EI_s1")
def _ei_s1_dismantler(G: nx.Graph, stop_condition: int, kk: int = 1000, seed: int = None, **kwargs) -> List[int]:
    return ei_dismantle(G, stop_condition=stop_condition, sigma=1, kk=kk, seed=seed)


@register_method("EI_s2")
def _ei_s2_dismantler(G: nx.Graph, stop_condition: int, kk: int = 1000, seed: int = None, **kwargs) -> List[int]:
    return ei_dismantle(G, stop_condition=stop_condition, sigma=2, kk=kk, seed=seed)


@register_method("EGND")
def _egnd_dismantler(G: nx.Graph, stop_condition: int, runs: int = 10, remove_strategy: int = 3, seed: int = None, **kwargs) -> List[int]:
    """
    Ensemble GND: run GND multiple times with different seeds and pick the best result
    (minimum number of removed nodes).
    """
    rng = np.random.default_rng(seed)
    best_seq = None
    best_len = float('inf')
    
    for i in range(runs):
        run_seed = rng.integers(0, 2**31)
        seq = gnd_dismantle(G, stop_condition=stop_condition, remove_strategy=remove_strategy, seed=run_seed)
        if len(seq) < best_len:
            best_len = len(seq)
            best_seq = seq
    
    return _fill_remaining(G, best_seq)


# ---------------------------------------------------------------------------
# Vertex Entanglement (networkx version)
# ---------------------------------------------------------------------------
from network_dismantling.vertex_entanglement.vertex_entanglement_nx import VertexEnt_nx


@register_method("vertex_entanglement")
def _vertex_entanglement_dismantler(G: nx.Graph, stop_condition: int, **kwargs) -> List[int]:
    """
    Vertex Entanglement dismantler.
    Uses numpy spectral computation on networkx graph.
    """
    # VE returns array where lower value = more important for dismantling
    ve = VertexEnt_nx(G, perturb_strategy='default')
    # Invert: lower VE -> higher priority
    scores = -ve
    return _dismantle_by_scores(G, scores, stop_condition)


# ---------------------------------------------------------------------------
# GDM (Graph Dismantling Machine) - networkx reimplementation
# ---------------------------------------------------------------------------
# NOTE: GDM requires the trained model files (network_dismantling/GDM/models/GAT.py
# and network_dismantling/GDM/models_newpg/gdm_nx_best.pth), which are gitignored
# and not shipped with this repository. The import is deferred until first use,
# so all other methods remain importable without them.


@register_method("GDM")
def _gdm_dismantler(G: nx.Graph, stop_condition: int, **kwargs) -> List[int]:
    """
    GDM (Graph Dismantling Machine) dismantler.
    Uses a pre-trained GAT model to predict node importance scores.
    """
    from network_dismantling.GDM.predictors_nx import gdm_dismantle_from_path
    model_path = kwargs.get("model_path", "network_dismantling/GDM/models_newpg/gdm_nx_best.pth")
    device = kwargs.get("device", "cuda" if __import__("torch").cuda.is_available() else "cpu")
    seq = gdm_dismantle_from_path(G, model_path, stop_condition, device=device)
    return _fill_remaining(G, seq)


@register_method("GDM+R")
def _gdm_reinsertion_dismantler(G: nx.Graph, stop_condition: int, **kwargs) -> List[int]:
    """
    GDM + Reinsertion dismantler.
    Uses GDM for initial prediction, then applies greedy reinsertion optimization.
    """
    from network_dismantling.GDM.predictors_nx import gdm_dismantle_from_path
    model_path = kwargs.get("model_path", "network_dismantling/GDM/models_newpg/gdm_nx_best.pth")
    device = kwargs.get("device", "cuda" if __import__("torch").cuda.is_available() else "cpu")
    seq = gdm_dismantle_from_path(G, model_path, stop_condition, device=device, use_reinsertion=True)
    return _fill_remaining(G, seq)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  【Mamba 对接 - 修改部分】                                             ║
# ║  以下为本项目新增内容：将 Mamba 方法注册到统一拆解接口                  ║
# ║  - 新增 import: network_dismantling.Mamba.mamba_dismantler            ║
# ║  - 新增 @register_method("mamba") 注册函数                            ║
# ║  - 支持 kwargs: device（计算设备）、model_path（训练权重检查点）         ║
# ║  调用方式与 degree / CoreHD 等方法完全一致:                             ║
# ║      dismantle(G, method='mamba')                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
# ---------------------------------------------------------------------------
# Mamba (基于 mamba_ssm 的序列模型)
# ---------------------------------------------------------------------------
from network_dismantling.Mamba.mamba_dismantler import mamba_dismantle


@register_method("mamba")
def _mamba_dismantler(G: nx.Graph, stop_condition: int, **kwargs) -> List[int]:
    """
    Mamba 网络拆解算法。
    使用 Mamba 序列模型对节点进行优先级评分。
    可通过 kwargs 传入:
    - device: 计算设备 ('cuda' 或 'cpu')
    - model_path: 训练好的检查点路径（可选，优先级高于全局注册权重）
    """
    device = kwargs.get("device", "cuda" if __import__("torch").cuda.is_available() else "cpu")
    seq = mamba_dismantle(
        G,
        stop_condition=stop_condition,
        device=device,
        model_path=kwargs.get("model_path"),
    )
    return _fill_remaining(G, seq)
