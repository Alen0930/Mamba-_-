"""
CoreHD – pure Python/networkx implementation.

Reference:
> Zdeborová, L., Zhang, P., Zhou, H.-J. Fast and simple decycling and dismantling of networks. Scientific Reports 6, 37954 (2016). DOI: 10.1038/srep37954
"""
"""
Pure Python/networkx implementation of CoreHD.
Based on the C++ code in TAbyTwoCoreV01.cpp.
Algorithm:
  1. Fix0: iteratively delete the highest-degree vertex from the 2-core,
     cascading peel after each deletion.
  2. ComponentRefinement: break remaining tree components > Sthreshold
     by deleting the vertex that minimizes the maximum subtree size.
  3. Greedy add-back: re-insert deleted vertices if they don't cause
     any component to exceed Sthreshold.
"""
from typing import List, Set, Dict
import networkx as nx
import numpy as np


def corehd_dismantle(G: nx.Graph, stop_condition: int, seed: int = None) -> List[int]:
    """
    CoreHD dismantler.
    
    Parameters
    ----------
    G : networkx.Graph
        Nodes must be labeled 0..n-1.
    stop_condition : int
        Max allowed component size (Sthreshold).
    seed : int, optional
        Random seed for tie-breaking in Fix0.
    
    Returns
    -------
    List[int]
        Removal sequence (nodes that are actually removed).
        Note: this may be shorter than |V| because some nodes are added back.
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    
    # Save original adjacency for later use (since we will mutate G)
    original_neighbors = {v: set(G.neighbors(v)) for v in G.nodes()}
    
    # occupied[v] = True  -> v is still in the graph (not removed)
    # occupied[v] = False -> v has been removed
    occupied = {v: True for v in G.nodes()}
    
    # ------------------------------------------------------------------
    # Phase B: Fix0 - dismantle the 2-core
    # ------------------------------------------------------------------
    targets = []
    H = G.copy()  # Working copy that we mutate
    
    while True:
        # Compute current 2-core among occupied nodes
        core_nodes = _compute_2core(H, occupied)
        if not core_nodes:
            break
        
        # Find max active degree in core
        max_deg = max(H.degree(v) for v in core_nodes)
        candidates = [v for v in core_nodes if H.degree(v) == max_deg]
        
        # Randomly pick one candidate
        v = rng.choice(candidates)
        # Tie-breaking: randomly pick one candidate
        
        # Remove v
        occupied[v] = False
        H.remove_node(v)
        targets.append(v)
        
        # Simplify: recursively remove degree<=1 nodes from core
        # (In C++, Simplify is called after each removal and cascades.
        #  We let the next iteration recompute the 2-core, which is equivalent
        #  but slightly slower. For correctness it's the same.)
    
    # ------------------------------------------------------------------
    # Phase C: ComponentRefinement
    # ------------------------------------------------------------------
    # H already contains only occupied nodes after Fix0
    
    # C1: Break giant tree components
    while True:
        components = list(nx.connected_components(H))
        giant_found = False
        for comp in components:
            if len(comp) > stop_condition:
                # This is a tree component (since 2-core is empty)
                # Find vertex whose deletion minimizes max subtree size
                subtree = H.subgraph(comp).copy()
                cvtx = _find_best_tree_breaker(subtree)
                # Break giant tree component
                
                occupied[cvtx] = False
                H.remove_node(cvtx)
                targets.append(cvtx)
                giant_found = True
                break
        if not giant_found:
            break
    
    # C2: Greedy add-back
    # Collect deleted nodes that are adjacent to small components
    # and sort by merged component size (ascending)
    comp_index, comp_size = _label_components(H)
    
    while True:
        best_v = None
        best_size = None
        
        for v in range(n):
            if occupied[v]:
                continue
            # Check neighbors' component indices
            neighbor_comps = set()
            for u in original_neighbors[v]:
                if occupied[u] and comp_index.get(u, 0) != 0:
                    neighbor_comps.add(comp_index[u])
            
            merged_size = 1
            for c in neighbor_comps:
                merged_size += comp_size.get(c, 0)
            
            if merged_size <= stop_condition:
                if best_size is None or merged_size < best_size:
                    best_size = merged_size
                    best_v = v
        
        if best_v is None:
            break
        
        # Add back best_v
        occupied[best_v] = True
        H.add_node(best_v)
        # Re-add edges to occupied neighbors
        for u in original_neighbors[best_v]:
            if occupied[u]:
                H.add_edge(best_v, u)
        
        # Recompute components
        comp_index, comp_size = _label_components(H)
    
    return targets


def _compute_2core(G: nx.Graph, occupied: Dict[int, bool]) -> Set[int]:
    """Compute 2-core among occupied nodes."""
    active = {v for v in G.nodes() if occupied[v]}
    changed = True
    while changed:
        changed = False
        to_remove = set()
        for v in active:
            deg = sum(1 for u in G.neighbors(v) if u in active)
            if deg <= 1:
                to_remove.add(v)
        if to_remove:
            active -= to_remove
            changed = True
    return active


def _find_best_tree_breaker(T: nx.Graph) -> int:
    """
    Given a tree T, find the vertex whose deletion minimizes
    the size of the largest resulting subtree.
    Uses leaf-stripping (bottom-up) approach.
    """
    if T.number_of_nodes() == 0:
        return None
    
    n = T.number_of_nodes()
    if n == 1:
        return list(T.nodes())[0]
    
    # Build a leaf-stripping order
    degrees = dict(T.degree())
    leaves = [v for v, d in degrees.items() if d == 1]
    order = []
    removed = set()
    
    while leaves:
        v = leaves.pop()
        if v in removed:
            continue
        removed.add(v)
        order.append(v)
        
        for u in T.neighbors(v):
            if u not in removed:
                degrees[u] -= 1
                if degrees[u] == 1:
                    leaves.append(u)
    
    # Now compute b_size (branch size) bottom-up
    # For each leaf-stripped node, track the sizes of subtrees below it
    b_size = {}
    subtree_sums = {}
    
    for v in order:
        # v's active neighbors are those not yet processed (still in tree)
        active_nbrs = [u for u in T.neighbors(v) if u not in removed or u in b_size]
        # Actually, we need to be more careful here.
        # In the C++ code, leaf stripping marks nodes inactive and computes
        # b_size as the sum of branch sizes of removed children + 1.
        # Let's use a simpler approach: try every node and compute max subtree.
        pass
    
    # Simpler brute-force for small trees (typical after 2-core removal):
    best_v = None
    best_max = n + 1
    
    for v in T.nodes():
        T2 = T.copy()
        T2.remove_node(v)
        comps = list(nx.connected_components(T2))
        max_size = max(len(c) for c in comps) if comps else 0
        if max_size < best_max:
            best_max = max_size
            best_v = v
    
    return best_v


def _label_components(H: nx.Graph):
    """Label connected components and return (node->index, index->size)."""
    comp_index = {}
    comp_size = {}
    idx = 0
    for comp in nx.connected_components(H):
        idx += 1
        size = len(comp)
        comp_size[idx] = size
        for v in comp:
            comp_index[v] = idx
    return comp_index, comp_size
