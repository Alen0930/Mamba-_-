"""
GND – pure Python/networkx implementation.

Reference:
> Ren, X.-L., Gleinig, N., Helbing, D., Antulov-Fantulin, N. Generalized network dismantling. PNAS 116(14), 6554–6559 (2019). DOI: 10.1073/pnas.1806108116
"""
"""
Pure Python/networkx implementation of Generalized Network Dismantling (GND).
Based on the C++ code in GND.cpp.

Algorithm (per iteration):
  1. Extract the GCC.
  2. Compute the Fiedler vector via power iteration on the graph Laplacian.
  3. Split GCC into two clusters by the sign of the eigenvector.
  4. Build cut-edge subgraph (edges crossing the sign boundary).
  5. Greedy vertex cover on cut edges.
  6. Remove cover nodes.
  7. Repeat until GCC size <= stop_condition.
"""
from typing import List, Tuple
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def gnd_dismantle(G: nx.Graph, stop_condition: int, remove_strategy: int = 3,
                  maxiter: int = None, tol: float = 1e-6, seed: int = None) -> List[int]:
    """
    GND dismantler.
    
    Parameters
    ----------
    G : networkx.Graph
        Nodes must be 0..n-1.
    stop_condition : int
        Target size for GCC.
    remove_strategy : int
        3 = unweighted (default), 1 = weighted.
    maxiter : int, optional
        Max power iteration steps. Default: 30 * log(n) * sqrt(log(n))
    tol : float
        Convergence tolerance for power iteration.
    seed : int, optional
        Random seed for power iteration initializer.
    
    Returns
    -------
    List[int]
        Removal sequence.
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    if maxiter is None:
        maxiter = int(30 * np.log(max(n, 2)) * np.sqrt(np.log(max(n, 2))))
    
    # Working graph
    H = G.copy()
    removed = []
    
    while True:
        # Get GCC
        gcc_nodes = _get_gcc(H)
        if len(gcc_nodes) <= stop_condition:
            break
        
        # Relabel GCC to 0..m-1 for sparse matrix operations
        gcc_list = sorted(gcc_nodes)
        idx_map = {v: i for i, v in enumerate(gcc_list)}
        m = len(gcc_list)
        
        # Build adjacency matrix of GCC
        row, col = [], []
        for u in gcc_list:
            for v in H.neighbors(u):
                if v in idx_map:
                    row.append(idx_map[u])
                    col.append(idx_map[v])
        data = np.ones(len(row), dtype=float)
        A = csr_matrix((data, (row, col)), shape=(m, m))
        
        # Compute degree array
        degrees = np.array(A.sum(axis=1)).flatten()
        d_max = degrees.max()
        
        # Power iteration to get Fiedler vector
        if remove_strategy == 3:
            # Unweighted: shifted Laplacian L_tilde = d_max * I - L
            x = rng.normal(size=m)
            x -= x.mean()
            x /= np.linalg.norm(x)
            for _ in range(maxiter):
                # y = (d_max * I - L) @ x = d_max * x - (D - A) @ x
                y = d_max * x - (degrees * x - A @ x)
                y -= y.mean()
                norm = np.linalg.norm(y)
                if norm == 0:
                    break
                y /= norm
                if np.linalg.norm(x - y) < tol:
                    x = y
                    break
                x = y
            eigvec = x.copy()
        elif remove_strategy == 1:
            # Weighted: operator B = WA + AW - A
            # y_i = (d_i - 1) * sum_j A_ij x_j + sum_j A_ij d_j x_j + (d_max - db_i) x_i
            # where db_i = d_i*(d_i-1) + sum_{j in N(i)} d_j
            db = degrees * (degrees - 1) + A @ degrees
            x = rng.normal(size=m)
            x -= x.mean()
            x /= np.linalg.norm(x)
            for _ in range(maxiter):
                Ax = A @ x
                y = (degrees - 1) * Ax + A @ (degrees * x) + (d_max - db) * x
                y -= y.mean()
                norm = np.linalg.norm(y)
                if norm == 0:
                    break
                y /= norm
                if np.linalg.norm(x - y) < tol:
                    x = y
                    break
                x = y
            eigvec = x.copy()
        else:
            raise ValueError(f"Unknown remove_strategy: {remove_strategy}")
        
        # Sign partition
        signs = np.sign(eigvec)
        # Adjust: if a node is surrounded entirely by neighbors of opposite sign, flip it
        for _ in range(5):
            flipped = False
            for v_idx in range(m):
                v = gcc_list[v_idx]
                neighbors = [idx_map[u] for u in H.neighbors(v) if u in idx_map]
                if not neighbors:
                    continue
                neighbor_signs = signs[neighbors]
                if len(neighbor_signs) > 0 and np.all(neighbor_signs != signs[v_idx]):
                    signs[v_idx] = -signs[v_idx]
                    flipped = True
            if not flipped:
                break
        
        # Build cut-edge subgraph
        cover_nodes = set()
        cut_edges = []
        for u in gcc_list:
            for v in H.neighbors(u):
                if v in idx_map and u < v:  # avoid duplicates
                    if signs[idx_map[u]] != signs[idx_map[v]]:
                        cut_edges.append((u, v))
                        cover_nodes.add(u)
                        cover_nodes.add(v)
        
        if not cut_edges:
            # No cut edges: remove highest-degree node in GCC
            v = max(gcc_list, key=lambda u: H.degree(u))
            H.remove_node(v)
            removed.append(v)
            continue
        
        # Greedy vertex cover on cut edges
        cover_set = set(cover_nodes)
        cover_degrees = {u: sum(1 for e in cut_edges if u in e) for u in cover_set}
        gcc_degrees = {u: H.degree(u) for u in cover_set}
        
        to_remove = []
        remaining_edges = set(cut_edges)
        
        while remaining_edges:
            # Pick node according to strategy
            if remove_strategy == 3:
                # Unweighted: argmax degree in cover graph -> remove largest-degree-first in original
                # C++ vertex_cover picks min(1 / cover_degree), which is equivalent to max cover_degree
                # But removal order is largest original degree first
                best_u = max(cover_set, key=lambda u: gcc_degrees.get(u, 0))
            else:
                # Weighted: argmin( gcc_degree / cover_degree )
                best_u = min(cover_set, key=lambda u: gcc_degrees.get(u, 1) / max(cover_degrees.get(u, 1), 1e-10))
            
            to_remove.append(best_u)
            cover_set.remove(best_u)
            
            # Remove all edges incident to best_u
            remaining_edges = {e for e in remaining_edges if best_u not in e}
            # Update cover_degrees
            for u in list(cover_degrees.keys()):
                cover_degrees[u] = sum(1 for e in remaining_edges if u in e)
            cover_degrees = {u: d for u, d in cover_degrees.items() if d > 0}
            cover_set = {u for u in cover_set if cover_degrees.get(u, 0) > 0}
        
        # Remove nodes from H
        # Strategy 3: remove largest current degree first
        # Strategy 1: remove smallest current degree first
        if remove_strategy == 3:
            to_remove.sort(key=lambda u: H.degree(u), reverse=True)
        else:
            to_remove.sort(key=lambda u: H.degree(u))
        
        for u in to_remove:
            if u in H:
                H.remove_node(u)
                removed.append(u)
    
    return removed


def _get_gcc(G: nx.Graph) -> set:
    """Return the set of nodes in the giant connected component."""
    if G.number_of_nodes() == 0:
        return set()
    return max(nx.connected_components(G), key=len)
