"""
Explosive Immunization (EI) – pure Python/networkx implementation.

Reference:
> Clusella, P., Grassberger, P., Pérez-Reche, F.J., Politi, A. Immunization and Targeted Destruction of Networks using Explosive Percolation. Phys. Rev. Lett. 117, 208301 (2016). DOI: 10.1103/PhysRevLett.117.208301
"""
"""
Pure Python/networkx implementation of Explosive Immunization (EI).
Based on the C code in EI/Library/.

The C++ algorithm builds a kept set incrementally (starting from empty).
It adds nodes one by one; nodes that keep the LCC <= threshold are recorded.

For dismantling, we invert this: nodes NOT in the recorded kept set are removed.
"""
from typing import List, Set
import networkx as nx
import numpy as np


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = np.ones(n, dtype=int)
    
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return
        if self.size[rx] < self.size[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]


def ei_dismantle(G: nx.Graph, stop_condition: int, sigma: int = 2,
                 kk: int = 1000, eff_thr: int = 6, seed: int = None) -> List[int]:
    """
    Explosive Immunization.
    
    Simulates the C++ algorithm: incrementally build a kept set.
    Returns the complement (removed nodes) ordered by degree descending.
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    neighbors = {v: set(G.neighbors(v)) for v in G.nodes()}
    degrees = np.array([G.degree(v) for v in range(n)], dtype=int)
    
    # Effective degree
    effective_deg = _effective_degree(neighbors, degrees, eff_thr)
    
    # State: n_status[v] = True if v is in the kept set
    n_status = np.zeros(n, dtype=bool)
    
    # DSU for kept set
    dsu = DSU(n)
    
    output_nodes = []
    
    for step in range(n):
        # Candidates: nodes not yet in kept set
        available = [v for v in range(n) if not n_status[v]]
        if not available:
            break
        
        imax = min(kk, len(available))
        if len(available) <= kk:
            candidates = available
        else:
            candidates = rng.choice(available, size=imax, replace=False).tolist()
        
        # Evaluate each candidate
        best_score = float('inf')
        best_u = candidates[0]
        
        for u in candidates:
            score = _compute_score(u, neighbors, n_status, dsu, sigma, effective_deg, n)
            if score < best_score:
                best_score = score
                best_u = u
        
        # Add best_u to kept set
        n_status[best_u] = True
        for v in neighbors[best_u]:
            if n_status[v]:
                dsu.union(best_u, v)
        
        # Compute LCC of kept set
        root_sizes = {}
        for u in range(n):
            if n_status[u]:
                r = dsu.find(u)
                root_sizes[r] = root_sizes.get(r, 0) + 1
        lcc = max(root_sizes.values()) if root_sizes else 0
        
        if lcc <= stop_condition:
            output_nodes.append(best_u)
        # Note: C++ code continues even after LCC > threshold,
        # but stops printing. We follow the same logic.
    
    # Removed nodes = all nodes not in output_nodes
    # (In C++, output_nodes are the printed nodes, but the kept set is larger.
    #  For simplicity, we treat output_nodes as the "safe" kept set.)
    removed = [v for v in range(n) if v not in output_nodes]
    # Sort removed by degree descending for dismantling order
    removed.sort(key=lambda v: G.degree(v), reverse=True)
    return removed


def _effective_degree(neighbors, degrees, eff_thr):
    n = len(degrees)
    effective = degrees.copy()
    changed = True
    while changed:
        changed = False
        new_eff = np.zeros(n, dtype=int)
        for i in range(n):
            count = 0
            for j in neighbors[i]:
                if effective[j] <= eff_thr and degrees[j] > 1:
                    count += 1
            new_eff[i] = count
        if not np.array_equal(new_eff, effective):
            changed = True
            effective = new_eff
    return effective


def _compute_score(u, neighbors, n_status, dsu, sigma, effective_deg, n):
    roots = set()
    sizes = []
    for v in neighbors[u]:
        if n_status[v]:
            r = dsu.find(v)
            if r not in roots:
                roots.add(r)
                sizes.append(dsu.size[r])
    
    k = len(roots)
    
    if sigma == 1:
        score = effective_deg[u] + sum(np.sqrt(float(s)) - 1.0 for s in sizes)
    elif sigma == 2:
        sizes_sorted = sorted(sizes, reverse=True)
        c2 = sizes_sorted[1] if len(sizes_sorted) > 1 else 0
        score = k + float(c2) / float(n)
    else:
        raise ValueError(f"Unknown sigma: {sigma}")
    
    return score
