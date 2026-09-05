"""
Networkx-based heuristic sorters for network dismantling.

Reference:
> Classic centrality-based heuristics: degree, betweenness, eigenvector, PageRank, random.
"""
"""
Networkx-based heuristic sorters for network dismantling.
Each function returns a numpy array of scores aligned with node indices 0..n-1.
Higher score = higher priority for removal.
"""
import numpy as np
import networkx as nx


def degree_scores(G: nx.Graph) -> np.ndarray:
    n = G.number_of_nodes()
    scores = np.zeros(n)
    for v in G.nodes():
        scores[v] = G.degree(v)
    return scores


def pagerank_scores(G: nx.Graph, **kwargs) -> np.ndarray:
    n = G.number_of_nodes()
    pr = nx.pagerank(G, **kwargs)
    scores = np.zeros(n)
    for v, val in pr.items():
        scores[v] = val
    return scores


def betweenness_scores(G: nx.Graph, **kwargs) -> np.ndarray:
    n = G.number_of_nodes()
    bc = nx.betweenness_centrality(G, **kwargs)
    scores = np.zeros(n)
    for v, val in bc.items():
        scores[v] = val
    return scores


def eigenvector_scores(G: nx.Graph, **kwargs) -> np.ndarray:
    n = G.number_of_nodes()
    try:
        ec = nx.eigenvector_centrality(G, **kwargs)
    except nx.PowerIterationFailedConvergence:
        # Fallback to a small number of iterations
        ec = nx.eigenvector_centrality(G, max_iter=1000, **kwargs)
    scores = np.zeros(n)
    for v, val in ec.items():
        scores[v] = val
    return scores


def random_scores(G: nx.Graph, seed: int = None) -> np.ndarray:
    n = G.number_of_nodes()
    rng = np.random.default_rng(seed)
    return rng.random(n)
