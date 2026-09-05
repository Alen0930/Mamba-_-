"""
Reinsertion optimizer for network dismantling (networkx version).
Given an initial dismantling sequence that breaks the network (LCC <= stop_condition),
try to reinsert removed nodes in reverse order while maintaining the stop condition.
Returns a minimized removal sequence.
"""
from typing import List
import networkx as nx


def reinsert_nodes(
    G: nx.Graph,
    removals: List[int],
    stop_condition: int,
) -> List[int]:
    """
    Greedy reinsertion optimizer.

    Parameters
    ----------
    G : nx.Graph
        Original graph (nodes 0..n-1)
    removals : List[int]
        Initial removal sequence that already satisfies LCC <= stop_condition
    stop_condition : int
        Maximum allowed LCC size

    Returns
    -------
    List[int]
        Optimized removal sequence (subset of initial removals)
    """
    if not removals:
        return removals

    # Build the graph after all removals
    G_remaining = G.copy()
    removed_set = set(removals)

    # Verify initial sequence indeed breaks the network
    if G_remaining.number_of_nodes() > 0:
        components = list(nx.connected_components(G_remaining))
        lcc = max(len(c) for c in components) if components else 0
    else:
        lcc = 0

    if lcc > stop_condition:
        # Initial sequence didn't fully break - remove nodes progressively
        # until condition is met, then proceed
        G_remaining.remove_nodes_from(removals)
        if G_remaining.number_of_nodes() > 0:
            components = list(nx.connected_components(G_remaining))
            lcc = max(len(c) for c in components) if components else 0
        else:
            lcc = 0

        if lcc > stop_condition:
            raise ValueError("Initial removal sequence does not satisfy stop_condition")

    # Try reinserting nodes in reverse removal order
    kept_removed = set(removals)

    for node in reversed(removals):
        if node not in kept_removed:
            continue

        # Try adding this node back
        # Add node and its edges to neighbors that are still in the graph
        neighbors_in_graph = [v for v in G.neighbors(node) if v not in kept_removed or v == node]
        if neighbors_in_graph:
            G_remaining.add_node(node)
            for v in neighbors_in_graph:
                if G_remaining.has_node(v):
                    G_remaining.add_edge(node, v)
        else:
            G_remaining.add_node(node)

        # Check if LCC is still within stop_condition
        if G_remaining.number_of_nodes() > 0:
            components = list(nx.connected_components(G_remaining))
            lcc = max(len(c) for c in components) if components else 0
        else:
            lcc = 0

        if lcc <= stop_condition:
            # Reinsertion successful: node can stay
            kept_removed.discard(node)
        else:
            # Reinsertion failed: remove node again
            if G_remaining.has_node(node):
                G_remaining.remove_node(node)

    # Return the kept removals in original order
    final_removals = [node for node in removals if node in kept_removed]
    return final_removals


def dismantle_with_reinsertion(
    G: nx.Graph,
    model,
    features: List[str],
    stop_condition: int,
    device: str = "cpu",
) -> List[int]:
    """
    Full GDM + Reinsertion dismantling.
    1. Get initial static prediction sequence
    2. Run reinsertion to minimize removals
    3. Return optimized sequence
    """
    from network_dismantling.GDM.predictors_nx import gdm_dismantle

    # Step 1: Get initial GDM sequence
    initial_removals = gdm_dismantle(G, model, features, stop_condition, device=device)

    # Step 2: Optimize with reinsertion
    optimized = reinsert_nodes(G, initial_removals, stop_condition)

    return optimized
