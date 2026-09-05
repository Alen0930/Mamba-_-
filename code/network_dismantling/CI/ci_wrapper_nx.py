"""
Collective Influence (CI) dismantler – networkx wrapper.

Reference:
> Morone, F., Min, B., Bo, L., Mari, R., Makse, H.A. Collective Influence Algorithm to find influencers via optimal percolation in massively large social media. Scientific Reports 6, 30062 (2016). DOI: 10.1038/srep30062
"""
"""
Networkx wrapper for CI (Collective Influence) dismantler.
Calls the precompiled CI.exe on Windows.
"""
import os
import tempfile
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np

CI_EXE = Path(__file__).parent / "CI.exe"


def ci_dismantle_nx(G: nx.Graph, l: int = 2, stop_condition: int = 1) -> List[int]:
    """
    Run CI dismantler on a networkx graph.
    Returns the dismantling sequence (node indices aligned with input G).
    """
    if not CI_EXE.exists():
        raise FileNotFoundError(f"CI.exe not found at {CI_EXE}. Please compile it first.")
    
    # Ensure nodes are 0..n-1
    mapping = {node: i for i, node in enumerate(G.nodes())}
    reverse_mapping = {i: node for node, i in mapping.items()}
    G = nx.relabel_nodes(G, mapping)
    
    # Write adjacency list (1-indexed for CI)
    network_fd, network_path = tempfile.mkstemp(suffix=".txt")
    output_fd, output_path = tempfile.mkstemp(suffix=".txt")
    
    try:
        with open(network_fd, "w") as f:
            for node in sorted(G.nodes()):
                neighbors = sorted(G.neighbors(node))
                # CI expects 1-indexed node IDs
                line = f"{node + 1}"
                for nb in neighbors:
                    line += f" {nb + 1}"
                f.write(line + "\n")
        
        import subprocess
        result = subprocess.run(
            [str(CI_EXE), network_path, str(l), str(stop_condition), output_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"CI.exe failed: {result.stderr}")
        
        # Read output: each line is "step node_id"
        nodes = []
        with open(output_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    node = int(parts[1]) - 1  # convert back to 0-indexed
                    nodes.append(node)
        
        # Map back to original IDs
        return [reverse_mapping[n] for n in nodes]
    finally:
        try:
            os.close(network_fd)
            os.remove(network_path)
        except Exception:
            pass
        try:
            os.close(output_fd)
            os.remove(output_path)
        except Exception:
            pass
