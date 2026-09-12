import networkx as nx
from network_dismantling.unified_interface import dismantle
from evaluate import calc_metrics

g = nx.barabasi_albert_graph(1000, 4, seed=42)
for m in ['degree', 'pagerank', 'random', 'CoreHD']:
    seq = dismantle(g, method=m, seed=42)
    stop_step, fc_value, reach_stop, reach_fc = calc_metrics(g, seq)
    print(f"{m}: stop_step={stop_step} fc={fc_value:.4f} reach_stop={reach_stop} reach_fc={reach_fc}")
