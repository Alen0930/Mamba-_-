import networkx as nx
import matplotlib.pyplot as plt
from network_dismantling.unified_interface import dismantle

g = nx.barabasi_albert_graph(1000, 4, seed=42)
n = g.number_of_nodes()
ms = ['degree', 'pagerank', 'random', 'CoreHD']

plt.figure(figsize=(8, 5))
for m in ms:
    seq = dismantle(g, method=m)
    gt = g.copy()
    x = [0]
    y = [1.0]
    
    for i, node in enumerate(seq, 1):
        gt.remove_node(node)
        if i % 10 == 0:
            # 空图直接结束循环
            if gt.number_of_nodes() == 0:
                x.append(i / n)
                y.append(0.0)
                break
            lcc = max(nx.connected_components(gt), key=len)
            x.append(i / n)
            y.append(len(lcc) / n)
    
    plt.plot(x, y, label=m)

plt.xlabel('Removed nodes fraction')
plt.ylabel('Largest component fraction')
plt.title('Dismantling robustness comparison')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
