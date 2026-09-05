"""
Vertex Entanglement – pure networkx/numpy implementation.

Reference:
> Huang, Y., Wang, H., Ren, X.-L., Lü, L. Identifying key players in complex networks via network entanglement. Communications Physics 7, 19 (2024). DOI: 10.1038/s42005-023-01483-8
"""
"""
Pure networkx/numpy implementation of Vertex Entanglement.
Extracted from VE.py to avoid graph_tool dependencies.
"""
import numpy as np
import networkx as nx
from tqdm.auto import tqdm


def new_generate_beta(lambda2):
    gap = 100
    a = list(np.linspace(0, 5 / lambda2, gap))
    b = list(np.linspace(a[-1], 10 / lambda2, gap))
    return a[0:-1] + b


def VertexEnt_nx(G, belta=None, perturb_strategy='default', printLog=False):
    """
    Approximate vertex entanglement using networkx and numpy.
    :param G: networkx.Graph, nodes should be numbered 0..n-1
    :return: np.ndarray of VE values for each node
    """
    A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).todense()
    assert 0 in G, "Node 0 should be in the input graph!"
    assert np.allclose(A, A.T), "adjacency matrix should be symmetric"
    
    L = np.diag(np.array(sum(A)).flatten()) - A
    N = G.number_of_nodes()
    
    eigValue, eigVector = np.linalg.eigh(L)
    eigValue = eigValue.real
    eigVector = eigVector.real
    
    if printLog:
        print(eigValue)
    
    if belta is None:
        num_components = nx.number_connected_components(G)
        belta = new_generate_beta(eigValue[num_components])
    
    S = np.zeros(len(belta))
    for i in range(0, len(belta)):
        b = belta[i]
        Z_ = np.sum(np.exp(eigValue * -b))
        t = -b * eigValue
        S[i] = -sum(np.exp(t) * (t / np.log(2) - np.log2(Z_)) / Z_)
    
    if printLog:
        print(S)
    
    lambda_ral = np.zeros((N, N))
    if perturb_strategy == 'default':
        for v_id in tqdm(range(0, N), desc="Computing eigenValues", unit="node"):
            neibour = list(G.neighbors(v_id))
            kx = G.degree(v_id)
            A_loc = A[neibour][:, neibour]
            N_loc = kx + np.sum(A_loc) / 2
            weight = 2 * N_loc / kx / (kx + 1) if kx > 0 else 1
            if weight == 1 or kx == 0:
                lambda_ral[v_id] = eigValue
            else:
                neibour.append(v_id)
                neibour = sorted(neibour)
                dA = weight - A[neibour][:, neibour]
                dA = dA - np.diag([weight] * (kx + 1))
                dL = np.diag(np.array(sum(dA)).flatten()) - dA
                for j in range(0, N):
                    t__ = eigVector[neibour, j].T @ dL @ eigVector[neibour, j]
                    if isinstance(t__, float):
                        lambda_ral[v_id, j] = eigValue[j] + t__
                    else:
                        lambda_ral[v_id, j] = eigValue[j] + t__[0, 0]
    elif perturb_strategy == 'remove':
        for v_id in tqdm(range(0, N), desc="Computing eigenValues for removed networks", unit="node", position=0, leave=True):
            neibour = list(G.neighbors(v_id))
            pt_A = np.array(A, copy=True)
            pt_A[v_id, :] = 0
            pt_A[:, v_id] = 0
            dA = pt_A - A
            dA = dA[neibour][:, neibour]
            dL = np.diag(np.array(sum(dA)).flatten()) - dA
            for j in range(0, N):
                lambda_ral[v_id, j] = eigValue[j] + (eigVector[neibour, j].T @ dL @ eigVector[neibour, j])[0, 0]
    
    E = np.zeros((len(belta), N))
    for x in tqdm(range(0, N), desc="Searching minium entanglement", unit="node"):
        xl_ = lambda_ral[x, :]
        for i in range(0, len(belta)):
            b = belta[i]
            Z_ = np.sum(np.exp(-b * xl_))
            t = -b * xl_
            E[i, x] = -sum(np.exp(t) * (t / np.log(2) - np.log2(Z_)) / Z_) - S[i]
    
    VE = np.min(E, axis=0)
    return VE
