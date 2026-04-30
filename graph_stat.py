"""
Utility functions for statistics :
- Node degree
- Clustering coefficient
- Eigenvector centrality
"""

import torch
import numpy as np
from torch_geometric.utils import to_dense_adj


def node_degree(A):
    return torch.sum(A, dim=1)

def clustering_coefficient(A):

    triangles = torch.diag(torch.matrix_power(A, 3))

    # node degree
    degrees = torch.sum(A, dim=1)
    possible_triangles = degrees * (degrees - 1)
    
    cc = torch.zeros_like(degrees)
    mask = possible_triangles > 0
    cc[mask] = triangles[mask] / possible_triangles[mask]
    
    return cc.numpy()

def eigenvector_centrality(A):
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    return eigenvectors[:, np.argmax(eigenvalues)].abs().numpy()


if __name__ == '__main__':
    # --- Setup for testing ---
    import networkx as nx
    from torch_geometric.utils import erdos_renyi_graph
    
    N_test = 10
    prob_test = 0.4
    
    # Generate an ER graph and convert it directly to an adjacency matrix
    edge_index = erdos_renyi_graph(N_test, prob_test, directed=False)
    test_adj = to_dense_adj(edge_index, max_num_nodes=N_test)[0]
    
    # --- 1. Reference calculations with NetworkX ---
    # Convert our matrix to NetworkX graph for validation
    G_nx = nx.from_numpy_array(test_adj.numpy())
    
    ref_degree = np.array([d for _, d in G_nx.degree()])
    ref_cc = np.array(list(nx.clustering(G_nx).values()))
    ref_ev = np.array(list(nx.eigenvector_centrality(G_nx, max_iter=1000).values()))
    ref_ev = np.abs(ref_ev) 

    # --- 2. Validation Tests ---
    print("--- Running Validation Tests with Adjacency Matrices ---")
    
    # Test Degree
    my_degree = node_degree(test_adj).numpy()
    deg_diff = np.linalg.norm(my_degree - ref_degree)
    print(f"Degree Test: {'PASSED' if deg_diff < 1e-6 else 'FAILED'} (Diff: {deg_diff:.2e})")
    
    # Test Clustering Coefficient
    my_cc = clustering_coefficient(test_adj)
    cc_diff = np.linalg.norm(my_cc - ref_cc)
    print(f"Clustering Test: {'PASSED' if cc_diff < 1e-6 else 'FAILED'} (Diff: {cc_diff:.2e})")
    
    # Test Eigenvector Centrality
    my_ev = eigenvector_centrality(test_adj)
    # Normalize to compare shapes (unit length)
    my_ev_norm = my_ev / np.linalg.norm(my_ev)
    ref_ev_norm = ref_ev / np.linalg.norm(ref_ev)
    ev_diff = np.linalg.norm(my_ev_norm - ref_ev_norm)
    print(f"Eigenvector Test: {'PASSED' if ev_diff < 1e-6 else 'FAILED'} (Diff: {ev_diff:.2e})")

    # --- 3. Visual check ---
    print("\nSample values (Node 0):")
    print(f"  Degree:   Manual={my_degree[0]}, NX={ref_degree[0]}")
    print(f"  Clust Co: Manual={my_cc[0]:.4f}, NX={ref_cc[0]:.4f}")
    print(f"  Eigenvec: Manual={my_ev_norm[0]:.4f}, NX={ref_ev_norm[0]:.4f}")