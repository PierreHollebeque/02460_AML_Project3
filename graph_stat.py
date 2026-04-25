"""
Utility functions for statistics :
- Node degree
- Clustering coefficient
- Eigenvector centrality
"""

import torch
import numpy as np
from torch_geometric.utils import to_dense_adj

def get_adjacency_matrix(graph):
    return to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes)[0]

def node_degree(graph):
    A = get_adjacency_matrix(graph)
    return torch.sum(A, dim=1)

def clustering_coefficient(graph):
    A = get_adjacency_matrix(graph)

    triangles = torch.diag(torch.matrix_power(A, 3))

    # node degree
    degrees = torch.sum(A, dim=1)
    possible_triangles = degrees * (degrees - 1)
    
    cc = torch.zeros_like(degrees)
    mask = possible_triangles > 0
    cc[mask] = triangles[mask] / possible_triangles[mask]
    
    return cc.numpy()

def eigenvector_centrality(graph):
    A = get_adjacency_matrix(graph)
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    return eigenvectors[:, np.argmax(eigenvalues)].numpy()