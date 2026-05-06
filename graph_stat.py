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
