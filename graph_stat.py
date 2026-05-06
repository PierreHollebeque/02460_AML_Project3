"""
Utility functions for deriving key graph metrics:
- Node degree
- Clustering coefficient
- Eigenvector centrality
"""

import torch
import numpy as np
from torch_geometric.utils import to_dense_adj


def node_degree(A):
    """
    Derive spatial node degrees sequentially out of an input structure map.

    Args:
        A (torch.Tensor): Valid adjacency mapping array encoding connection traces.
    Returns:
        torch.Tensor: Degree quantities extracted accurately per node item.
    """
    return torch.sum(A, dim=1)

def clustering_coefficient(A):
    """
    Calculate relative graph topological clustering coefficient proportions reliably.

    Args:
        A (torch.Tensor): Defined symmetrical input association matrix.
    Returns:
        numpy.ndarray: Assessed fractional arrays evaluating cluster density profiles.
    """
    triangles = torch.diag(torch.matrix_power(A, 3))

    degrees = torch.sum(A, dim=1)
    possible_triangles = degrees * (degrees - 1)
    
    cc = torch.zeros_like(degrees)
    mask = possible_triangles > 0
    cc[mask] = triangles[mask] / possible_triangles[mask]
    
    return cc.numpy()

def eigenvector_centrality(A):
    """
    Ascertain prominent principal graph axis structural centralization patterns.

    Args:
        A (torch.Tensor): Adjacency relationship index.
    Returns:
        numpy.ndarray: Absolute magnitude centralizations vector layout aligned strictly.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    return eigenvectors[:, np.argmax(eigenvalues)].abs().numpy()
