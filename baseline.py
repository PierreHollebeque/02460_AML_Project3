import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, to_dense_adj

def compute_empirical_distribution(train_dataset):
    """
    Analyze the given training set to derive the empirical distribution of node counts (N),
    calculating the link probability ratio mapping assigned specifically for each N.

    Args:
        train_dataset (Dataset): Graph instances for extracting baseline heuristics.
    Returns:
        tuple: A sequence array of sampled graph N frequencies and a respective probability map.
    """
    all_n = []
    densities_per_n = {}

    for data in train_dataset:
        n = data.num_nodes
        e = data.num_edges # 2 * actual edges
        
        # Formula: actual_edges / (N*(N-1)/2)  => num_edges / (N*(N-1))
        if n > 1:
            possible_edges = n * (n - 1)
            r_i = e / possible_edges
        else:
            r_i = 0.0
            
        all_n.append(n)
        
        if n not in densities_per_n:
            densities_per_n[n] = []
        densities_per_n[n].append(r_i)

    # Compute the average r for each specific N
    r_map = {n: np.mean(rs) for n, rs in densities_per_n.items()}

    return all_n, r_map

def generate_ER_baseline(all_n, r_map, num_graphs, num_features=7):
    """
    Generate synthetic Erdös-Rényi (ER) graphs serving as unstructured statistical baselines.

    Args:
        all_n (list): Catalog distribution sequence populated with valid node counts.
        r_map (dict): The edge presence probabilities associated mapped strictly to specific N.
        num_graphs (int): Amount of benchmark outputs required.
        num_features (int): (Optional) Number of dummy features preserved locally.
    Returns:
        list: Reconstructed baseline tensors formatted systematically as dense adjacency matrices.
    """
    generated_adj_matrices = []
    
    for _ in range(num_graphs):
        # Step 1: Sample N from the empirical distribution
        n_sampled = int(np.random.choice(all_n))
        
        # Step 2: Get the link probability r for this specific N (global_r as a fallback)
        r_sampled = r_map.get(n_sampled)
        
        # Step 3: Sample a random graph using the Erdös-Rényi model
        edge_index = erdos_renyi_graph(n_sampled, r_sampled, directed=False)
        
        adj = to_dense_adj(edge_index, max_num_nodes=n_sampled)[0]
        generated_adj_matrices.append(adj)
        
    return generated_adj_matrices
