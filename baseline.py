import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, to_dense_adj

def compute_empirical_distribution(train_dataset):
    '''
    Analyzes the training set to build the empirical distribution of N
    and the mapping of link probabilities r for each N.
    '''
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
    '''
    Generates synthetic graphs following the 3-step baseline instructions.
    '''
    generated_adj_matrices = []
    
    for _ in range(num_graphs):
        # Step 1: Sample N from the empirical distribution
        n_sampled = int(np.random.choice(all_n))
        
        # Step 2: Get the link probability r for this specific N (global_r as a fallack)
        r_sampled = r_map.get(n_sampled)
        
        # Step 3: Sample a random graph using the Erdös-Rényi model
        # directed=False ensures we get a symmetric edge_index (undirected)
        edge_index = erdos_renyi_graph(n_sampled, r_sampled, directed=False)
        
        adj = to_dense_adj(edge_index, max_num_nodes=n_sampled)[0]
        generated_adj_matrices.append(adj)
        
    return generated_adj_matrices

if __name__ == '__main__':
    from utils import load_dataset
    train_set, val_set, test_set = load_dataset()

    # 1. Analyze training data
    all_n, r_map = compute_empirical_distribution(train_set)

    print('MeanN : ', np.mean(all_n))
    # 2. Generate baseline graphs
    adj_matrices = generate_ER_baseline(all_n, r_map, num_graphs=10)

    print(f"Generated {len(adj_matrices)} graphs.")
    print(f"Shape of the first matrix : {adj_matrices[0].shape}")
    print("Partial content :\n", adj_matrices[0][:5, :5]) # Plot corner 5x5