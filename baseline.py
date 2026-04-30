import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph

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
    generated_graphs = []
    
    for _ in range(num_graphs):
        # Step 1: Sample N from the empirical distribution
        n_sampled = int(np.random.choice(all_n))
        
        # Step 2: Get the link probability r for this specific N (global_r as a fallack)
        r_sampled = r_map.get(n_sampled)
        
        # Step 3: Sample a random graph using the Erdös-Rényi model
        # directed=False ensures we get a symmetric edge_index (undirected)
        edge_index = erdos_renyi_graph(n_sampled, r_sampled, directed=False)
        
        # Create node features -> all 1
        x = torch.ones((n_sampled, num_features)) 
        
        # Build the PyG Data object
        new_graph = Data(x=x, edge_index=edge_index)
        generated_graphs.append(new_graph)
        
    return generated_graphs

if __name__ == '__main__':
    from torch_geometric.datasets import TUDataset
    from torch.utils.data import random_split

    dataset = TUDataset(root='./data/', name='MUTAG')
    
    # Split the dataset
    rng = torch.Generator().manual_seed(0)
    train_set, val_set, test_set = random_split(dataset, (100, 44, 44), generator=rng)

    # 1. Analyze training data
    all_n, r_map = compute_empirical_distribution(train_set)

    print('MeanN : ', np.mean(all_n))
    # 2. Generate baseline graphs
    synthetic_baseline = generate_ER_baseline(all_n, r_map, num_graphs=10)

    print(f"Generated {len(synthetic_baseline)} graphs.")
    print(f"Example Graph 0: {synthetic_baseline[0]}")