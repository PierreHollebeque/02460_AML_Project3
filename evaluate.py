from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split
import torch
import os, sys
import networkx as nx
from torch_geometric.utils import to_networkx

from utils import load_dataset
from baseline import compute_empirical_distribution, generate_ER_baseline


if __name__ == '__main__':
    train_set, _, _ = load_dataset()

    all_n, r_map = compute_empirical_distribution(train_set)
    print("Baselne Training Done")
    A = generate_ER_baseline(all_n, r_map, num_graphs=1000)
    
    train_hashes = []
    for data in train_set:  
        h = to_networkx(data, to_undirected=True)
        train_hashes.append(h)
        
    gen_hashes = []
    for data in A:  
        h = nx.from_numpy_array(data.numpy())
        gen_hashes.append(h)
    
    gen_wl_hashes = []
    train_wl_hashes = []
    for h in train_hashes:
       h = nx.weisfeiler_lehman_graph_hash(h, iterations=10)
       train_wl_hashes.append(h)
       
    for h in gen_hashes:
       h = nx.weisfeiler_lehman_graph_hash(h, iterations=10)
       gen_wl_hashes.append(h)
       
    gen_set = set(gen_wl_hashes)
    train_set_hashes = set(train_wl_hashes)

    novelty    = sum(h not in train_set_hashes for h in gen_wl_hashes) / len(gen_wl_hashes)
    uniqueness = len(gen_set) / len(gen_wl_hashes)
    novel_and_unique = len(gen_set - train_set_hashes) / len(gen_wl_hashes)
    
    print("Novelty = ",novelty)
    print("Uniqueness =",uniqueness)
    print("Novelty & Uniqueness =",novel_and_unique)