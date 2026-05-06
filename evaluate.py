from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split
import torch
import os, sys
import networkx as nx
from torch_geometric.utils import to_networkx

from utils import load_dataset
from baseline import compute_empirical_distribution, generate_ER_baseline

def hashes(graphs, graph_type='adjacency_matrix'):
    """
    Computes Weisfeiler-Lehman hashes for a list of graphs.

    Args:
        graphs (list): A list of graph representations.
                       If graph_type is 'geometric', elements are torch_geometric.data.Data objects.
                       If graph_type is 'adjacency_matrix', elements are torch.Tensor adjacency matrices.
        graph_type (str): Specifies the type of graph representation in the list.
                          Can be 'geometric' or 'adjacency_matrix'.
    Returns:
        list: A list of WL hashes (strings).
    """
    nx_graphs = []
    if graph_type == 'geometric':
        for data in graphs:
            # Ensure data is on CPU before converting to NetworkX
            h = to_networkx(data.cpu(), to_undirected=True)
            nx_graphs.append(h)
    elif graph_type == 'adjacency_matrix':
        for data in graphs:
            # Ensure tensor is on CPU and convert to numpy array
            h = nx.from_numpy_array(data.cpu().numpy())
            nx_graphs.append(h)
    else:
        raise ValueError(f"Unsupported graph_type: {graph_type}")

    wl_hashes = []
    for g in nx_graphs:
        try:
            h = nx.weisfeiler_lehman_graph_hash(g, iterations=10)
            wl_hashes.append(h)
        except Exception as e:
            print(f"Warning: Could not compute WL hash for a graph. Error: {e}")
    return wl_hashes


def compare_graphs_generation(generated_graphs, baseline_graphs, train_set):
    """
    Compares generated and baseline graphs against the training set using WL hashes.

    Args:
        generated_graphs (list): List of generated adjacency matrices (torch.Tensor).
        baseline_graphs (list): List of baseline adjacency matrices (torch.Tensor).
        train_set (list): List of training graphs (torch_geometric.data.Data).
    """
    # Compute hashes for the training set
    train_wl_hashes = hashes(train_set, graph_type='geometric')
    train_set_hashes = set(train_wl_hashes)

    # Compute metrics for Generated Graphs
    gen_wl_hashes = hashes(generated_graphs, graph_type='adjacency_matrix')
    gen_set = set(gen_wl_hashes)
    gen_novelty = sum(h not in train_set_hashes for h in gen_wl_hashes) / len(gen_wl_hashes) if len(gen_wl_hashes) > 0 else 0.0
    gen_uniqueness = len(gen_set) / len(gen_wl_hashes) if len(gen_wl_hashes) > 0 else 0.0
    gen_novel_and_unique = len(gen_set - train_set_hashes) / len(gen_wl_hashes) if len(gen_wl_hashes) > 0 else 0.0

    # Compute metrics for Baseline Graphs
    baseline_wl_hashes = hashes(baseline_graphs, graph_type='adjacency_matrix')
    baseline_set = set(baseline_wl_hashes)
    baseline_novelty = sum(h not in train_set_hashes for h in baseline_wl_hashes) / len(baseline_wl_hashes) if len(baseline_wl_hashes) > 0 else 0.0
    baseline_uniqueness = len(baseline_set) / len(baseline_wl_hashes) / len(baseline_wl_hashes) if len(baseline_wl_hashes) > 0 else 0.0
    baseline_novel_and_unique = len(baseline_set - train_set_hashes) / len(baseline_wl_hashes) if len(baseline_wl_hashes) > 0 else 0.0

    # Structured print
    print("\n--- Graph Generation Comparison ---")
    print(f"{'Metric':<25} | {'Generated Model':<20} | {'Baseline (ER)':<20}")
    print("-" * 70)
    print(f"{'Novelty':<25} | {gen_novelty:<20.4f} | {baseline_novelty:<20.4f}")
    print(f"{'Uniqueness':<25} | {gen_uniqueness:<20.4f} | {baseline_uniqueness:<20.4f}")
    print(f"{'Novelty & Uniqueness':<25} | {gen_novel_and_unique:<20.4f} | {baseline_novel_and_unique:<20.4f}")


if __name__ == '__main__':
    train_set, _, _ = load_dataset()

    all_n, r_map = compute_empirical_distribution(train_set)
    print("Baseline Training Done")
    num_graphs_to_generate = 100

    # Generate baseline graphs
    baseline_adj_matrices = generate_ER_baseline(all_n, r_map, num_graphs=num_graphs_to_generate)
    print(f"Generated {len(baseline_adj_matrices)} baseline graphs.")

    # Placeholder for generated model graphs
    generated_adj_matrices = generate_ER_baseline(all_n, r_map, num_graphs=num_graphs_to_generate)
    print(f"Generated {len(generated_adj_matrices)} dummy model graphs.")

    # Perform the comparison
    compare_graphs_generation(generated_adj_matrices, baseline_adj_matrices, train_set)