from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split
import torch, numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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
    A_float = A.to(torch.float32)
    triangles = torch.diag(torch.matrix_power(A_float, 3))

    degrees = torch.sum(A_float, dim=1)
    possible_triangles = degrees * (degrees - 1)
    
    cc = torch.zeros_like(degrees, dtype=torch.float)
    mask = possible_triangles > 0
    cc[mask] = triangles[mask] / possible_triangles[mask]
    
    return cc.cpu().numpy()

def eigenvector_centrality(A):
    """
    Ascertain prominent principal graph axis structural centralization patterns.

    Args:
        A (torch.Tensor): Adjacency relationship index.
    Returns:
        numpy.ndarray: Absolute magnitude centralizations vector layout aligned strictly.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(A.to(torch.float32))
    return eigenvectors[:, torch.argmax(eigenvalues)].abs().cpu().numpy()

def hashes(graphs, graph_type='adjacency_matrix'):
    import networkx as nx
    from torch_geometric.utils import to_networkx

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


def plot_statistics(baseline_adj_matrices, generated_adj_matrices,train_set):
    # --- 2. Collect Statistics ---
    train_degrees, train_ccs, train_ecs = [], [], []
    baseline_degrees, baseline_ccs, baseline_ecs = [], [], []
    gen_degrees, gen_ccs, gen_ecs = [], [], []

    print("Calculating statistics for training set...")
    for data in tqdm(train_set, desc="Train set stats"):
        if data.num_nodes == 0: continue # Skip empty graphs
        # Convert to binary adjacency matrix
        # to_dense_adj returns (batch_size, max_num_nodes, max_num_nodes, num_edge_features) if edge_attr is present
        # or (batch_size, max_num_nodes, max_num_nodes) if edge_attr is None (binary)
        # For MUTAG, edge_attr is present and one-hot encoded.
        adj_onehot = to_dense_adj(edge_index=data.edge_index, edge_attr=data.edge_attr, max_num_nodes=data.num_nodes)[0]
        is_edge = adj_onehot.sum(dim=-1) > 0 # Sum over one-hot dim to find if any edge type is present
        adj_binary = is_edge.int()

        if adj_binary.shape[0] > 0: # Ensure there are nodes
            train_degrees.append(node_degree(adj_binary).cpu().numpy())
            train_ccs.append(clustering_coefficient(adj_binary))
            train_ecs.append(eigenvector_centrality(adj_binary))

    print("Calculating statistics for baseline graphs...")
    for adj in tqdm(baseline_adj_matrices, desc="Baseline stats"):
        if adj.shape[0] == 0: continue # Skip empty graphs
        baseline_degrees.append(node_degree(adj).cpu().numpy())
        baseline_ccs.append(clustering_coefficient(adj))
        baseline_ecs.append(eigenvector_centrality(adj))

    print("Calculating statistics for generated graphs...")
    for adj in tqdm(generated_adj_matrices, desc="Generated stats"):
        if adj.shape[0] == 0: continue # Skip empty graphs
        gen_degrees.append(node_degree(adj).cpu().numpy())
        gen_ccs.append(clustering_coefficient(adj))
        gen_ecs.append(eigenvector_centrality(adj))

    # Flatten lists of arrays into single 1D arrays
    train_degrees = np.concatenate(train_degrees) if train_degrees else np.array([])
    train_ccs = np.concatenate(train_ccs) if train_ccs else np.array([])
    train_ecs = np.concatenate(train_ecs) if train_ecs else np.array([])

    baseline_degrees = np.concatenate(baseline_degrees) if baseline_degrees else np.array([])
    baseline_ccs = np.concatenate(baseline_ccs) if baseline_ccs else np.array([])
    baseline_ecs = np.concatenate(baseline_ecs) if baseline_ecs else np.array([])

    gen_degrees = np.concatenate(gen_degrees) if gen_degrees else np.array([])
    gen_ccs = np.concatenate(gen_ccs) if gen_ccs else np.array([])
    gen_ecs = np.concatenate(gen_ecs) if gen_ecs else np.array([])

    # --- 3. Plotting ---
    fig, axs = plt.subplots(3, 3, figsize=(18, 15))
    
    # Titles for columns
    col_titles = ['Empirical Distribution (Train)', 'Baseline (ER)', 'Generated Model']
    for j, title in enumerate(col_titles):
        axs[0, j].set_title(title, fontsize=14)

    # Row labels (statistics)
    row_labels = ['Node Degree', 'Clustering Coefficient', 'Eigenvector Centrality']

    # Data for plotting
    stats_data = {
        'degrees': [train_degrees, baseline_degrees, gen_degrees],
        'ccs': [train_ccs, baseline_ccs, gen_ccs],
        'ecs': [train_ecs, baseline_ecs, gen_ecs]
    }
    stat_keys = ['degrees', 'ccs', 'ecs']

    for i, stat_key in enumerate(stat_keys):
        all_values_for_stat = np.concatenate([arr for arr in stats_data[stat_key] if arr.size > 0])
        
        # Determine common bins, handling empty or all-NaN cases
        valid_values = all_values_for_stat[~np.isnan(all_values_for_stat)] if all_values_for_stat.size > 0 else np.array([])
        if valid_values.size > 0:
            min_val, max_val = np.min(valid_values), np.max(valid_values)
            if min_val == max_val: bins = [min_val - 0.5, min_val + 0.5]
            else: bins = np.histogram_bin_edges(valid_values, bins='auto')
        else: bins = 10 # Default bins if no valid data

        for j, data_array in enumerate(stats_data[stat_key]):
            data_to_plot = data_array[~np.isnan(data_array)] if data_array.size > 0 else np.array([])
            
            if data_to_plot.size > 0:
                axs[i, j].hist(data_to_plot, bins=bins, edgecolor='black', alpha=0.7)
            else:
                axs[i, j].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=axs[i, j].transAxes, fontsize=12)

            axs[i, j].set_xlabel(row_labels[i])
            axs[i, j].set_ylabel('Frequency')
            
            # Set y-axis label only for the first column
            if j == 0:
                axs[i, j].set_ylabel('Frequency')
            else:
                axs[i, j].set_ylabel('')

    plt.tight_layout()
    plot_filename = 'graph_statistics_comparison.png'
    fig.savefig(plot_filename)
    print(f"Graph statistics comparison plot saved to {plot_filename}")

if __name__ == '__main__':
    from utils import load_dataset
    from baseline import compute_empirical_distribution, generate_ER_baseline

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

    # Plot statistics
    plot_statistics(baseline_adj_matrices, generated_adj_matrices,train_set)