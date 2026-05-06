import torch, numpy as np
import sys, os
import argparse

import matplotlib.pyplot as plt
from tqdm import tqdm
from train import train
from network import MODEL_REGISTRY
from baseline import compute_empirical_distribution
from torch_geometric.loader import DataLoader
from utils import load_dataset, load_model, save_model, DatasetInfos
from network import MODEL_REGISTRY
from ddpm import DDPM


current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir)
sys.path.append(project_root_dir)


def create_model(train_set, args):
    """
    Bootstrap definition dynamically registering requested networks wrapping around Graph DDPM.

    Args:
        train_set: The training dataset.
        args (argparse.Namespace): Execution command runtime hyperparameter arguments list.
    Returns:
        DDPM: Model instance equipped explicitly per specific layer requests.
    """
    x_sample = train_set[0] if not isinstance(train_set[0], (list, tuple)) else train_set[0][0]
    node_feature_dim = x_sample.x.shape[1]
    edge_feature_dim = x_sample.edge_attr.shape[1] + 1
    global_feature_dim = 1

    node_counts = torch.zeros(node_feature_dim)
    edge_counts = torch.zeros(edge_feature_dim)

    for data in train_set:
        data = data if not isinstance(data, (list, tuple)) else data[0]
        n = data.num_nodes
        if n > 0:
            node_counts += data.x.sum(dim=0)
            possible_edges = n * (n - 1)
            edge_counts[1:] += data.edge_attr.sum(dim=0)
            edge_counts[0] += (possible_edges - data.edge_attr.shape[0])

    node_dist = node_counts / node_counts.sum()
    edge_dist = edge_counts / edge_counts.sum()

    dataset_infos = DatasetInfos(
        node_feature_dim, edge_feature_dim, global_feature_dim,
        node_dist, edge_dist
    )

    network_type = args.network_type
    if network_type == 'GraphTransformer':
        hidden_mlp_dims = {'X': args.num_hidden * 2, 'E': args.num_hidden, 'y': args.num_hidden}
        hidden_dims = {'dx': args.num_hidden, 'de': args.num_hidden // 2, 'dy': args.num_hidden // 2, 'n_head': 4, 'dim_ffX': args.num_hidden, 'dim_ffE': args.num_hidden}
        network = MODEL_REGISTRY[network_type](
            n_layers=args.n_layers,
            input_dims=dataset_infos.input_dims,
            output_dims=dataset_infos.output_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            act_fn_in=torch.nn.ReLU(),
            act_fn_out=torch.nn.ReLU()
        )
    else:
        raise ValueError(f"Unsupported network type: {network_type}. Check registration in network.py")

    T = args.T
    model = DDPM(network, dataset_infos=dataset_infos, device=args.device, T=T).to(args.device)
    return model


parser = argparse.ArgumentParser()
parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'hyperparameter_search','baseline','stats'], help='what to do when running the script (default: %(default)s)')

parser.add_argument('--model-path', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')

parser.add_argument('--sample-view', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
parser.add_argument('--num-sample', type=int, default=4, metavar='N', help='number of samples to perform (used for sampling and stats) (default: %(default)s)')

parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training and sampling (default: %(default)s)')

parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')

parser.add_argument('--num-hidden', type=int, default=128, help='Number of hidden units (default: %(default)s)')
parser.add_argument('--n-layers', type=int, default=4, help='Number of transformer layers (default: %(default)s)')
parser.add_argument('--network-type', type=str, default='GraphTransformer', choices=MODEL_REGISTRY.keys(), help='Choose the network type (default: %(default)s)')
parser.add_argument('--T', type=int, default=128, metavar='V', help='Number of steps in the diffusion process (default: %(default)s)')

parser.add_argument('--hparams-search-file', type=str, default='params.json', help='file containing all the hyperparameters combinations to search over (default: %(default)s)')

args = parser.parse_args()
print('# Options')
for key, value in sorted(vars(args).items()):
    print(key, '=', value)
print()

train_set, _, _ = load_dataset()

all_n, r_map = compute_empirical_distribution(train_set)

print('Mean nodes : ', np.mean(all_n))
print('Std nodes : ', np.std(all_n))

total_samples = args.num_sample
batch_size = args.batch_size
num_rounds = (total_samples + batch_size - 1) // batch_size


if args.mode == 'train':
    if args.model_path and os.path.exists(args.model_path): # Check if model_path exists
        model = load_model(args.model_path, args.device)
    else :
        model = create_model(train_set, args).to(args.device)
        


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    train(model, optimizer, train_loader, args.epochs, args.device, plot_loss=True, scheduler=scheduler)

    save_model(model, args.model_path)

elif args.mode == 'sample':
    if args.model_path and os.path.exists(args.model_path):
        model = load_model(args.model_path, args.device)
    else :
        raise ValueError(f"No model provided")
    model.eval()
    
    all_generated_adj_matrices = []

    print(f"Sampling {total_samples} graphs in batches of {batch_size}...")
    for i in tqdm(range(num_rounds), desc="Generating samples"):
        current_batch_size = min(batch_size, total_samples - i * batch_size)
        if current_batch_size == 0:
            break

        n_sampled_batch = np.random.choice(all_n, size=current_batch_size, replace=True)
        n_nodes_tensor = torch.tensor(n_sampled_batch, device=args.device)

        X_batch, E_batch, y_batch = model.sample(n_nodes=n_nodes_tensor)

        # Conserver les valeurs catégorielles pour afficher les différents types de liaisons
        adj_matrix_batch = E_batch.int()
        for j in range(current_batch_size):
            actual_adj = adj_matrix_batch[j, :n_nodes_tensor[j], :n_nodes_tensor[j]]
            all_generated_adj_matrices.append(actual_adj)

    if args.sample_view:
        from utils import plot_view
        plot_view(train_set,all_generated_adj_matrices,args.sample_view)


elif args.mode == 'baseline':
    from baseline import generate_ER_baseline
    adj_matrices = generate_ER_baseline(all_n, r_map, num_graphs=args.num_sample)

    if args.sample_view:
        from utils import plot_view
        plot_view(train_set,adj_matrices,args.sample_view)
    

elif args.mode == 'stats':
    from baseline import generate_ER_baseline
    from evaluate import compare_graphs_generation, plot_statistics

    print("Generating baseline graphs...")
    baseline_adj_matrices = generate_ER_baseline(all_n, r_map, num_graphs=args.num_sample)

    print(f"Generating {total_samples} model graphs in batches of {batch_size}...")
    if args.model_path and os.path.exists(args.model_path):
        model = load_model(args.model_path, args.device)
    else :
        raise ValueError(f"No model provided")
    model.eval()
    
    generated_adj_matrices = []
    for i in tqdm(range(num_rounds), desc="Generating model samples"):
        current_batch_size = min(batch_size, total_samples - i * batch_size)
        if current_batch_size == 0:
            break

        n_sampled_batch = np.random.choice(all_n, size=current_batch_size, replace=True)
        n_nodes_tensor = torch.tensor(n_sampled_batch, device=args.device)
        _, E_batch, _ = model.sample(n_nodes=n_nodes_tensor)

        # Conserver les valeurs catégorielles
        adj_matrix_batch = E_batch.int()
        for j in range(current_batch_size):
            actual_adj = adj_matrix_batch[j, :n_nodes_tensor[j], :n_nodes_tensor[j]]
            generated_adj_matrices.append(actual_adj)

    print("Computing stats...")
    compare_graphs_generation(generated_adj_matrices, baseline_adj_matrices, train_set)
    print('Plot statistics')
    plot_statistics(baseline_adj_matrices, generated_adj_matrices,train_set)

elif args.mode == 'hyperparameter_search':
    import itertools,json
    from evaluate import hashes # Import the hashing function
    
    with open(args.hparams_search_file, 'r') as f:
        hparam_grid = json.load(f)

    T_values = hparam_grid.get("T", [args.T])
    num_hidden_values = hparam_grid.get("num_hidden", [args.num_hidden])
    n_layers_values = hparam_grid.get("n_layers", [args.n_layers])

    # We will now optimize for a generation quality metric instead of loss
    best_quality_score = -1.0 
    best_hparams = {}
    all_loss_curves = []

    # Pre-compute train set hashes for novelty calculation
    train_wl_hashes = hashes(train_set, graph_type='geometric')
    train_set_hashes = set(train_wl_hashes)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    print("Starting hyperparameter search...")
    for T, num_hidden, n_layers in itertools.product(T_values, num_hidden_values, n_layers_values):
        print(f"\n--- Testing HParams: T={T}, num_hidden={num_hidden}, n_layers={n_layers} ---")

        current_args = argparse.Namespace(**vars(args))
        current_args.T = T
        current_args.num_hidden = num_hidden
        current_args.n_layers = n_layers

        model = create_model(train_set, current_args)
        model.to(current_args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=current_args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        _,_  = train(model, optimizer, train_loader, current_args.epochs, current_args.device, scheduler)
        
        # --- Evaluation Step ---
        print("  > Generating samples for evaluation...")
        model.eval() # Set model to evaluation mode
        num_eval_samples = 32 # Use a smaller number for faster search
        
        n_sampled_batch = np.random.choice(all_n, size=num_eval_samples, replace=True)
        n_nodes_tensor = torch.tensor(n_sampled_batch, device=current_args.device)

        _, E_batch, _ = model.sample(n_nodes=n_nodes_tensor)

        # Conserver les valeurs catégorielles
        adj_matrix_batch = E_batch.int()
        generated_adj_matrices = []
        for j in range(num_eval_samples):
            actual_adj = adj_matrix_batch[j, :n_nodes_tensor[j], :n_nodes_tensor[j]]
            generated_adj_matrices.append(actual_adj)

        # Calculate novelty and uniqueness
        gen_wl_hashes = hashes(generated_adj_matrices, graph_type='adjacency_matrix')
        gen_set = set(gen_wl_hashes)
        gen_novelty = sum(h not in train_set_hashes for h in gen_wl_hashes) / len(gen_wl_hashes) if len(gen_wl_hashes) > 0 else 0.0
        gen_uniqueness = len(gen_set) / len(gen_wl_hashes) if len(gen_wl_hashes) > 0 else 0.0
        
        # Use a combined score for optimization, e.g., the product of novelty and uniqueness
        quality_score = gen_novelty * gen_uniqueness

        print(f"HParams: T={T}, num_hidden={num_hidden}, n_layers={n_layers} -> Quality Score (Novelty*Uniqueness): {quality_score:.4f}")

        if quality_score > best_quality_score:
            best_quality_score = quality_score
            best_hparams = {'T': T, 'num_hidden': num_hidden, 'n_layers': n_layers}
            # Optionally save the best model
            save_model(model, "best_hparam_model.pt")

    print("\n--- Hyperparameter Search Complete ---")
    print(f"Best Quality Score: {best_quality_score:.4f}")
    print(f"Best Hyperparameters: {best_hparams}")
