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


def create_model(x_sample, args):
    """
    Bootstrap definition dynamically registering requested networks wrapping around Graph DDPM.

    Args:
        x_sample (torch_geometric.data.Data): Baseline graph extracting proper layout shapes.
        args (argparse.Namespace): Execution command runtime hyperparameter arguments list.
    Returns:
        DDPM: Model instance equipped explicitly per specific layer requests.
    """
    node_feature_dim = x_sample.x.shape[1]
    edge_feature_dim = x_sample.edge_attr.shape[1] + 1
    global_feature_dim = 1

    dummy_node_dist = torch.ones(node_feature_dim) / node_feature_dim
    dummy_edge_dist = torch.ones(edge_feature_dim) / edge_feature_dim

    dataset_infos = DatasetInfos(
        node_feature_dim, edge_feature_dim, global_feature_dim,
        dummy_node_dist, dummy_edge_dist
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
parser.add_argument('--num-sample', type=int, default=1, metavar='N', help='number of samples to perform (default: %(default)s)')

parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training and sampling (default: %(default)s)')

parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')

parser.add_argument('--num-hidden', type=int, default=128, help='Number of hidden units (default: %(default)s)')
parser.add_argument('--n-layers', type=int, default=4, help='Number of transformer layers (default: %(default)s)')
parser.add_argument('--network-type', type=str, default='GraphTransformer', choices=MODEL_REGISTRY.keys(), help='Choose the network type (default: %(default)s)')
parser.add_argument('--T', type=int, default=100, metavar='V', help='Number of steps in the diffusion process (default: %(default)s)')

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
    if args.model_path :
        model = load_model(args.model_path, args.device)
    else :
        x_sample = train_set[0] if not isinstance(train_set[0], (list, tuple)) else train_set[0][0]
        model = create_model(x_sample, args).to(args.device)
        


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    train(model, optimizer, train_loader, args.epochs, args.device, plot_loss=True, scheduler=scheduler)

    save_model(model, args.model_path)

elif args.mode == 'sample':
    if args.model_path :
        model = load_model(args.model_path, args.device)
    else :
        raise ValueError(f"No model provided")
    
    all_generated_adj_matrices = []

    print(f"Sampling {total_samples} graphs in batches of {batch_size}...")
    for i in tqdm(range(num_rounds), desc="Generating samples"):
        current_batch_size = min(batch_size, total_samples - i * batch_size)
        if current_batch_size == 0:
            break

        n_sampled_batch = np.random.choice(all_n, size=current_batch_size, replace=True)
        n_nodes_tensor = torch.tensor(n_sampled_batch, device=args.device)

        X_batch, E_batch, y_batch = model.sample(n_nodes=n_nodes_tensor)

        adj_matrix_batch = (E_batch > 0).int()
        for j in range(current_batch_size):
            actual_adj = adj_matrix_batch[j, :n_nodes_tensor[j], :n_nodes_tensor[j]]
            all_generated_adj_matrices.append(actual_adj)

    for idx, adj in enumerate(all_generated_adj_matrices):
        print(f"\nGenerated Adjacency Matrix (sample {idx+1}) - shape : {adj.shape}")
        print(adj)

elif args.mode == 'baseline':
    from baseline import generate_ER_baseline
    adj_matrices = generate_ER_baseline(all_n, r_map, num_graphs=args.num_sample)
    for i in range(args.num_sample):  
        print(f"Generated Adjacency Matrix (sample {i}) - shape : {adj_matrices[i].shape}")
        print(adj_matrices[i])

elif args.mode == 'stats':
    from baseline import generate_ER_baseline
    from evaluate import compare_graphs_generation

    print("Generating baseline graphs...")
    baseline_adj_matrices = generate_ER_baseline(all_n, r_map, num_graphs=args.num_sample)

    print(f"Generating {total_samples} model graphs in batches of {batch_size}...")
    if args.model_path :
        model = load_model(args.model_path, args.device)
    else :
        raise ValueError(f"No model provided")
    
    generated_adj_matrices = []
    for i in tqdm(range(num_rounds), desc="Generating model samples"):
        current_batch_size = min(batch_size, total_samples - i * batch_size)
        if current_batch_size == 0:
            break

        n_sampled_batch = np.random.choice(all_n, size=current_batch_size, replace=True)
        n_nodes_tensor = torch.tensor(n_sampled_batch, device=args.device)
        _, E_batch, _ = model.sample(n_nodes=n_nodes_tensor)

        adj_matrix_batch = (E_batch > 0).int()
        for j in range(current_batch_size):
            actual_adj = adj_matrix_batch[j, :n_nodes_tensor[j], :n_nodes_tensor[j]]
            generated_adj_matrices.append(actual_adj)

    print("Computing stats...")
    compare_graphs_generation(generated_adj_matrices, baseline_adj_matrices, train_set)


elif args.mode == 'hyperparameter_search':
    import itertools,json
    
    with open(args.hparams_search_file, 'r') as f:
        hparam_grid = json.load(f)

    T_values = hparam_grid.get("T", [args.T])
    num_hidden_values = hparam_grid.get("num_hidden", [args.num_hidden])
    n_layers_values = hparam_grid.get("n_layers", [args.n_layers])

    best_loss = float('inf')
    best_hparams = {}
    all_loss_curves = []

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    x_sample = train_set[0] if not isinstance(train_set[0], (list, tuple)) else train_set[0][0]

    print("Starting hyperparameter search...")
    for T, num_hidden, n_layers in itertools.product(T_values, num_hidden_values, n_layers_values):
        print(f"\n--- Testing HParams: T={T}, num_hidden={num_hidden}, n_layers={n_layers} ---")

        current_args = argparse.Namespace(**vars(args))
        current_args.T = T
        current_args.num_hidden = num_hidden
        current_args.n_layers = n_layers

        model = create_model(x_sample, current_args)
        model.to(current_args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=current_args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        current_loss_epochs = train(model, optimizer, train_loader, current_args.epochs, current_args.device, scheduler)
        
        if current_loss:
            all_loss_curves.append((current_loss, {'T': T, 'num_hidden': num_hidden, 'n_layers': n_layers}))
            final_loss = current_loss[-1]

        print(f"HParams: T={T}, num_hidden={num_hidden}, n_layers={n_layers} -> Final Loss: {final_loss:.4f}")

        if final_loss < best_loss:
            best_loss = final_loss
            best_hparams = {'T': T, 'num_hidden': num_hidden, 'n_layers': n_layers}

    print("\n--- Hyperparameter Search Complete ---")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Best Hyperparameters: {best_hparams}")

    fig, ax = plt.subplots(figsize=(10, 6))
    for loss_curve, hparams in all_loss_curves:
        label = f"T={hparams['T']}, H={hparams['num_hidden']}, L={hparams['n_layers']}"
        ax.plot(loss_curve, label=label, marker='o', linestyle='-') # Added marker and linestyle

    ax.set_title('Hyperparameter Search: Training Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Epoch Loss')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    plot_filename = 'hparam_loss_curves.png'
    fig.savefig(plot_filename)
    print(f"All loss curves plotted and saved to {plot_filename}")