import torch
import sys, os
import argparse

from train import train
from ddpm import DDPM
from network import MODEL_REGISTRY
from baseline import compute_empirical_distribution
from torch_geometric.loader import DataLoader
from utils import load_dataset, load_model, save_model, DummyDatasetInfos

    
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Add project root to sys.path to ensure modules like 'utils' are found
project_root_dir = os.path.dirname(current_script_dir)
sys.path.append(project_root_dir)

# HACK: This is to ensure that models saved when DummyDatasetInfos was defined
# in `main` can be loaded now that the class has been moved to `utils.py`.
# It maps the old location ('main.DummyDatasetInfos') to the new one for pickle.
sys.modules['__main__'].DummyDatasetInfos = DummyDatasetInfos

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test','baseline','stats'], help='what to do when running the script (default: %(default)s)')

parser.add_argument('--model-path', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')

parser.add_argument('--sample-path', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
parser.add_argument('--num-sample', type=int, default=1, metavar='N', help='number of samples to perform (default: %(default)s)')

parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')

parser.add_argument('--num-hidden', type=int, default=128, help='Number of hidden units (default: %(default)s)')
parser.add_argument('--n-layers', type=int, default=4, help='Number of transformer layers (default: %(default)s)')
parser.add_argument('--network-type', type=str, default='GraphTransformer', choices=MODEL_REGISTRY.keys(), help='Choose the network type (default: %(default)s)')
parser.add_argument('--T', type=int, default=100, metavar='V', help='Number of steps in the diffusion process (default: %(default)s)')


args = parser.parse_args()
print('# Options')
for key, value in sorted(vars(args).items()):
    print(key, '=', value)


# Load data
train_set, val_set, test_set = load_dataset()

if args.model_path :
    model = load_model(args.model_path, args.device)
else :
    model = None



# Choose mode to run
if args.mode == 'train':
    if not model:
        x_sample = train_set[0]
        if isinstance(x_sample, (list, tuple)):
            x_sample = x_sample[0]

        # --- Start of changes for DDPM compatibility ---
        # Define dataset_infos required by DDPM.
        # These dimensions and distributions need to be properly derived from your dataset.
        # For MUTAG, x_sample.x.shape[1] is typically the node feature dimension (e.g., 7 for atom types).
        # Edge features (E) and global features (y) need to be determined from the dataset structure.
        # Assuming for MUTAG:
        node_feature_dim = x_sample.x.shape[1]
        # Add + 1 to edge_feature_dim to integrate the "no edge" class
        edge_feature_dim = x_sample.edge_attr.shape[1] + 1
        # The 'y' feature will be used to pass the timestep t. It's a single scalar.
        global_feature_dim = 1

        # Placeholder for stationary distributions (should be computed from training data)
        # For discrete features, these are typically uniform or empirical distributions.
        dummy_node_dist = torch.ones(node_feature_dim) / node_feature_dim
        dummy_edge_dist = torch.ones(edge_feature_dim) / edge_feature_dim

        dataset_infos = DummyDatasetInfos(
            node_feature_dim, edge_feature_dim, global_feature_dim,
            dummy_node_dist, dummy_edge_dist
        )

        # Define the network
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

        # Set the number of steps in the diffusion process
        T = args.T
        # Define model
        model = DDPM(network, dataset_infos=dataset_infos, device=args.device, T=T).to(args.device)
        # --- End of changes for DDPM compatibility ---


    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    # Print the shape of a batch of data

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # Train model
    train(model, optimizer, train_loader, args.epochs, args.device, scheduler)

    # Save model
    save_model(model, args.model_path)

elif args.mode == 'test':
    if not model:
        raise ValueError(f"No model provided")
    raise NotImplementedError(f"Module not implemented")


elif args.mode == 'sample':

    import numpy as np

    if not model:
        raise ValueError(f"No model provided")
    
    all_n, _ = compute_empirical_distribution(train_set)
    # Sample several graphs with a number of nodes from the empirical distribution
    n_sampled = np.random.choice(all_n,size=args.num_sample, replace=True)
    n_nodes = torch.tensor(n_sampled, device=args.device)
    print(f"Sampling a graph with {n_sampled} nodes...")
    X, E, y = model.sample(n_nodes=n_nodes)

    # The returned E is a group of dense matrix with categorical edge types (0 means no edge).
    # We create a binary adjacency matrix from the first sample in the batch.
    adj_matrix = (E > 0).int()
    for i in range(args.num_sample):  
        print(f"Generated Adjacency Matrix (sample {i}):")
        print(adj_matrix[i,:n_nodes[i],:n_nodes[i]])

elif args.mode == 'baseline':
    raise NotImplementedError(f"Module not implemented")

elif args.mode == 'stats':
    raise NotImplementedError(f"Module not implemented")
