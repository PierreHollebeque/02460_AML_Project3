import torch
import sys, os
from train import train
from ddpm import DDPM
from network import MODEL_REGISTRY
import argparse


    
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Add project root to sys.path to ensure modules like 'utils' are found
project_root_dir = os.path.dirname(current_script_dir)
sys.path.append(project_root_dir)

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test','baseline','stats'], help='what to do when running the script (default: %(default)s)')

parser.add_argument('--model-path', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
parser.add_argument('--sample-path', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')

parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')

parser.add_argument('--num-hidden', type=int, default=128, help='Number of hidden units for FcNetwork (default: %(default)s)')
parser.add_argument('--network-type', type=str, choices=MODEL_REGISTRY.keys(), help='Choose the network type')
parser.add_argument('--T', type=float, default=1000, metavar='V', help='Number of steps in the diffusion process (default: %(default)s)')


args = parser.parse_args()
print('# Options')
for key, value in sorted(vars(args).items()):
    print(key, '=', value)


from utils import load_dataset, load_model, save_model # Moved import after sys.path.append
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
        node_feature_dim = x_sample.x.shape[1] if hasattr(x_sample, 'x') and x_sample.x is not None else 1
        # For MUTAG, edge features are typically 3 (bond types: single, double, triple).
        # If your Data object doesn't have explicit edge features, you might need to infer this or set a default.
        edge_feature_dim = 3
        global_feature_dim = 0 # MUTAG typically doesn't have global features for this task

        # Placeholder for stationary distributions (should be computed from training data)
        # For discrete features, these are typically uniform or empirical distributions.
        dummy_node_dist = torch.ones(node_feature_dim) / node_feature_dim
        dummy_edge_dist = torch.ones(edge_feature_dim) / edge_feature_dim

        # Create a simple object to hold dataset information
        class DummyDatasetInfos:
            def __init__(self, node_f_dim, edge_f_dim, global_f_dim, node_dist, edge_dist):
                self.input_dims = {'X': node_f_dim, 'E': edge_f_dim, 'y': global_f_dim}
                self.output_dims = {'X': node_f_dim, 'E': edge_f_dim, 'y': global_f_dim} # For discrete, output dims are usually same as input
                self.node_dist = node_dist
                self.edge_dist = edge_dist

        dataset_infos = DummyDatasetInfos(
            node_feature_dim, edge_feature_dim, global_feature_dim,
            dummy_node_dist, dummy_edge_dist
        )

        # Define the network
        network_type = args.network_type
        if network_type == 'FcNetwork':
            # WARNING: FcNetwork is a simple MLP and is NOT a Graph Neural Network.
            # The DDPM expects a GNN that processes (X, E, t, node_mask) and outputs logits.
            # This instantiation is syntactically correct for FcNetwork but semantically incorrect for the DDPM's task.
            network = MODEL_REGISTRY[network_type](input_dim=node_feature_dim, num_hidden=args.num_hidden)
        elif network_type == 'Unet':
            # WARNING: Unet is for image data and is NOT suitable for graph data.
            network = MODEL_REGISTRY[network_type]()
        else:
            raise ValueError(f"Unsupported network type: {network_type}")

        # Set the number of steps in the diffusion process
        T = args.T
        # Define model
        model = DDPM(network, dataset_infos=dataset_infos, T=T).to(args.device)
        # --- End of changes for DDPM compatibility ---


    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    # Print the shape of a batch of data

    # Train model
    train(model, optimizer, train_set, args.epochs, args.device, scheduler)

    # Save model
    save_model(model, args.model_path)

elif args.mode == 'test':
    if not model:
        raise ValueError(f"No model provided")
    raise NotImplementedError(f"Module not implemented")


elif args.mode == 'sample':

    if not model:
        raise ValueError(f"No model provided")
    raise NotImplementedError(f"Module not implemented")

elif args.mode == 'baseline':
    raise NotImplementedError(f"Module not implemented")

elif args.mode == 'stats':
    raise NotImplementedError(f"Module not implemented")
