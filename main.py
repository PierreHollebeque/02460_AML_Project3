import torch
import sys, os
from utils import load_dataset, load_model, save_model
from train import train
from ddpm import DDPM
from network import MODEL_REGISTRY
import argparse


    
current_script_dir = os.path.dirname(os.path.abspath(__file__))
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

parser.add_argument('--network-type', type=str, choices=MODEL_REGISTRY.keys(), help='Choose the network type')
parser.add_argument('--T', type=float, default=1000, metavar='V', help='Number of steps in the diffusion process (default: %(default)s)')


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

        # Define prior distribution
        D = x_sample.shape[1]

        # Initialize num_hidden to None, in case it's not a fully connected network
        num_hidden = None

        # Define the network
        network_type = args.network_type
        network = MODEL_REGISTRY[network_type](D)

        # Set the number of steps in the diffusion process
        T = args.T
        # Define model
        model = DDPM(network, T=T).to(args.device)


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




