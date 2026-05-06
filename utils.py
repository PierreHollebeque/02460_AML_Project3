from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split
import torch
import os, sys



class PlaceHolder:
    """
    Container encapsulating graph variables: node (X), edge (E), and global (y) features.
    """
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """
        Change the device and dtype of X, E, y features dynamically.
        """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        """
        Apply a binary validity mask matching nodes and corresponding edges.
        """
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


class DatasetInfos:
    """
    A placeholder for dataset information. It holds dimensions and distributions
    of node, edge, and global features, compatible with the model's expectations.
    """
    def __init__(self, node_feature_dim: int, edge_feature_dim: int,
                 global_feature_dim: int, node_dist: torch.Tensor,
                 edge_dist: torch.Tensor):
        self.node_dist = node_dist
        self.edge_dist = edge_dist

        self.input_dims = {
            'X': node_feature_dim,
            'E': edge_feature_dim,
            'y': global_feature_dim
        }

        self.output_dims = {
            'X': node_feature_dim,
            'E': edge_feature_dim,
            'y': 0
        }




def load_dataset():
    """
    Load the MUTAG graph classification dataset.
    """
    dataset = TUDataset(root='./data/', name='MUTAG')

    # Fix seed for reproducible splits
    generator = torch.Generator().manual_seed(42)
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    return train_set, val_set, test_set

def load_model(model_path: str, device: str):
    """
    Instantiate and load pre-trained DDPM checkpoint weights into memory.
    """
    from ddpm import DDPM
    from network import MODEL_REGISTRY

    saved = torch.load(model_path, weights_only=False)
    model_type = saved.get("model_type")
    model_parameters = saved.get("model_parameters")
    model_parameters['device'] = device
    intern_model_parameters = saved.get("intern_model_parameters")
    model_dict = saved.get("state_dict")

    print("Loading model of type", model_type,file=sys.stderr, flush=True)
    if model_type is None:
        raise ValueError("Saved model file does not contain model_type.")

    intern_model_class = MODEL_REGISTRY.get(model_type)
    if intern_model_class is None:
        raise ValueError(f"Unknown model_type {model_type}")
    
    intern_model = intern_model_class(**intern_model_parameters)
    model = DDPM(intern_model, **model_parameters)
    model.load_state_dict(model_dict)

    return model.to(device)

def save_model(model, model_path: str):
    """
    Persist the current state of a trained DDPM model securely to disk.
    """
    save_dict = {
        "model_type": model.network.__class__.__name__,
        "intern_model_parameters": model.network.get_init_args(),
        "model_parameters": model.get_init_args(),
        "state_dict": model.state_dict(),
    }
    torch.save(save_dict, model_path)


def plot_view(train_set,all_generated_adj_matrices,sample_view):
    import networkx as nx
    from torch_geometric.utils import to_networkx
    import matplotlib.pyplot as plt
    import numpy as np

    
    num_plot = min(len(all_generated_adj_matrices), 4)
    if num_plot > 0:
        fig, axes = plt.subplots(num_plot, 2, figsize=(10, 5 * num_plot))
        axes = np.array(axes).reshape(num_plot, 2) # Ensure 2D format to avoid subscript errors
        
        for i in range(num_plot):
            # Train set example (left column)
            train_data = train_set[i] if not isinstance(train_set[i], (list, tuple)) else train_set[i][0]
            train_nx = to_networkx(train_data, to_undirected=True)
            
            # Generated example (right column)
            gen_adj = all_generated_adj_matrices[i].cpu().numpy()
            gen_nx = nx.from_numpy_array(gen_adj)
            
            axes[i, 0].set_title(f'Train Sample {i+1}')
            pos_train = nx.spring_layout(train_nx, seed=42)
            nx.draw(train_nx, pos=pos_train, ax=axes[i, 0], node_size=100, node_color='#1f78b4', edgecolors='black', edge_color='gray', width=1.5)
            
            axes[i, 1].set_title(f'Generated Sample {i+1}')
            pos_gen = nx.spring_layout(gen_nx, seed=42)
            nx.draw(gen_nx, pos=pos_gen, ax=axes[i, 1], node_size=100, node_color='#d62728', edgecolors='black', edge_color='gray', width=1.5)
            
        plt.tight_layout()
        plt.savefig(sample_view)
        plt.close()