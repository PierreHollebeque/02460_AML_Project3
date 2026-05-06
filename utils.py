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

    train_set = dataset
    val_set, test_set = [], []
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
    # Define a color map for different edge types. 
    # Types usually map to: 1=Single, 2=Double, 3=Triple, 4=Aromatic etc.
    
    num_plot = min(len(all_generated_adj_matrices), 4)
    if num_plot > 0:
        fig, axes = plt.subplots(num_plot, 2, figsize=(10, 5 * num_plot))
        axes = np.array(axes).reshape(num_plot, 2) # Ensure 2D format to avoid subscript errors
        edge_colors_map = {1: 'black', 2: 'red', 3: 'blue', 4: 'green', 5: 'orange', 6: 'purple'}

        for i in range(num_plot):
            # Train set example (left column)
            train_data = train_set[i] if not isinstance(train_set[i], (list, tuple)) else train_set[i][0]
            has_edge_attr = hasattr(train_data, 'edge_attr') and train_data.edge_attr is not None
            train_nx = to_networkx(train_data, edge_attrs=['edge_attr'] if has_edge_attr else None, to_undirected=True)
            
            # Remove isolated nodes (nodes with no edges)
            train_nx.remove_nodes_from(list(nx.isolates(train_nx)))
            
            train_edge_colors = []
            for u, v, data in train_nx.edges(data=True):
                if 'edge_attr' in data and data['edge_attr'] is not None:
                    attr = data['edge_attr']
                    # Retrieve original class integer from one-hot encoding (+1 for class mapping)
                    if isinstance(attr, (list, tuple)):
                        edge_type = np.argmax(attr) + 1
                    elif hasattr(attr, 'numpy'):
                        edge_type = torch.argmax(attr).item() + 1
                    elif isinstance(attr, np.ndarray):
                        edge_type = np.argmax(attr) + 1
                    else:
                        edge_type = int(attr)
                    train_edge_colors.append(edge_colors_map.get(edge_type, 'gray'))
                else:
                    train_edge_colors.append('black')
                
            # Generated example (right column)
            gen_adj = all_generated_adj_matrices[i].cpu().numpy()
            gen_nx = nx.from_numpy_array(gen_adj)
            
            # Remove isolated nodes (nodes with no edges)
            gen_nx.remove_nodes_from(list(nx.isolates(gen_nx)))
            
            gen_edge_colors = []
            for u, v, data in gen_nx.edges(data=True):
                edge_type = int(data.get('weight', 1))
                gen_edge_colors.append(edge_colors_map.get(edge_type, 'black'))
                
            axes[i, 0].set_title(f'Train Sample {i+1}')
            pos_train = nx.spring_layout(train_nx)
            nx.draw(train_nx, pos_train, ax=axes[i, 0], node_size=50, node_color='#1f78b4', 
                    edge_color=train_edge_colors if train_edge_colors else 'black', width=2.0)
            
            axes[i, 1].set_title(f'Generated Sample {i+1}')
            nx.draw(gen_nx, ax=axes[i, 1], node_size=50, node_color='#d62728', edge_color='gray')
            
        plt.tight_layout()
        plt.savefig(sample_view)
        plt.close()