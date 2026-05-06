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