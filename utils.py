from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split
import torch
import os, sys
from ddpm import DDPM
from network import MODEL_REGISTRY



class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
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




def load_dataset():
    # Load the MUTAG dataset
    dataset = TUDataset(root='./data/', name='MUTAG')

    # Split the dataset
    rng = torch.Generator().manual_seed(0)
    train_set, val_set, test_set = random_split(dataset, (100, 44, 44), generator=rng)
    return train_set, val_set, test_set

def load_model(model_path: str, device: str):
    map_location = torch.device(device) if device != "cpu" else None
    if not os.path.exists(model_path):
        return None

    saved = torch.load(model_path, map_location=map_location)
    model_type = saved.get("model_type")
    model_parameters = saved.get("model_parameters")
    intern_model_parameters = saved.get("intern_model_parameters")
    model_dict = saved.get("state_dict")


    print("Loading model of type", model_type,file=sys.stderr, flush=True)
    if model_type is None:
        raise ValueError("Saved model file does not contain model_type.")

    intern_model = MODEL_REGISTRY.get(model_type)
    if intern_model is None:
        raise ValueError(f"Unknown model_type {model_type}")
    else : 
        intern_model = intern_model(**intern_model_parameters)

    intern_model.load_state_dict(model_dict)



    return DDPM(intern_model, **model_parameters).to(device)

def save_model(model, model_path: str):
    save_dict = {
        "model_type": model.network.__name__,
        "intern_model_parameters": model.network.get_init_args(),
        "model_parameters": model.get_init_args(),
        "state_dict": model.state_dict(),
    }
    torch.save(save_dict, model_path)