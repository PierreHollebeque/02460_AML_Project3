from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split
import torch
import os, sys
from ddpm import DDPM
from network import MODEL_REGISTRY




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