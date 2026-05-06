import torch
from tqdm import tqdm
from torch_geometric.data import Data
import matplotlib.pyplot as plt


def train(model, optimizer, data_loader, epochs, device, scheduler=None):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()
    if scheduler is not None and not isinstance(scheduler, (torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
        raise TypeError("scheduler must be a torch.optim.lr_scheduler._LRScheduler or ReduceLROnPlateau instance or None")

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    loss_train = [] # to plot the loss over training

    for epoch in range(epochs):
        epoch_loss = 0.0
        data_iter = iter(data_loader)
        for graph in data_iter:
            graph = graph.to(device)
            optimizer.zero_grad()
            loss = model.loss(graph)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            loss_train.append(loss.item())

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                avg_epoch_loss = epoch_loss / len(data_loader)
                scheduler.step(avg_epoch_loss)
            else:
                scheduler.step() # Update learning rate after each epoch
    
    fig,ax = plt.subplots()
    ax.plot(loss_train)
    ax.grid(True)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Training Loss')
    fig.savefig('loss.png')
