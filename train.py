import torch
from tqdm import tqdm
from torch_geometric.data import Data
import matplotlib.pyplot as plt


def train(model, optimizer, data_loader, epochs, device, plot_loss=False, scheduler=None):
    """
    Train the graph diffusion model.

    Args:
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        data_loader (torch.utils.data.DataLoader): Data loader supplying batch data.
        epochs (int): Number of epochs to train.
        device (str): Compute device for execution.
        plot_loss (bool, optional): If True, plots and saves a training loss curve.
        scheduler (torch.optim.lr_scheduler, optional): Training learning rate scheduler.

    Returns:
        list: The average training loss per epoch.
    """
    model.train()
    if scheduler is not None and not isinstance(scheduler, (torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
        raise TypeError("scheduler must be a torch.optim.lr_scheduler._LRScheduler or ReduceLROnPlateau instance or None")

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    loss_train = []
    loss_epoch = []

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

            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()
            
        loss_epoch.append(epoch_loss / len(data_loader))
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                avg_epoch_loss = epoch_loss / len(data_loader)
                scheduler.step(avg_epoch_loss)
            else:
                scheduler.step()
    

    if plot_loss :
        fig,ax = plt.subplots()
        ax.plot(loss_train, alpha=0.3, label='Per-step Loss')
        
        steps_per_epoch = len(data_loader)
        epoch_x = [steps_per_epoch * (i + 0.5) for i in range(epochs)]
        ax.plot(epoch_x, loss_epoch, color='red', linewidth=2, label='Epoch Avg Loss')
        
        ax.grid(True)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Training Loss')
        ax.legend()
        fig.savefig('loss.png')

    return loss_epoch
