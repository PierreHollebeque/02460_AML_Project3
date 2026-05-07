import torch
import torch.nn as nn


class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X, node_mask):
        """ X: bs, n, dx. 
            node_mask: bs, n
        """
        x_mask = node_mask.unsqueeze(-1).bool()
        
        m = (X * x_mask).sum(dim=1) / x_mask.sum(dim=1).clamp(min=1)
        
        X_min = X.masked_fill(~x_mask, float('inf'))
        mi = X_min.min(dim=1)[0]
        mi = torch.where(torch.isinf(mi), torch.zeros_like(mi), mi)
        
        X_max = X.masked_fill(~x_mask, -float('inf'))
        ma = X_max.max(dim=1)[0]
        ma = torch.where(torch.isinf(ma), torch.zeros_like(ma), ma)
        
        diff = X - m.unsqueeze(1)
        var = (diff ** 2 * x_mask).sum(dim=1) / (x_mask.sum(dim=1) - 1).clamp(min=1)
        std = torch.sqrt(var + 1e-6)
        
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E, node_mask):
        """ E: bs, n, n, de
            node_mask: bs, n
        """
        e_mask = (node_mask.unsqueeze(1).bool() & node_mask.unsqueeze(2).bool()).unsqueeze(-1)
        
        m = (E * e_mask).sum(dim=(1, 2)) / e_mask.sum(dim=(1, 2)).clamp(min=1)
        
        E_min = E.masked_fill(~e_mask, float('inf'))
        mi = E_min.min(dim=2)[0].min(dim=1)[0]
        mi = torch.where(torch.isinf(mi), torch.zeros_like(mi), mi)
        
        E_max = E.masked_fill(~e_mask, -float('inf'))
        ma = E_max.max(dim=2)[0].max(dim=1)[0]
        ma = torch.where(torch.isinf(ma), torch.zeros_like(ma), ma)
        
        diff = E - m.unsqueeze(1).unsqueeze(2)
        var = (diff ** 2 * e_mask).sum(dim=(1, 2)) / (e_mask.sum(dim=(1, 2)) - 1).clamp(min=1)
        std = torch.sqrt(var + 1e-6)
        
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


def masked_softmax(x, mask, **kwargs):
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)