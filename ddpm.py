# Code inspired from DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn

class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """
        batch_size = x.shape[0]
        # Sampling t uniformly from {0,....,T-1})
        t = torch.randint(0, self.T, size=(batch_size, 1), device=x.device)
        # Sampling the uniform
        epsilon = torch.randn_like(x,device=x.device)
        output = self.network(torch.sqrt(self.alpha_cumprod[t]) * x + torch.sqrt(1 - self.alpha_cumprod[t]) * epsilon, t.float() / self.T)
        # Division of time by self.T in order to cap the time between 0 and 1, else time is predominating compared to images.
        neg_elbo = torch.sum((epsilon - output) ** 2, dim=1)
        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)
        batch_size = x_t.shape[0]
        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T-1, -1, -1) :
            if t > 0 : 
                z = torch.randn(shape).to(self.alpha.device)
            else :
                z = 0
            # Pass loop
            x_t = 1 / torch.sqrt(self.alpha[t]) * (x_t - (1-self.alpha[t]) /
                                torch.sqrt(1-self.alpha_cumprod[t]) * self.network(
                                    x_t,torch.full((batch_size,1),fill_value=t/self.T, dtype=torch.float).to(self.alpha.device)
                                    )
                                ) + torch.sqrt(self.beta[t]) * z
            # Division of time by self.T in order to cap the time between 0 and 1 - else it will predominate compared to the x.

        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()
