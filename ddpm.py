# Code inspired from DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import diffusion_utils, utils

class DDPM(nn.Module):
    def __init__(self, network, dataset_infos,beta_1=1e-4, beta_T=2e-2, T=100):
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

        self.Xdim = dataset_infos.input_dims['X']
        self.Edim = dataset_infos.input_dims['E']
        self.ydim = dataset_infos.input_dims['y']
        self.Xdim_output = dataset_infos.output_dims['X']
        self.Edim_output = dataset_infos.output_dims['E']
        self.ydim_output = dataset_infos.output_dims['y']
        self.node_dist = dataset_infos.nodes_dist





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

    def sample(self, n_nodes, number_chain_steps=None):
        """
        Sample from the model.

        Parameters:
        n_nodes: [torch.Tensor]
            The size of the graph to generate (batch size, number of nodes).
        number_chain_steps: [int]
            The number of steps in the diffusion process (default:None).
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        batch_size = n_nodes.shape[0]
        if number_chain_steps is None:
            number_chain_steps = self.T
        assert number_chain_steps < self.T

        n_nodes_max = torch.max(n_nodes).item()

        # Build the masks
        arange = torch.arange(n_nodes_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        node_mask = node_mask.float()

        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_feature_noise(X_size=(batch_size, n_nodes_max, self.Xdim_output),
                                                   E_size=(batch_size, n_nodes_max, n_nodes_max, self.Edim_output),
                                                   y_size=(batch_size, self.ydim_output),
                                                   node_mask=node_mask)
        
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()
        

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            z_s = self.sample_p_zs_given_zt(s=s_norm, t=t_norm, X_t=X, E_t=E, y_t=y, node_mask=node_mask)
            X, E, y = z_s.X, z_s.E, z_s.y


        final_graph = self.sample_discrete_graph_given_z0(X, E, y, node_mask)
        X, E, y = final_graph.X, final_graph.E, final_graph.y
        assert (E == torch.transpose(E, 1, 2)).all()

        return X,E,y


    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = diffusion_utils.sigma_and_alpha_t_given_s(gamma_t,
                                                                                                       gamma_s,
                                                                                                       X_t.size())
        sigma_s = diffusion_utils.sigma(gamma_s, target_shape=X_t.size())
        sigma_t = diffusion_utils.sigma(gamma_t, target_shape=X_t.size())

        E_t = (E_t + E_t.transpose(1, 2)) / 2
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t}
        extra_data = self.compute_extra_data(noisy_data)
        eps = self.forward(noisy_data, extra_data, node_mask)

        # Compute mu for p(zs | zt).
        mu_X = X_t / alpha_t_given_s - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)) * eps.X
        mu_E = E_t / alpha_t_given_s.unsqueeze(1) - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)).unsqueeze(1) * eps.E
        mu_y = y_t / alpha_t_given_s.squeeze(1) - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)).squeeze(1) * eps.y

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the parameters derived from zt.
        z_s = diffusion_utils.sample_normal(mu_X, mu_E,  mu_y, sigma, node_mask).type_as(mu_X)

        return z_s

    def sample_discrete_graph_given_z0(self, X_0, E_0, y_0, node_mask):
        """ Samples X, E, y ~ p(X, E, y|z0): once the diffusion is done, we need to map the result
        to categorical values.
        """
        zeros = torch.zeros(size=(X_0.size(0), 1), device=X_0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma = diffusion_utils.SNR(-0.5 * gamma_0).unsqueeze(1)
        noisy_data = {'X_t': X_0, 'E_t': E_0, 'y_t': y_0, 't': torch.zeros(y_0.shape[0], 1).type_as(y_0)}
        extra_data = self.compute_extra_data(noisy_data)
        eps0 = self.forward(noisy_data, extra_data, node_mask)

        # Compute mu for p(zs | zt).
        sigma_0 = diffusion_utils.sigma(gamma_0, target_shape=eps0.X.size())
        alpha_0 = diffusion_utils.alpha(gamma_0, target_shape=eps0.X.size())

        pred_X = 1. / alpha_0 * (X_0 - sigma_0 * eps0.X)
        pred_E = 1. / alpha_0.unsqueeze(1) * (E_0 - sigma_0.unsqueeze(1) * eps0.E)
        pred_y = 1. / alpha_0.squeeze(1) * (y_0 - sigma_0.squeeze(1) * eps0.y)
        assert (pred_E == torch.transpose(pred_E, 1, 2)).all()

        sampled = diffusion_utils.sample_normal(pred_X, pred_E, pred_y, sigma, node_mask).type_as(pred_X)
        assert (sampled.E == torch.transpose(sampled.E, 1, 2)).all()

        sampled = utils.unnormalize(sampled.X, sampled.E, sampled.y, self.norm_values,
                                    self.norm_biases, node_mask, collapse=True)
        return sampled






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
