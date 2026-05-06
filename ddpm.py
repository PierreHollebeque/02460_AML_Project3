# Code inspired from DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import diffusion_utils, utils
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_dense_batch


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
        self.device = 'cpu'

        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self._init_args = {'dataset_infos': dataset_infos, 'beta_1': beta_1, 'beta_T': beta_T, 'T': T}
        self.T = T

        self.Xdim = dataset_infos.input_dims['X']
        self.Edim = dataset_infos.input_dims['E']
        self.ydim = dataset_infos.input_dims['y']
        self.Xdim_output = dataset_infos.output_dims['X']
        self.Edim_output = dataset_infos.output_dims['E']
        self.ydim_output = dataset_infos.output_dims['y']
        self.node_dist = dataset_infos.node_dist

        # Stationary distributions for the discrete features, as per the user's request
        # for the transition matrices Qt_X and Qt_E.
        # These are the 'a_m' and 'b_m' distributions.
        self.limit_dist_X = dataset_infos.node_dist
        self.limit_dist_E = dataset_infos.edge_dist # Assuming this is provided





        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)

        # This is log(sigma^2/alpha^2) = -log(SNR) = log( (1-alpha_cumprod) / alpha_cumprod )
        self.gamma_schedule = nn.Parameter(torch.log(1. - self.alpha_cumprod) - torch.log(self.alpha_cumprod), requires_grad=False)
    
    def get_init_args(self):
        return self._init_args

    def compute_extra_data(self, noisy_data):
        """
        Computes additional data to be concatenated with the noisy data before passing to the network.
        In this setup, the timestep 't' is passed as a global feature.
        """
        bs = noisy_data['X_t'].shape[0]
        n_max = noisy_data['X_t'].shape[1]
        device = noisy_data['X_t'].device

        # The timestep 't' from noisy_data is the global feature 'y' for the network.
        # noisy_data['y_t'] is the noisy global feature, which is (bs, 0) in this case.
        # The network expects a global feature of dimension self.ydim (which is 1).
        # So, extra_data.y should be noisy_data['t'] to make the total y dimension 1.
        return utils.PlaceHolder(
            X=torch.empty((bs, n_max, 0), device=device),
            E=torch.empty((bs, n_max, n_max, 0), device=device),
            y=noisy_data['t'] # This is the timestep t_norm (bs, 1)
        )

    def gamma(self, t_normalized):
        """
        Computes gamma values for a given normalized time tensor.
        t_normalized is a tensor of values in [0, 1].
        It is used to look up the gamma value from the precomputed schedule.
        """
        # The schedule has T values, indexed 0 to T-1.
        # We map t_normalized from [0, 1] to indices [0, T-1].
        indices = (t_normalized * (self.T - 1)).round().long().clamp(0, self.T - 1)
        return self.gamma_schedule[indices]


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
        assert number_chain_steps <= self.T

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
        for s_int in reversed(range(0, number_chain_steps)):
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

        # The 'unnormalize' function is not defined and seems to be a leftover.
        # The goal is to convert the continuous predictions to discrete class indices.
        # The `mask` method with `collapse=True` performs this by taking argmax.
        sampled.mask(node_mask, collapse=True)
        return sampled

    def forward(self, noisy_data, extra_data, node_mask):
        """ Concatenates extra data to the noisy data, then calls the network. """
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2)
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3)
        y = torch.hstack((noisy_data['y_t'], extra_data.y)) # noisy_data['y_t'] is (bs, 0), extra_data.y is (bs, 1) -> y is (bs, 1)
        return self.network(X, E, y, node_mask)

    def loss(self, graph, lambda_E=1.0):
        """
        Computes the loss for the DDPM.

        Algorithm:
        1. Input: A graph G = (X, E) with discrete features.
        2. Sample t ~ U(1, ..., T)
        3. Sample a noisy graph Gt ~ q(Gt | G)
        4. The model predicts the original graph features: p_theta(G | Gt) -> (pX, pE)
        5. The loss is the cross-entropy: lCE(pX, X) + λ * lCE(pE, E)
        """
        # Assuming `graph` is an object with .X, .E, and .node_mask attributes.
        # This is not a standard torch_geometric.data.Data object.
        # The data loader needs to be adapted to yield such objects.
        # X should be [bs, n_max, 1] with integer class values.
        # E should be [bs, n_max, n_max] with integer class values.

        # Convert torch_geometric.data.Batch to dense tensors
        X_0_onehot, node_mask = to_dense_batch(graph.x, batch=graph.batch)
        X_0 = X_0_onehot.argmax(dim=-1) # [bs, n_max]

        # The to_dense_adj function returns a dense adjacency matrix with edge attributes.
        # For MUTAG, edge_attr is one-hot, so we take argmax to get class labels.
        E_0_onehot = to_dense_adj(edge_index=graph.edge_index, batch=graph.batch, edge_attr=graph.edge_attr, max_num_nodes=X_0.shape[1])
        
        # 0 class for "no-edges"
        is_edge = E_0_onehot.sum(dim=-1) > 0
        E_0 = E_0_onehot.argmax(dim=-1) + 1  
        E_0[~is_edge] = 0                    
        
        bs, n_max = X_0.shape

        # 1. Sample t ~ U(0, ..., T-1)
        t = torch.randint(0, self.T, (bs,), device=X_0.device)

        # 2. Sample noisy graph Gt (corruption process)
        alpha_bar = self.alpha_cumprod.to(X_0.device)[t]  # (bs,)

        # Sample uniform noise for corruption
        # The user specified transition matrices imply sampling from a stationary distribution
        # for the corruption part.
        limit_dist_X = self.limit_dist_X.to(X_0.device)
        X_random = torch.multinomial(limit_dist_X.expand(X_0.numel(), -1), 1).view(X_0.shape)

        limit_dist_E = self.limit_dist_E.to(E_0.device)
        E_random = torch.multinomial(limit_dist_E.expand(E_0.numel(), -1), 1).view(E_0.shape)

        # Create corruption masks. Where mask is True, we replace with random noise.
        mask_X = torch.rand(X_0.shape, device=X_0.device) > alpha_bar.view(bs, 1)
        mask_E = torch.rand(E_0.shape, device=E_0.device) > alpha_bar.view(bs, 1, 1)

        X_t = torch.where(mask_X, X_random, X_0)
        E_t = torch.where(mask_E, E_random, E_0)

        # 3. & 4. Predict logits for original graph features using the network
        # The network needs one-hot encoded features.
        X_t_onehot = F.one_hot(X_t, num_classes=self.Xdim).float()
        E_t_onehot = F.one_hot(E_t, num_classes=self.Edim).float()

        # The network should be a graph neural network that takes (X, E, t, node_mask)
        # and returns logits for node and edge features. The networks in `network.py`
        # are not suitable for this task. This part requires a proper GNN.

        # Reshape and cast t for the network, which expects a float tensor for 'y'.
        t_float = t.float().unsqueeze(-1)
        pred = self.network(X_t_onehot, E_t_onehot, t_float, node_mask)
        pred_X_logits, pred_E_logits = pred.X, pred.E

        # 5. Compute Cross-Entropy Loss
        loss_X = F.cross_entropy(pred_X_logits.view(-1, self.Xdim_output), X_0.view(-1), reduction='none')
        loss_X = (loss_X * node_mask.view(-1).float()).sum() / (node_mask.sum() + 1e-8)

        # For edges, we only consider edges between existing nodes and off-diagonal.
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(n_max, device=X_0.device, dtype=torch.bool).unsqueeze(0)
        edge_mask = edge_mask * diag_mask

        loss_E = F.cross_entropy(pred_E_logits.view(-1, self.Edim_output), E_0.view(-1), reduction='none')
        loss_E = (loss_E * edge_mask.view(-1).float()).sum() / (edge_mask.sum() + 1e-8)

        return loss_X + lambda_E * loss_E