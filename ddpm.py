# Code inspired from DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import diffusion_utils, utils
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_dense_batch


class DDPM(nn.Module):
    def __init__(self, network, dataset_infos, device, beta_1=1e-4, beta_T=2e-2, T=100):
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
        self.device = device

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
        self.limit_dist_X = dataset_infos.node_dist.to(device)
        self.limit_dist_E = dataset_infos.edge_dist.to(device)




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

    def get_Q(self, alpha_bar, limit_dist):
        """
        Calculates the transition matrix Q for the discrete diffusion.
        """
        bs = alpha_bar.shape[0]
        d = limit_dist.shape[0]
        alpha_bar = alpha_bar.view(bs, 1, 1)
        limit_dist = limit_dist.view(1, 1, d)
        Q = alpha_bar * torch.eye(d, device=alpha_bar.device).unsqueeze(0) + (1 - alpha_bar) * limit_dist.expand(bs, d, d)
        return Q

    def sample(self, n_nodes, number_chain_steps=None):
        """
        Samples from the discrete diffusion model.
        """
        batch_size = n_nodes.shape[0]
        device = self.alpha.device
        if number_chain_steps is None:
            number_chain_steps = self.T
        assert number_chain_steps <= self.T

        with torch.no_grad(): # Sampling is an inference process, no gradients needed
            n_nodes_max = torch.max(n_nodes).item()

            # Build the masks
            arange = torch.arange(n_nodes_max, device=device).unsqueeze(0).expand(batch_size, -1)
            node_mask = arange < n_nodes.unsqueeze(1)

            # 1. Sample z_T from the limit distribution (returns one-hot)
            limit_dist = utils.PlaceHolder(
                X=self.limit_dist_X,
                E=self.limit_dist_E,
                y=torch.zeros(self.ydim_output, device=device)
            )
            z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist, node_mask)
            
            X, E, y = z_T.X, z_T.E, z_T.y
            
            # Iteratively sample p(z_s | z_t) for t = T-1 down to 0
            for s_int in reversed(range(0, number_chain_steps)):
                s_array = s_int - 1

                t_array = s_int * torch.ones((batch_size, 1), device=device)
                t_norm = t_array / self.T

                # Get alpha and alpha_bar
                alpha_bar_t = self.alpha_cumprod[s_int].expand(batch_size)
                alpha_t = self.alpha[s_int].expand(batch_size)
                if s_array >= 0:
                    alpha_bar_s = self.alpha_cumprod[s_array].expand(batch_size)
                else:
                    alpha_bar_s = torch.ones(batch_size, device=device)
                
                # Compute Transition Matrices (limit_dist_X/E are already on device)
                Qt_X = self.get_Q(alpha_t, self.limit_dist_X)
                Qtb_X = self.get_Q(alpha_bar_t, self.limit_dist_X)
                Qsb_X = self.get_Q(alpha_bar_s, self.limit_dist_X)

                Qt_E = self.get_Q(alpha_t, self.limit_dist_E)
                Qtb_E = self.get_Q(alpha_bar_t, self.limit_dist_E)
                Qsb_E = self.get_Q(alpha_bar_s, self.limit_dist_E)

                Qt = utils.PlaceHolder(X=Qt_X, E=Qt_E, y=None)
                Qtb = utils.PlaceHolder(X=Qtb_X, E=Qtb_E, y=None)
                Qsb = utils.PlaceHolder(X=Qsb_X, E=Qsb_E, y=None)

                z_s = self.sample_p_zs_given_zt_discrete(t_norm, X, E, y, node_mask, Qt, Qsb, Qtb)
                
                # z_s contains class integers. Convert them to one-hot for the next step.
                X = F.one_hot(z_s.X, num_classes=self.Xdim).float()
                E = F.one_hot(z_s.E, num_classes=self.Edim).float()
                y = z_s.y

            # After the loop, X and E are the final one-hot generated features
            final_graph = utils.PlaceHolder(X=X, E=E, y=y)
            final_graph.mask(node_mask, collapse=True) # Transforms one-hot to class integers
            X, E, y = final_graph.X, final_graph.E, final_graph.y
            
            return X, E, y

    def sample_p_zs_given_zt_discrete(self, t_norm, X_t, E_t, y_t, node_mask, Qt, Qsb, Qtb):
        """Samples from zs ~ p(zs | zt) using discrete transition matrices."""
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t_norm}
        extra_data = self.compute_extra_data(noisy_data)
        
        # Predict X_0, E_0 logits
        pred = self.forward(noisy_data, extra_data, node_mask)
        
        # Convert logits to probability distributions
        pred_prob_X = F.softmax(pred.X, dim=-1)
        pred_prob_E = F.softmax(pred.E, dim=-1)
        
        # Compute posterior q(z_s | z_t, z_0)
        posteriors = diffusion_utils.posterior_distributions(
            X=pred_prob_X, E=pred_prob_E, y=pred.y, 
            X_t=X_t, E_t=E_t, y_t=y_t, 
            Qt=Qt, Qsb=Qsb, Qtb=Qtb
        )
        
        # Sample from the posterior
        z_s = diffusion_utils.sample_discrete_features(posteriors.X, posteriors.E, node_mask)
        
        return z_s

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

        # Compute cumulative transition matrices Qt_bar
        Qtb_X = self.get_Q(alpha_bar, self.limit_dist_X)
        Qtb_E = self.get_Q(alpha_bar, self.limit_dist_E)

        # Node corruption: sample from q(X_t | X_0) = X_0 @ Qt_bar
        prob_X = torch.bmm(F.one_hot(X_0, num_classes=self.Xdim).float(), Qtb_X)
        X_t = torch.multinomial(prob_X.view(-1, self.Xdim), 1).view(bs, n_max)

        # Edge corruption: sample from q(E_t | E_0) = E_0 @ Qt_bar
        E_0_flat = F.one_hot(E_0, num_classes=self.Edim).float().view(bs, n_max * n_max, self.Edim)
        prob_E = torch.bmm(E_0_flat, Qtb_E)
        E_t = torch.multinomial(prob_E.view(-1, self.Edim), 1).view(bs, n_max, n_max)

        # Ensure symmetric edges (undirected graphs) and 0 on diagonal
        E_t = torch.triu(E_t, diagonal=1)
        E_t = E_t + E_t.transpose(1, 2)

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