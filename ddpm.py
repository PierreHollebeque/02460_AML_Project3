import torch
import torch.nn as nn
import diffusion_utils, utils
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_dense_batch


class DDPM(nn.Module):
    def __init__(self, network, dataset_infos, device, beta_1=1e-4, beta_T=2e-2, T=100, schedule='cosine', lambda_E=2.0):
        """
        Initialize a Discrete Denoising Diffusion Probabilistic Model (DDPM).

        Args:
            network (nn.Module): The network to use for the diffusion process.
            dataset_infos (DatasetInfos): The dataset feature dimensions and distributions.
            device (str): The execution device.
            beta_1 (float): The noise at the first step of the diffusion process.
            beta_T (float): The noise at the last step of the diffusion process.
            T (int): The number of steps in the diffusion process.
            schedule (str): The noise schedule to use ('linear' or 'cosine').
            lambda_E (float): The weight of the edge cross-entropy loss.
        """
        super(DDPM, self).__init__()
        self.device = device

        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self._init_args = {'dataset_infos': dataset_infos, 'beta_1': beta_1, 'beta_T': beta_T, 'T': T, 'schedule': schedule, 'lambda_E': lambda_E}
        self.T = T
        self.lambda_E = lambda_E

        self.Xdim = dataset_infos.input_dims['X']
        self.Edim = dataset_infos.input_dims['E']
        self.ydim = dataset_infos.input_dims['y']
        self.Xdim_output = dataset_infos.output_dims['X']
        self.Edim_output = dataset_infos.output_dims['E']
        self.ydim_output = dataset_infos.output_dims['y']
        self.node_dist = dataset_infos.node_dist

        # Stationary distributions for the discrete features to compute
        # the transition matrices Qt_X and Qt_E.
        self.limit_dist_X = dataset_infos.node_dist.to(device)
        self.limit_dist_E = dataset_infos.edge_dist.to(device)



        if schedule == 'linear':
            betas = diffusion_utils.custom_beta_schedule_discrete(T)
        elif schedule == 'cosine':
            betas = diffusion_utils.cosine_beta_schedule_discrete(T)
        self.beta = nn.Parameter(torch.tensor(betas, dtype=torch.float32), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)

        # Gamma schedule definition mapping to Signal-to-Noise Ratio (SNR)
        self.gamma_schedule = nn.Parameter(torch.log(1. - self.alpha_cumprod) - torch.log(self.alpha_cumprod), requires_grad=False)
    
    def get_init_args(self):
        return self._init_args

    def compute_extra_data(self, noisy_data):
        """
        Compute additional data to be concatenated with the noisy data before passing to the network.
        In this setup, the normalized timestep 't' is passed as a global feature.
        """
        bs = noisy_data['X_t'].shape[0]
        n_max = noisy_data['X_t'].shape[1]
        device = noisy_data['X_t'].device

        # The timestep 't' is assigned to the global feature 'y' for the network conditioning.
        return utils.PlaceHolder(
            X=torch.empty((bs, n_max, 0), device=device),
            E=torch.empty((bs, n_max, n_max, 0), device=device),
            y=noisy_data['t'] # This is the timestep t_norm (bs, 1)
        )

    def gamma(self, t_normalized):
        """
        Compute gamma values for a given normalized time tensor in [0, 1].
        """
        indices = (t_normalized * (self.T - 1)).round().long().clamp(0, self.T - 1)
        return self.gamma_schedule[indices]

    def get_Q(self, alpha_bar, limit_dist):
        """
        Calculate the transition matrix Q for the discrete diffusion process.
        """
        bs = alpha_bar.shape[0]
        d = limit_dist.shape[0]
        alpha_bar = alpha_bar.view(bs, 1, 1)
        limit_dist = limit_dist.view(1, 1, d)
        Q = alpha_bar * torch.eye(d, device=alpha_bar.device).unsqueeze(0) + (1 - alpha_bar) * limit_dist.expand(bs, d, d)
        return Q

    def sample(self, n_nodes, number_chain_steps=None):
        """
        Sample synthetic graphs iteratively from the discrete diffusion model.
        """
        batch_size = n_nodes.shape[0]
        device = self.alpha.device
        if number_chain_steps is None:
            number_chain_steps = self.T
        assert number_chain_steps <= self.T

        with torch.no_grad():
            n_nodes_max = torch.max(n_nodes).item()

            # Build valid node masks
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
            for s_int in tqdm(reversed(range(0, number_chain_steps)), total=number_chain_steps, desc="Reverse diffusion steps", leave=False):
                s_array = s_int - 1

                t_array = s_int * torch.ones((batch_size, 1), device=device)
                t_norm = t_array / self.T

                # Fetch diffusion coefficients
                alpha_bar_t = self.alpha_cumprod[s_int].expand(batch_size)
                alpha_t = self.alpha[s_int].expand(batch_size)
                if s_array >= 0:
                    alpha_bar_s = self.alpha_cumprod[s_array].expand(batch_size)
                else:
                    alpha_bar_s = torch.ones(batch_size, device=device)
                
                # Compute discrete transition matrices
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
                
                # Convert class integer outputs into one-hot representations for the next step
                X = F.one_hot(z_s.X, num_classes=self.Xdim).float()
                E = F.one_hot(z_s.E, num_classes=self.Edim).float()
                y = z_s.y

            # Collapse one-hot vectors to final class predictions
            final_graph = utils.PlaceHolder(X=X, E=E, y=y)
            final_graph.mask(node_mask, collapse=True)
            X, E, y = final_graph.X, final_graph.E, final_graph.y
            
            return X, E, y

    def sample_p_zs_given_zt_discrete(self, t_norm, X_t, E_t, y_t, node_mask, Qt, Qsb, Qtb):
        """
        Sample from the conditional distribution zs ~ p(zs | zt) using discrete transition matrices.
        """
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t_norm}
        extra_data = self.compute_extra_data(noisy_data)
        
        # Predict logits for features X_0, E_0
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
        """
        Concatenate extra temporal features to the noisy data and execute network forward pass.
        """
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2)
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3)
        y = torch.hstack((noisy_data['y_t'], extra_data.y))
        return self.network(X, E, y, node_mask)

    def loss(self, graph, lambda_E=None):
        """
        Compute the Cross-Entropy training loss for the DDPM algorithm applied to graphs.
        """
        if lambda_E is None:
            lambda_E = self.lambda_E

        # Convert torch_geometric.data.Batch to dense tensors
        X_0_onehot, node_mask = to_dense_batch(graph.x, batch=graph.batch)
        X_0 = X_0_onehot.argmax(dim=-1) # [bs, n_max]

        # Extract dense adjacency matrix and recover edge class labels
        E_0_onehot = to_dense_adj(edge_index=graph.edge_index, batch=graph.batch, edge_attr=graph.edge_attr, max_num_nodes=X_0.shape[1])
        
        # Assign class 0 for missing edges
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

        # 3. & 4. Predict unnormalized logits mapped from one-hot features
        X_t_onehot = F.one_hot(X_t, num_classes=self.Xdim).float()
        E_t_onehot = F.one_hot(E_t, num_classes=self.Edim).float()

        # Cast and normalize diffusion timestep for network input conditioning
        t_norm = t.float().unsqueeze(-1) / self.T
        pred = self.network(X_t_onehot, E_t_onehot, t_norm, node_mask)
        pred_X_logits, pred_E_logits = pred.X, pred.E

        # 5. Compute masked Cross-Entropy Loss
        loss_X = F.cross_entropy(pred_X_logits.view(-1, self.Xdim_output), X_0.view(-1), reduction='none')
        node_mask_sum = node_mask.sum().float()
        if node_mask_sum > 0:
            loss_X = (loss_X * node_mask.view(-1).float()).sum() / node_mask_sum
        else:
            loss_X = torch.tensor(0.0, device=X_0.device) # No nodes, no node loss

        # Evaluate edge loss only within existing nodes and strictly off-diagonal
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(n_max, device=X_0.device, dtype=torch.bool).unsqueeze(0)
        edge_mask = edge_mask * diag_mask

        loss_E = F.cross_entropy(pred_E_logits.view(-1, self.Edim_output), E_0.view(-1), reduction='none')
        edge_mask_sum = edge_mask.sum().float()
        if edge_mask_sum > 0:
            loss_E = (loss_E * edge_mask.view(-1).float()).sum() / edge_mask_sum
        else:
            loss_E = torch.tensor(0.0, device=X_0.device) # No edges, no edge loss

        return loss_X + lambda_E * loss_E