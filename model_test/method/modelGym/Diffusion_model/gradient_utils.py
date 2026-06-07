import torch
import networkx as nx
import torch_geometric as tg
import math
import time
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from torch.func import vmap

def get_guided_rates(
    predictor_log_prob,
    xt: torch.Tensor,  # Shape: (B, D)
    t: float,
    R_t: torch.Tensor,  # Shape: (B, D, S)
    S: int,
    use_tag: bool = False,
    guide_temp: float = 0.5,
    log_prob_ratio_cutoff: float = 80.0,
) -> torch.Tensor:
    """
    Computes guide-adjusted rates for predictor guidance.

    Implements both exact guidance by computing likelihood ratios for all possible transitions,
    and Taylor-approximated guidance (TAG) using gradients of the predictor.

    Args:
        predictor_log_prob (callable): Function that takes (x,t) and returns log p(y|x,t)
        xt (torch.Tensor): Current states of shape (B, D)
        t (float): Current time
        R_t (torch.Tensor): Unconditional rates of shape (B, D, S)
        S (int): Size of categorical state space
        use_tag (bool, optional): Whether to use Taylor approximation. Defaults to False.
        guide_temp (float, optional): Guidance temperature. Defaults to 1.
        log_prob_ratio_cutoff (float, optional): Maximum value for log ratios. Defaults to 80.

    Returns:
        torch.Tensor: Guide-adjusted rates of shape (B, D, S)
    """
    B, D = xt.shape
    device = xt.device

    if not use_tag:
        # Exact guidance case
        # log p(y|x=z_t), shape (B,)
        log_prob_xt = predictor_log_prob(xt, t)

        # Get all jump transitions, shape (B*D*S, D)
        xt_jumps = get_all_jump_transitions(xt, S)

        # Get log probs for all transitions
        # Shape: (B*D*S,) -> (B, D, S)
        log_prob_xt_jumps = predictor_log_prob(
            xt_jumps, t
        ).view(B, D, S)

        # Compute log ratios
        # Shape (B, D, S)
        log_prob_ratio = log_prob_xt_jumps - log_prob_xt.view(B, 1, 1)

    else:
        # Taylor-approximated guidance (TAG) case
        # One-hot encode categorical data, shape (B, D, S)
        xt_ohe = F.one_hot(xt.long(), num_classes=S).to(torch.float)

        # \grad_{x}{log p(y|x)}(z_t), shape (B, D, S)
        with torch.inference_mode(False):
            xt_ohe.requires_grad_(True)
            # log p(y|x=z_t), shape (B,)
            xt=xt_ohe[...,1]#[B,D]
            log_prob_xt_ohe_sum = predictor_log_prob(xt, t).requires_grad_(True)

            log_prob_xt_ohe_sum.backward()
            # Shape (B, D, S)
            grad_log_prob_xt_ohe = xt_ohe.grad
            grad_log_prob_xt_ohe=nn.functional.normalize(grad_log_prob_xt_ohe, p=2, dim=-1)

        # 1st order Taylor approximation of the log difference
        # Shape (B, D, S)
        log_prob_ratio = grad_log_prob_xt_ohe - (xt_ohe * grad_log_prob_xt_ohe).sum(
            dim=-1, keepdim=True
        )

    # Scale log prob ratio by temperature
    log_prob_ratio /= guide_temp

    # Clamp the log prob ratio to avoid overflow in exp
    log_prob_ratio = torch.clamp(log_prob_ratio, max=log_prob_ratio_cutoff)
    # Exponentiate to get p(y|x=z~) / p(y|x=z_t)
    prob_ratio = torch.exp(log_prob_ratio).to(device)
    # Multiply the reverse rate elementwise with the density ratio
    # Note this doesn't deal with the diagonals
    R_t = R_t * prob_ratio

    if R_t.isnan().any():
        raise ValueError(f"The rate matrix 'R_t' contains NaNs.")

    return R_t


def get_all_jump_transitions(
    xt: torch.Tensor,  # Shape: (B, D)
    S: int,
) -> torch.Tensor:  # Shape: (B*D*S, D)
    """
    Gets all possible single-dimension transitions from current states.

    Creates a tensor containing all possible states that differ from input states
    in exactly one position, for each sequence in the batch.

    Args:
        xt: Current state tensor of shape (batch_size, sequence_length)
        S: Size of categorical state space (number of possible values per position)

    Returns:
        Tensor of shape (batch_size * sequence_length * state_space, sequence_length)
        containing all possible single-token transitions
    """
    B, D = xt.shape
    device = xt.device

    # Create B*D*S copies of input sequences
    # Shape: (B, 1, D) -> (B, D*S, D)
    xt_expand = xt.unsqueeze(1).repeat(1, D * S, 1)
    # Flatten batch and transition dimensions
    # Shape: (B, D*S, D) -> (B*D*S, D)
    xt_expand = xt_expand.view(-1, D)

    # Create indices for all possible transitions
    # Shape: (D*S,) -> (B, D*S) -> (B*D*S,)
    jump_idx = torch.arange(D * S).to(device)
    jump_idx = jump_idx.repeat(B, 1).flatten()

    # Create tensor for states after one transition
    xt_jumps = xt_expand.clone()

    # Calculate which dimension changes for each transition
    # Shape: (B*D*S,)
    jump_dims = jump_idx // S

    # Calculate new value for changed dimension
    # Shape: (B*D*S,)
    jump_states = jump_idx % S

    # Apply transitions by assigning new values at transition dimensions
    # Shape: (B*D*S, D)
    xt_jumps[
        torch.arange(jump_idx.size(0), device=device),
        jump_dims,  # Index the transitioned dimension
    ] = jump_states  # Assign the new state

    return xt_jumps


class fitness_map:
    def __init__(self, fitness_func_type,tensor_G, diffusion_type, temperature=1.0,scale_T=1.0,guide_model=None,CEP_guide_model=None):
        self.fitness_func_type = fitness_func_type
        self.temperature = temperature
        self.tensor_G = tensor_G
        self.diffusion_type=diffusion_type
        self.scale_T=scale_T
        self.guide_model=guide_model
        self.CEP_guide_model=CEP_guide_model
    def fitness_func(self, node_v):
        self.tensor_G=self.tensor_G.to(node_v.device)

        if self.diffusion_type=="gaussian":
            node_v=0.5*node_v+0.5

        node_v = 1 - node_v
        node_mask = torch.round(node_v).to(torch.bool)
        nodes_to_keep = torch.nonzero(node_mask).reshape(-1).to(node_v.device)
        subgraph = self.tensor_G.subgraph(nodes_to_keep)
        subgraph = tg.utils.to_networkx(subgraph, to_undirected=True)
      #  net_G=tg.utils.to_networkx(self.tensor_G, to_undirected=True)
        if self.fitness_func_type == "LCC" or self.fitness_func_type == "MVC":
            if subgraph.number_of_edges() == 0:
                fitness_value = 0
            else:
                fitness_value = len(max(nx.connected_components(subgraph), key=len))
        elif self.fitness_func_type == "GCC":
            fitness_value = nx.transitivity(subgraph)
        elif self.fitness_func_type == "ACC":
            fitness_value = nx.average_clustering(subgraph)
        elif self.fitness_func_type == "NC":
            estrada_index = nx.estrada_index(subgraph)
            # 自然连通性
            n = nx.number_of_nodes(subgraph)
            fitness_value = np.log(estrada_index / n)
        elif self.fitness_func_type == "GE":
            fitness_value = nx.global_efficiency(subgraph)

        elif self.fitness_func_type == "Density":
            fitness_value = nx.density(subgraph)
        else:
            raise ValueError("Invalid fitness function type: {}. ".format(self.fitness_func_type))
        return torch.tensor([fitness_value]
                            , dtype=torch.float).to(node_v.device)

    def forward(self, x):
        power_list = []
        for i in range(x.shape[0]):
            power_list.append( -self.fitness_func(x[i]) /(self.scale_T * self.temperature))
        power = torch.tensor(power_list)
        p = torch.exp(power)
        return p
    def log_prob(self, x,t):

        power_list = []
        for i in range(x.shape[0]):
            power_list.append(-self.fitness_func(x[i]) / (self.scale_T * (1-t+1e-9)*self.temperature))
        power = torch.tensor(power_list)

        return power
    def differentiable_log_prob(self, x,t):
        #
        power=torch.tensor([0.]).to(x.device)
        if self.guide_model is not None:
            for i in range(x.shape[0]):

                power+=-self.guide_model(x[i]).squeeze() / (self.scale_T * (1-t+1e-9)*self.temperature)
        elif self.CEP_guide_model is not None:
            for i in range(x.shape[0]):
                self.tensor_G.x=x[i]
                graph_data=self.tensor_G.to(x.device)
                power+=-self.CEP_guide_model(graph_data,(1-t+1e-9)*self.temperature)/self.scale_T
        else:
            ValueError("No guide model provided for Taylor-approximated guidance.")

        return power

    def __call__(self, x):
        return self.forward(x)
#前向过程 连续过程，转移概率是高斯分布
class BayesianEstimator:
    """Bayesian Estimator of the origin points, based on current samples and fitness values."""

    def __init__(self, x: torch.tensor, fitness: torch.tensor, alpha):
        # self.x size: (n, d) n population size, d dimension
        self.x = x
        self.fitness = fitness
        self.alpha = alpha

    def density(self, x):

        return torch.ones(x.shape[0]) / x.shape[0]
    @staticmethod
    def norm(x):
        if x.shape[-1] == 1:
            # for some reason, torch.norm become very slow when dim=1, so we use torch.abs instead
            return torch.abs(x).squeeze(-1)
        else:
            return torch.norm(x, dim=-1)
    def gaussian_prob(self, x, mu, sigma):
        dist = self.norm(x - mu)
        return torch.exp(-(dist ** 2) / (2 * sigma ** 2))
    def _estimate(self, x_t, p_x_t):
        # diffusion proability, P = N(x_t; \sqrt{α_t}x,1-α_t)
        mu = self.x * (self.alpha ** 0.5)
        sigma = (1 - self.alpha) ** 0.5
        p_diffusion = self.gaussian_prob(x_t, mu, sigma.to(x_t.device))

        # estimate the origin
        prob = (self.fitness + 1e-9) * (p_diffusion + 1e-9) / (p_x_t + 1e-9)
        z = torch.sum(prob)
        origin = torch.sum(prob.reshape(-1,1) * self.x, dim=0) / (z + 1e-9)
        return origin
    def estimate(self, x_t):
        p_x_t = self.density(x_t).to(x_t.device)
        origin = torch.vmap(self._estimate, (0, 0))(x_t, p_x_t)
        # size: (n, d) n population size, d dimension
        return origin

    def __call__(self, x_t):
        return self.estimate(x_t)

    def __repr__(self):
        return f'<BayesianEstimator {len(self.x)} samples>'



def diffusion_evolution_estimator_grad(target_func, xt, alpha,device="cuda"):
    # x: (n, d) n population size, d dimension
    x=xt.detach().clone()
    fitness = target_func(x).to(device)  # #[n,]
    estimator = BayesianEstimator(x, fitness, alpha)
    x0_est = estimator(x).to(device)# [n,d]
    grad_x=-(x-x0_est*alpha ** 0.5)/(1-alpha)
    return grad_x.reshape(-1) # [n*d,]



##############################################################################################################
if __name__ == "__main__":
    from graph_matrix_utils import natural_connectivity_sparse
    import networkx as nx
    import torch_geometric as tg
    import math
    import time
    # nodes_list = [1, 2, 3, 4, 5, 6]
    # edges_list = [(1, 2), (2, 3), (2, 5), (3, 4), (3, 5), (5, 6)]
    # G = nx.Graph()
    # G.add_nodes_from(nodes_list)
    # G.add_edges_from(edges_list)
    file = "dataset/dataset_real/cfinder-google.graphml"
    G = nx.read_graphml(file)

    print("G:", G)
    G.remove_edges_from(nx.selfloop_edges((G)))
    G.remove_nodes_from(list(nx.isolates(G)))
    print("G:", G)
    device = "cpu"
    tensor_G = tg.utils.from_networkx(G).to(device)
    tensor_G=tensor_G.coalesce()
    print("tensor_G:", tensor_G.edge_index)

    adj_matrix = torch.sparse_coo_tensor(indices=tensor_G.edge_index,
                                         values=torch.ones(tensor_G.edge_index.size(1)).to(device),
                                         size=(tensor_G.num_nodes, tensor_G.num_nodes), dtype=torch.float).coalesce().to(
        device)
    print("adj_matrix:", adj_matrix.indices())
