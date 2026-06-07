import copy
import functools
import gc
import torch_geometric.nn as gnn
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.distributions import Categorical

from .diffusion_schedulers import CategoricalDiffusion, InferenceSchedule, GaussianDiffusion, AdaptiveSchedule, \
    DiscreteFlowDiffusion
from torch_sparse import SparseTensor
from torch_geometric.data import Batch,Data
from utils.utils import load_graph_model
from .diffusion_model_utils import fvs_decode_np, guided_cost, select_subgraphs, model_inference, model_target_update
from .gradient_utils import fitness_map, diffusion_evolution_estimator_grad, get_guided_rates
import os
from .sde_lib import VPSDE



class Diffusion_model_CO(nn.Module):
    def __init__(self,
                 net_param,GNNEncoder,
                ):
        super(Diffusion_model_CO, self).__init__()

        self.diffusion_type = net_param["diffusion_type"]
        self.diffusion_schedule = net_param["diffusion_schedule"]
        self.diffusion_steps = net_param["diffusion_steps"]
        self.inference_diffusion_steps=net_param["inference_diffusion_steps"]
        self.inference_schedule=net_param["inference_schedule"]
        self.probability_flow=True
       
        self.timewarping_set=net_param["timewarping_set"] if net_param.get("timewarping_set") else False#
        self.loss_weight_schedule=net_param["loss_weight_schedule"] if net_param.get("loss_weight_schedule") and not self.timewarping_set else "constant"
        self.parallel_sampling = net_param["parallel_sampling"]
        self.sequential_sampling = net_param["sequential_sampling"]
        self.sparse = True
        self.target_snr = 0.1


        if self.timewarping_set:
            self.scheduler_train=AdaptiveSchedule(tmin=1, tmax=self.diffusion_steps,
                                                  mu=(self.diffusion_steps+1)/2,
                                                  sigma=(self.diffusion_steps-1)/2,
                                                  height=2,
                                                  offset=-0.5
                                                  )

        if self.diffusion_type == 'gaussian':
            self.inference_trick = net_param["inference_trick"] if net_param.get("inference_trick") else "ddim"
            out_channels = 1
            self.diffusion = GaussianDiffusion(
                T=self.diffusion_steps, schedule=self.diffusion_schedule,
                loss_weight_schedule=self.loss_weight_schedule)
            self.predictor_type = net_param["predictor_type"] if net_param.get("predictor_type") else "no_predictor"
            self.corrector_type=net_param["corrector_type"] if net_param.get("corrector_type") else "no_corrector"
            self.corrector_steps=net_param["corrector_steps"] if net_param.get("corrector_steps") else 1
            if net_param.get("sde_type") :
                self.sde_type = net_param["sde_type"]
                if self.sde_type == 'VPSDE':
                    self.sde = VPSDE(diffusion=self.diffusion, N=self.inference_diffusion_steps)

        elif self.diffusion_type == 'categorical':
            out_channels = 2
            self.diffusion = CategoricalDiffusion(
                T=self.diffusion_steps, schedule=self.diffusion_schedule,
                loss_weight_schedule=self.loss_weight_schedule)
        elif self.diffusion_type == "discretFlow":
            out_channels = 2
            self.diffusion = DiscreteFlowDiffusion(T=self.diffusion_steps,loss_weight_schedule=self.loss_weight_schedule,discrete_classes=out_channels)
            self.use_tag=net_param["use_tag"] if net_param.get("use_tag") else False

        else:
            raise ValueError(f"Unknown diffusion type {self.diffusion_type}")

        self.model = GNNEncoder(net_param=net_param["GNNEncoder_param"],out_channels=out_channels)

    def discretFlow_posterior(self, target_t, t, x1_probs,x1_probs_at_xt, xt):
        dt=target_t-t
        S=2
        if t + dt < 1.0:
            N = 1
        else:
            N = 0

            # Calculate the off-diagonal step probabilities
        step_probs = (
                dt * ((1 + N + N * (S - 1) * t) / (1 - t)) * x1_probs +
                dt * N * x1_probs_at_xt
        ).clamp(max=1.0)  # (num_nodes, S=2)

        # Calculate the on-diagnoal step probabilities
        # 1) Zero out the diagonal entries
        step_probs.scatter_(-1, xt[:,  None], 0.0)
        # 2) Calculate the diagonal entries such that the probability row sums to 1
        step_probs.scatter_(-1, xt[:,  None], (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0))
        if target_t<1.0:
            xt = Categorical(step_probs).sample()  # (num_nodes,)
        else:
            xt=step_probs[...,1].reshape(-1).clamp(min=0.0,max=1.0)
        return xt.detach()

    def categorical_posterior(self, target_t, t, x0_pred_prob, xt):
        """Sample from the categorical posterior for a given time step.
           See https://arxiv.org/pdf/2107.03006.pdf for details.
        """
        diffusion = self.diffusion

        if target_t is None:
            target_t = t - 1
        else:
            target_t = target_t.view(1)

        # Thanks to Daniyar and Shengyu, who found the "target_t == 0" branch is not needed :)
        # if target_t > 0:
        Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
        Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)
        # else:
        #   Q_t = torch.eye(2).float().to(x0_pred_prob.device)
        Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device)
        Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device)

        xt = F.one_hot(xt.long(), num_classes=2).float()
        xt = xt.reshape(x0_pred_prob.shape)

        x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous())
        x_t_target_prob_part_2 = Q_bar_t_target[0]
        x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True)

        x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3

        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]
        x_t_target_prob_part_2_new = Q_bar_t_target[1]
        x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True)

        x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / x_t_target_prob_part_3_new

        sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]

        if target_t > 0:
            xt = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
        else:
            xt = sum_x_t_target_prob.clamp(min=0)

        if self.sparse:
            xt = xt.reshape(-1)
        return xt.detach()

    def gaussian_posterior(self, target_t, t, pred, xt):
        """Sample (or deterministically denoise) from the Gaussian posterior for a given time step.
           See https://arxiv.org/pdf/2010.02502.pdf for details.
        """
        diffusion = self.diffusion
        if target_t is None:
            target_t = t - 1
        else:
            target_t = target_t

        atbar = diffusion.alphabar[t]
        atbar_target = diffusion.alphabar[target_t]

        if self.inference_trick =="ddpm"or t <= 1:
            # Use DDPM posterior
            at = diffusion.alpha[t]
            z = torch.randn_like(xt)
            atbar_prev = diffusion.alphabar[t - 1]
            beta_tilde = diffusion.beta[t - 1] * (1 - atbar_prev) / (1 - atbar)

            xt_target = (1 / np.sqrt(at)).item() * (xt - ((1 - at) / np.sqrt(1 - atbar)).item() * pred)
            xt_target = xt_target + np.sqrt(beta_tilde).item() * z
        elif self.inference_trick == 'ddim':
            xt_target = np.sqrt(atbar_target / atbar).item() * (xt - np.sqrt(1 - atbar).item() * pred)
            xt_target = xt_target + np.sqrt(1 - atbar_target).item() * pred
        else:
            raise ValueError('Unknown inference trick {}'.format(self.args.inference_trick))
        return xt_target.detach()
    def guided_categorical_posterior(self, target_t, t, x0_pred_prob, xt, grad=None,guided_weight=1.0):
      # xt: b* n, 1
      if grad is None:
        grad = xt.grad
      with torch.no_grad():
        diffusion = self.diffusion
        if target_t is None:
          target_t = t - 1
        else:
          target_t = torch.tensor(target_t).view(1)

        if target_t > 0:
          Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
          Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)  # [2, 2], transition matrix
        else:
          Q_t = torch.eye(2).float().to(x0_pred_prob.device)
        Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device)
        Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device)

        xt_grad_zero, xt_grad_one = torch.zeros(xt.shape, device=xt.device).unsqueeze(-1).repeat(1, 1, 1, 2), \
          torch.zeros(xt.shape, device=xt.device).unsqueeze(-1).repeat(1, 1, 1, 2)
        xt_grad_zero[..., 0] = (1 - xt) * grad
        xt_grad_zero[..., 1] = -xt_grad_zero[..., 0]
        xt_grad_one[..., 1] = xt * grad
        xt_grad_one[..., 0] = -xt_grad_one[..., 1]
        xt_grad = xt_grad_zero + xt_grad_one

        xt = F.one_hot(xt.long(), num_classes=2).float()
        xt = xt.reshape(x0_pred_prob.shape)  # [1, b*n, 1, 2]

        # q(xt−1|xt,x0=0)pθ(x0=0|xt)
        x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous())
        x_t_target_prob_part_2 = Q_bar_t_target[0]
        x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True)

        x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3  # [b, n, n, 2]

        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]

        # q(xt−1|xt,x0=1)pθ(x0=1|xt)
        x_t_target_prob_part_2_new = Q_bar_t_target[1]
        x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True)

        x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / x_t_target_prob_part_3_new

        sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]

        p_theta = torch.cat((1 - sum_x_t_target_prob.unsqueeze(-1), sum_x_t_target_prob.unsqueeze(-1)), dim=-1)
        p_phi = torch.exp(-guided_weight*xt_grad)
        if self.sparse:
          p_phi = p_phi.reshape(p_theta.shape)
        posterior = (p_theta * p_phi) / torch.sum((p_theta * p_phi), dim=-1, keepdim=True)

        # plt.clf()
        # sns.heatmap(posterior[..., 1].clamp(0, 1).float().detach().cpu()[0])
        # plt.savefig('imgs/guided_{}.png'.format(target_t[0]))

        if target_t > 0:
          xt = torch.bernoulli(posterior[..., 1].clamp(0, 1))
        else:
          xt = posterior[..., 1].clamp(min=0)
        if self.sparse:
          xt = xt.reshape(-1)
        return xt

    def predict_correct_gaussian_posterior(self, target_t, t, epsilon_pred, xt,score_fn=None):
        """
        Predictor-Corrector (PC) sampling:
        see https://arxiv.org/abs/2011.13456 for details.
        """
        atbar = torch.tensor(self.diffusion.alphabar[t]).float().to(epsilon_pred.device)
        atbar_target = torch.tensor(self.diffusion.alphabar[target_t]).float().to(epsilon_pred.device)
        alphat = torch.tensor(self.diffusion.alpha[t]).float().to(epsilon_pred.device)

        if self.predictor_type == "no_predictor" and self.corrector_type == "no_corrector":
            raise ValueError("No predictor and corrector are selected.")
        # predictor:
        if self.predictor_type == "discretize_reverse_diffusion":
            score_xt = -epsilon_pred / torch.sqrt(1 - atbar)
            #DDPM discretize reverse diffusion
            f, G = self.sde.discretize(xt, t)
            dt=t-target_t #t>target_t
            rev_f = f - G ** 2*score_xt * (0.5 if self.probability_flow else 1.)*dt
            rev_G = torch.zeros_like(G) if self.probability_flow else G
            z = torch.randn_like(xt)
            xt=xt-rev_f+rev_G*z
        elif self.predictor_type == "diffusion_sampling":
            xt_target = torch.sqrt(atbar_target / atbar).item() * (xt - torch.sqrt(1 - atbar).item() * epsilon_pred)
            xt = xt_target + np.sqrt(1 - atbar_target).item() * epsilon_pred
        elif self.predictor_type == "no_predictor":
            xt = xt
        else:
            raise ValueError("Unknown predictor type {}".format(self.predictor_type))
        # corrector: annealed Langevin dynamics MCMC
        if self.corrector_type == "annealed_langevin_dynamics":
            for i in range(self.corrector_steps):
                if i == 0:
                    epsilon_pred = epsilon_pred.detach()
                else:
                    if score_fn is None:
                        raise ValueError("No score function is provided for corrector.")
                    epsilon_pred = score_fn(xt.float())

                score_xt = -epsilon_pred / torch.sqrt(1 - atbar)
                score_xt_norm = nn.functional.normalize(score_xt, p=2, dim=-1)
                noise=torch.randn_like(xt)
                noise_norm = nn.functional.normalize(noise, p=2, dim=-1)
                step_size = (self.target_snr * noise_norm / score_xt_norm) ** 2 * 2 * alphat
                x_mean = xt + step_size* score_xt
                xt = x_mean + noise*torch.sqrt(step_size * 2)
        elif self.corrector_type == "no_corrector":
            xt=xt
        else:
            raise ValueError("Unknown corrector type {}".format(self.corrector_type))

        return xt.detach()
    def duplicate_edge_index(self,parallel_sampling, edge_index, num_nodes, device):
        """Duplicate the edge index (in sparse graphs) for parallel sampling."""
        edge_index = edge_index.reshape((2, 1, -1))
        edge_index_indent = torch.arange(0, parallel_sampling).view(1, -1, 1).to(device)
        edge_index_indent = edge_index_indent * num_nodes
        edge_index = edge_index + edge_index_indent
        edge_index = edge_index.reshape((2, -1))
        return edge_index

    def duplicate_batch_index(self,parallel_sampling, batch_index, num_graphs, num_nodes, device):
        """Duplicate the batch index (in batch-based graphs) for parallel sampling."""
        if batch_index is not None:
            batch_index = batch_index.reshape((1, 1, -1))

            batch_index_indent = torch.arange(0, parallel_sampling).view(1, -1, 1).to(device)

            batch_index_indent = batch_index_indent * num_graphs
            batch_index = batch_index + batch_index_indent
            batch_index = batch_index.reshape(-1)
        else:
            batch_index = torch.arange(0, parallel_sampling).view(-1).to(device)
            batch_index = batch_index.repeat_interleave(repeats=num_nodes)

        return batch_index.to(device)
    def get_ODE_sample(self):
        """Sample from the ODE solution for a given time step."""
        pass


class FVSModel(Diffusion_model_CO):
  def __init__(self,
               net_param,GNNEncoder,):
    super(FVSModel, self).__init__(net_param=net_param,GNNEncoder=GNNEncoder)
    self.norm=net_param["norm"]
    self.rewrite=net_param["rewrite"]
    self.rewrite_steps=net_param["rewrite_steps"]
    self.rewrite_ratio=net_param["rewrite_ratio"]
    self.infer_direction=net_param["infer_direction"]
    self.infer_direction_params={"MVC_P": net_param.get("MVC_P") ,
                                }
    self.gradient_estimator_type=net_param["gradient_estimator_type"] if net_param.get("gradient_estimator_type") else "cost_estimator"
    self.target_weight = net_param["target_weight"] if net_param.get("target_weight") else 1

    if net_param["infer_direction"]!="MVC" and self.gradient_estimator_type=="cost_estimator":

        model_path = os.path.join(net_param["guided_model_path"],"{}_metrics_model_trained".format(net_param["infer_direction"]))
        self.guided_model,_ = load_graph_model(model_path,device="cuda")
        self.infer_direction_params["guided_model"]=self.guided_model
    else:
        self.guided_model=None

    if self.gradient_estimator_type=="CEP_estimator":
        model_path = os.path.join(net_param["CEP_model_path"],
                                  "{}_metrics_model_trained".format(net_param["infer_direction"]))
        self.CEP_model, _ = load_graph_model(model_path, device="cuda")
    else:
        self.CEP_model=None


  def forward(self, x, t, edge_index,batch_index=None):
      return self.model(x, t, edge_index=edge_index,batch_index=batch_index)

  def categorical_training_step(self,data, device):
      ptr=data.ptr

      point_indicator=ptr[1:]-ptr[:-1]
      graph_data=data
      #print("graph nodes:{}".format(data.num_nodes))

      node_labels = graph_data.y
      edge_index = graph_data.edge_index
      batch_index=graph_data.batch

      # Sample from diffusion
      node_labels_onehot = F.one_hot(node_labels.long(), num_classes=2).float()
      #[节点数，1,1,2]
      node_labels_onehot = node_labels_onehot.unsqueeze(1).unsqueeze(1)
      #print("node_labels_onehot",node_labels_onehot.shape)
      if self.timewarping_set:
          timesteps = self.scheduler_train.sample((point_indicator.shape[0],))#[batch_size]
          timesteps=torch.floor(timesteps).long()
      else:
          timesteps = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
          timesteps = torch.from_numpy(timesteps).long()#[batch_size]

      t = timesteps.repeat_interleave(point_indicator.cpu().reshape(-1), dim=0).numpy()#[总节点数]

      loss_weight = self.diffusion.get_loss_weights(t).reshape(-1).to(device)

      xt = self.diffusion.sample(node_labels_onehot, t)
      xt = xt * 2 - 1
      xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

      t = torch.from_numpy(t).float()
      # [总节点数]
      t = t.reshape(-1)
     # print("t shape:{}".format(t.shape))
      # [节点数] 可视为加噪声后的标签值
      xt = xt.reshape(-1)
     # print("xt shape;{}".format(xt.shape))
      edge_index = edge_index.to(device).reshape(2, -1)

      # Denoise
      x0_pred = self.forward(
          xt.float().to(device),
          t.float().to(device),
          edge_index,
          batch_index
      )
      loss = F.cross_entropy(x0_pred, node_labels.long().to(device),reduction="none")
      weight_loss = loss * loss_weight

      all_loss = weight_loss.mean()
      if self.timewarping_set:
          per_loss = gnn.global_mean_pool(weight_loss, batch_index).cpu().detach()  # [batch_size]
          with torch.no_grad():
              self.scheduler_train.add_data(
                  per_loss,
                  timesteps
              )
      return all_loss



  def gaussian_training_step(self, data, device):
    ptr = data.ptr

    point_indicator = ptr[1:] - ptr[:-1]
    graph_data = data

    node_labels = graph_data.y
    edge_index = graph_data.edge_index
    batch_index = graph_data.batch


    # Sample from diffusion
    node_labels = node_labels.float() * 2 - 1
    node_labels = node_labels * (1.0 + 0.05 * torch.rand_like(node_labels))
    node_labels = node_labels.unsqueeze(1).unsqueeze(1)

    if self.timewarping_set:
        timesteps = self.scheduler_train.sample((point_indicator.shape[0],))  # [batch_size]

        timesteps = torch.floor(timesteps).long()
    else:
        timesteps = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
        timesteps = torch.from_numpy(timesteps).long()  # [batch_size]

    t = timesteps.repeat_interleave(point_indicator.reshape(-1).cpu(), dim=0).numpy()
    loss_weight = self.diffusion.get_loss_weights(t).reshape(-1).to(device)

    xt, epsilon = self.diffusion.sample(node_labels, t)

    t = torch.from_numpy(t).float()
    t = t.reshape(-1)
    xt = xt.reshape(-1)
    edge_index = edge_index.to(device).reshape(2, -1)
    epsilon = epsilon.reshape(-1)

    # Denoise
    epsilon_pred = self.forward(
        xt.float().to(device),
        t.float().to(device),
        edge_index,
        batch_index
    )
    epsilon_pred = epsilon_pred.reshape(-1)
    loss=F.mse_loss(epsilon_pred,epsilon.float(),reduction="none")

    weight_loss=loss*loss_weight

    all_loss=weight_loss.mean()

    if self.timewarping_set:
        per_loss = gnn.global_mean_pool(weight_loss, batch_index).cpu().detach()  # [batch_size]

        with torch.no_grad():
            self.scheduler_train.add_data(
                per_loss,
                timesteps
            )

    return all_loss

  def discretFlow_training_step(self, data, device):
      ptr = data.ptr

      point_indicator = ptr[1:] - ptr[:-1]
      graph_data = data
      # print("graph nodes:{}".format(data.num_nodes))

      node_labels = graph_data.y
      edge_index = graph_data.edge_index
      batch_index = graph_data.batch
      if self.timewarping_set:
          timesteps = self.scheduler_train.sample((point_indicator.shape[0],))#[batch_size]
          timesteps=torch.floor(timesteps).long()
      else:
          timesteps = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
          timesteps = torch.from_numpy(timesteps).long()#[batch_size]

      t = timesteps.repeat_interleave(point_indicator.cpu().reshape(-1), dim=0).numpy()#[总节点数]
      loss_weight = self.diffusion.get_loss_weights(t).reshape(-1).to(device)
      xt = self.diffusion.sample(node_labels, t)

      t = torch.from_numpy(t).float()
      # [总节点数]
      t = t.reshape(-1)
      # print("t shape:{}".format(t.shape))
      # [节点数] 可视为加噪声后的标签值
      xt = xt.reshape(-1)
      # print("xt shape;{}".format(xt.shape))
      edge_index = edge_index.to(device).reshape(2, -1)

      # Denoise
      x0_pred = self.forward(
          xt.float().to(device),
          t.float().to(device),
          edge_index,
          batch_index
      )
      loss = F.cross_entropy(x0_pred, node_labels.long().to(device), reduction="none")
      weight_loss = loss * loss_weight

      all_loss = weight_loss.mean()
      if self.timewarping_set:
          per_loss = gnn.global_mean_pool(weight_loss, batch_index).cpu().detach()  # [batch_size]
          with torch.no_grad():
              self.scheduler_train.add_data(
                  per_loss,
                  timesteps
              )
      return all_loss
  def training_step(self, batch, device):
    if self.diffusion_type == 'gaussian':
        return self.gaussian_training_step(batch, device)
    elif self.diffusion_type == 'categorical':
        return self.categorical_training_step(batch, device)
    elif self.diffusion_type == 'discretFlow':
        return self.discretFlow_training_step(batch, device)
    else:
        raise ValueError("diffusion_type {} is not supported!".format(self.diffusion_type))

  def categorical_denoise_step(self, xt, t, device, edge_index=None, target_t=None,batch_index=None):
    with torch.no_grad():
      xt_scale = xt * 2 - 1
      xt_scale = xt_scale * (1.0 + 0.05 * torch.rand_like(xt.float(), device=device))

      #[节点数，2]
      x0_pred = self.forward(
        xt_scale,
        t.float().to(device),
        edge_index.long().to(device) if edge_index is not None else None,
        batch_index
      )
      x0_pred_prob = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
      xt= self.categorical_posterior(target_t, t, x0_pred_prob, xt)
      return xt.detach().to(device)

  def discretFlow_denoise_step(self, xt, t, device, edge_index=None, target_t=None,batch_index=None):
    with torch.no_grad():
      t=self.diffusion.discreteflow_time(t)
      target_t=self.diffusion.discreteflow_time(target_t)

      #[num_nodes，2]
      x1_pred = self.forward(
        xt.float().to(device),
        t.float().to(device),
        edge_index.long().to(device) if edge_index is not None else None,
        batch_index
      )
      x1_pred_prob = x1_pred.softmax(dim=-1)#[Numnodes,2]
      x1_probs_at_xt = torch.gather(x1_pred_prob, -1, xt[:, None])#[num_nodes,1]
      xt= self.discretFlow_posterior( target_t=target_t, t=t, x1_probs=x1_pred_prob,
                                     x1_probs_at_xt=x1_probs_at_xt, xt=xt)
      return xt.detach().to(device)


  def gaussian_denoise_step(self, xt, t, device, edge_index=None, target_t=None,batch_index=None):
    with torch.no_grad():

      pred = self.forward(
          xt.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
          batch_index
      )
      pred = pred.squeeze(1)
      xt = self.gaussian_posterior(target_t, t, pred, xt)
      return xt.detach().to(device)

  def guided_discreteFlow_denoise_step(self, xt, t, edge_index, device, graph_name=None, target_t=None, batch_index=None):
      t=self.diffusion.discreteflow_time(t)
      target_t=self.diffusion.discreteflow_time(target_t)
      dt=target_t-t

      x1_pred = self.forward(
          xt.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
          batch_index
      )
      x1_pred_prob = x1_pred.softmax(dim=-1)  # [parallel_sampling_num*n,2]
      x1=Categorical(probs=x1_pred_prob).sample()#[parallel_sampling_num*n,]

      R_t=self.diffusion.get_rate_matrix(xt=xt,x1=x1,t=t)#[parallel_sampling_num*n,2]
      xt=xt.reshape(self.parallel_sampling, -1)  # [parallel_sampling_num,n]
      R_t=R_t.reshape(self.parallel_sampling, -1,2)  # [parallel_sampling_num,n,2]
      fitness_map = self.infer_direction_params["fitness_map"]
      if not self.use_tag:

          predictor_log_prob=fitness_map.log_prob
      else:
          predictor_log_prob=fitness_map.differentiable_log_prob
      guided_R_t = get_guided_rates(
          predictor_log_prob=predictor_log_prob,
          xt=xt,
          t=t.to(device),
          R_t=R_t,
          S=2,
          use_tag=self.use_tag,
          guide_temp=1/self.target_weight,

      )

      guided_R_t.scatter_(-1, xt[:, :, None], 0.0)
      guided_R_t.scatter_(-1, xt[:, :, None], (-R_t.sum(dim=-1, keepdim=True)))

      # Obtain probabilities from the rates
      step_probs = (guided_R_t * dt).clamp(min=0.0, max=1.0)
      step_probs.scatter_(-1, xt[:, :, None], 0.0)
      step_probs.scatter_(
          -1,
          xt[:, :, None],
          (1.0 - torch.sum(step_probs, dim=-1, keepdim=True)).clamp(min=0.0),
      )
      step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
      step_probs=step_probs.reshape(-1, 2)#[parallel_sampling_num*n,2]
      if target_t < 1.0:
          xt = Categorical(step_probs).sample()  # (num_nodes,)
      else:
          xt = step_probs[..., 1].reshape(-1).clamp(min=0.0, max=1.0)
      return xt.detach().to(device)
  def guided_categorical_denoise_step(self, xt, t,edge_index, device,graph_name=None,  target_t=None,batch_index=None):
    # torch.set_grad_enabled(True)
    xt = xt.float()  # [B*n,] if sparse
    t = t

    #目标函数可微情况下求xt对目标函数的梯度信息
    if self.gradient_estimator_type=="cost_estimator":
        with torch.inference_mode(False):
            xt.requires_grad = True
            xt_scale = xt * 2 - 1

            xt_scale = xt_scale * (1.0 + 0.05 * torch.rand_like(xt.float(), device=device))
            x0_pred = self.forward(
                xt_scale,
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
                batch_index
              )

            x0_pred_prob = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
            pred_nodes = x0_pred_prob[..., 1].reshape(-1)#[N,]
            cost_est=guided_cost(pred_nodes,edge_index,device=device,infer_direction=self.infer_direction,
                                   **self.infer_direction_params).squeeze()
            cost_est.requires_grad_(True)
            cost_est.backward()
            assert xt.grad is not None
            grad_xt = xt.grad.clone()#[b*n,]
    # 目标函数不可微情况下 近似求xt对目标函数的梯度
    else:
        raise ValueError("gradient_estimator_type {} is not supported!".format(self.gradient_estimator_type))


    if self.norm is True:
        grad_xt = nn.functional.normalize(grad_xt, p=2, dim=-1)
    xt = self.guided_categorical_posterior(target_t, t, x0_pred_prob, xt,grad=grad_xt,guided_weight=self.target_weight)

    return xt.detach().to(device)

  def guided_gaussian_denoise_step(self, xt, t,edge_index, device,graph_name=None,  target_t=None,batch_index=None):
    """
    参考“Score-Based Generative Modeling through Stochastic Differential Equations“”
    “Diffusion Models as Constrained Samplers for Optimization with Unknown Constraints"
    将模型预测生成的噪声转换为对训练数据分布的梯度信息，然后融入目标函数梯度信息，生成样本
    采样方式采取PC采样，即预测-纠正采样
    在预测步骤中，根据扩散方程生成预测样本
    在纠正步骤中，使用朗之万动力学算法纠正预测样本
    """

    #torch.set_grad_enabled(True)
    xt = xt.float()  # [B*n,] if sparse
    t = t.view(1)
    update_epsilon=self.target_score_gaussian( xt, t,edge_index, device,graph_name,batch_index)
    if self.corrector_type == "annealed_langevin_dynamics":
        score_fn=functools.partial(self.target_score_gaussian,
                                   t=t,edge_index=edge_index, device=device, graph_name=graph_name, batch_index=batch_index)
    else:
        score_fn=None
    xt = self.predict_correct_gaussian_posterior(target_t, t, update_epsilon, xt,score_fn)

    return xt.detach().to(device)



  def decode_step(self, batch, device):

      if self.diffusion_type=="gaussian" or self.diffusion_type=="categorical":
        return self.decode_test_step(batch, device)
      elif self.diffusion_type=="discretFlow":
        return self.discretFlow_decode_test_step(batch,device)

  def heatmap_pred_step(self, batch, device):
      if self.diffusion_type == "gaussian" or self.diffusion_type == "categorical":
          return self.pred_test_step(batch, device)
      elif self.diffusion_type == "discretFlow":
          return self.discretFlow_pred_test_step( batch, device)
  def decode_test_step(self,data,device):
    if isinstance(data, Batch):
        ptr = data.ptr
        point_indicator = ptr[1:] - ptr[:-1]
        batch_index=data.batch
        num_graphs=point_indicator.shape[0]
    else:
        batch_index=None
        num_graphs=1
    if data.get("graph_name"):
        graph_name = data.graph_name
    else:
        graph_name = None
    graph_data = copy.deepcopy(data)
    node_labels = graph_data.y
    edge_index = graph_data.edge_index
    num_nodes=graph_data.num_nodes

    stacked_predict_labels = []
    edge_index = edge_index.to(device).reshape(2, -1)


    if self.parallel_sampling > 1:

      edge_index = self.duplicate_edge_index(self.parallel_sampling, edge_index,num_nodes, device)
      batch_index=self.duplicate_batch_index(self.parallel_sampling, batch_index,
                                             num_graphs=num_graphs,
                                             num_nodes=num_nodes,
                                             device=device)

    self.infer_direction_params["fitness_map"] = fitness_map(
                diffusion_type=self.diffusion_type,fitness_func_type=self.infer_direction,
                                                             tensor_G=graph_data.to(device),
                                                             temperature=float(self.diffusion_steps),
                )


    with torch.no_grad():
        for _ in range(self.sequential_sampling):
          xt = torch.randn(num_nodes).to(device)
          if self.parallel_sampling > 1:
            xt = xt.repeat(self.parallel_sampling)
            xt = torch.randn_like(xt)

          if self.diffusion_type == 'gaussian':
              xt.requires_grad = True #??
          elif self.diffusion_type == 'categorical':
             xt = (xt > 0).long()
          xt = xt.reshape(-1).to(device)

          batch_size=1
          steps = self.inference_diffusion_steps
          time_schedule = InferenceSchedule(inference_schedule=self.inference_schedule,
                                            T=self.diffusion.T, inference_T=steps)

          for i in range(steps):
            t1, t2 = time_schedule(i)
            t1 = torch.tensor([t1 ]).int()
            t2 = torch.tensor([t2 ]).int()

            if self.diffusion_type == 'gaussian':
                xt = self.gaussian_denoise_step(xt, t1, device, edge_index, target_t=t2, batch_index=batch_index)
            elif self.diffusion_type == 'categorical':
                xt = self.categorical_denoise_step(xt, t1, device, edge_index, target_t=t2,batch_index=batch_index)
          if self.diffusion_type == 'gaussian':
            predict_labels = xt.float().cpu().detach().numpy() * 0.5 + 0.5
          elif self.diffusion_type == 'categorical':
            predict_labels = xt.float().cpu().detach().numpy() + 1e-6

          stacked_predict_labels.append(predict_labels)

        predict_labels = np.concatenate(stacked_predict_labels, axis=0)
        all_sampling = self.sequential_sampling * self.parallel_sampling

        splitted_predict_labels = np.split(predict_labels, all_sampling)#
        solved_solutions = [fvs_decode_np(predict_labels, graph_data) for predict_labels in splitted_predict_labels]#
        solved_costs = [solved_solution.sum() for solved_solution in solved_solutions]
        best_solved_cost = np.min(solved_costs)
        best_solved_id = np.argmin(solved_costs)
        best_solution=solved_solutions[best_solved_id]

        if node_labels is not None:
            gt_cost = node_labels.cpu().numpy().mean()
        else:
            gt_cost = None

        guided_gap, g_best_solved_cost = -1, num_nodes# or num_nodes?
        if self.rewrite:
          g_best_solution = best_solution
          for _ in range(self.rewrite_steps):
            g_stacked_predict_labels = []
            g_x0 = torch.from_numpy(g_best_solution).to(device)
            if self.parallel_sampling > 1:
              g_x0 = g_x0.repeat(self.parallel_sampling)

            steps_T = int(self.diffusion_steps * self.rewrite_ratio)
            if self.diffusion_type == 'gaussian':
                g_xt = self.gaussian_x0_to_xt(g_x0, target_t=steps_T)
            elif self.diffusion_type == 'categorical':
                g_xt,_ = self.categorical_x0_to_xt(g_x0, target_t=steps_T)
            g_xt=g_xt.to(device)
            steps_inf = int(self.inference_diffusion_steps * self.rewrite_ratio)
            time_schedule = InferenceSchedule(inference_schedule=self.inference_schedule,
                                              T=steps_T, inference_T=steps_inf)
            for i in range(steps_inf):
              t1, t2 = time_schedule(i)
              t1 = torch.tensor([t1 ]).int()
              t2 = torch.tensor([t2 ]).int()

              if self.diffusion_type == 'gaussian':
                  g_xt = self.guided_gaussian_denoise_step(g_xt, t1, edge_index,device,
                                                          graph_name=graph_name, target_t=t2,batch_index=batch_index)
              elif self.diffusion_type == 'categorical':
                  g_xt = self.guided_categorical_denoise_step(g_xt, t1,  edge_index,device,
                                                          graph_name=graph_name, target_t=t2,batch_index=batch_index)

            if self.diffusion_type == 'gaussian':
                g_predict_labels = g_xt.float().cpu().detach().numpy() * 0.5 + 0.5
            elif self.diffusion_type == 'categorical':
                g_predict_labels = g_xt.float().cpu().detach().numpy() + 1e-6
            g_stacked_predict_labels.append(g_predict_labels)
            g_predict_labels = np.concatenate(g_stacked_predict_labels, axis=0)#

            g_splitted_predict_labels = np.split(g_predict_labels, self.parallel_sampling)
            g_solved_solutions = [fvs_decode_np(g_predict_labels, graph_data) for g_predict_labels in g_splitted_predict_labels]
            g_solved_costs = [g_solved_solution.sum() for g_solved_solution in g_solved_solutions]
            g_best_solved_cost_cur=np.min(g_solved_costs)
            if g_best_solved_cost_cur < g_best_solved_cost:
                g_best_solved_cost=g_best_solved_cost_cur
                g_best_solved_id = np.argmin(g_solved_costs)
                g_best_solution = g_solved_solutions[g_best_solved_id]

          return g_best_solution, node_labels, gt_cost, best_solved_cost/num_nodes, g_best_solved_cost/num_nodes
        else:
          return best_solution, node_labels, gt_cost, best_solved_cost/num_nodes, g_best_solved_cost/num_nodes


  def discretFlow_decode_test_step(self,data,device):
    if isinstance(data, Batch):
        ptr = data.ptr
        point_indicator = ptr[1:] - ptr[:-1]
        batch_index=data.batch
        num_graphs=point_indicator.shape[0]
    else:
        batch_index=None
        num_graphs=1
    if data.get("graph_name"):
        graph_name = data.graph_name
    else:
        graph_name = None
    graph_data = copy.deepcopy(data)
    node_labels = graph_data.y
    edge_index = graph_data.edge_index
    num_nodes=graph_data.num_nodes

    stacked_predict_labels = []
    edge_index = edge_index.to(device).reshape(2, -1)


    discretFlow_guided_model = functools.partial(guided_cost,
                                 edge_index=edge_index, device=device, infer_direction=self.infer_direction,
                                   **self.infer_direction_params)
    if self.parallel_sampling > 1:

      edge_index = self.duplicate_edge_index(self.parallel_sampling, edge_index,num_nodes, device)
      batch_index=self.duplicate_batch_index(self.parallel_sampling, batch_index,
                                             num_graphs=num_graphs,
                                             num_nodes=num_nodes,
                                             device=device)

    self.infer_direction_params["fitness_map"] = fitness_map(
        diffusion_type=self.diffusion_type, fitness_func_type=self.infer_direction,
        tensor_G=graph_data.to(device),
        temperature=float(self.diffusion_steps),

        guide_model=discretFlow_guided_model if self.gradient_estimator_type=="cost_estimator" else None,
        CEP_guide_model=self.CEP_model if self.gradient_estimator_type=="CEP_estimator" else None,)
    with torch.no_grad():
        for _ in range(self.sequential_sampling):
          xt = torch.randn(num_nodes).to(device)
          if self.parallel_sampling > 1:
            xt = xt.repeat(self.parallel_sampling)
            xt = torch.randn_like(xt)


          xt = (xt > 0).long()
          xt = xt.reshape(-1).to(device)

          batch_size=1
          steps = self.inference_diffusion_steps
          time_schedule = InferenceSchedule(inference_schedule=self.inference_schedule,
                                            T=self.diffusion.T, inference_T=steps)

          for i in range(steps):
            t1, t2 = time_schedule(i)

            t1 = torch.tensor([t1 ]).int().to(device)
            t2 = torch.tensor([t2 ]).int().to(device)


            xt = self.discretFlow_denoise_step(xt, t1, device, edge_index, target_t=t2,batch_index=batch_index)

          predict_labels = xt.float().cpu().detach().numpy() + 1e-6

          stacked_predict_labels.append(predict_labels)

        predict_labels = np.concatenate(stacked_predict_labels, axis=0)
        all_sampling = self.sequential_sampling * self.parallel_sampling

        splitted_predict_labels = np.split(predict_labels, all_sampling)#
        solved_solutions = [fvs_decode_np(predict_labels, graph_data) for predict_labels in splitted_predict_labels]#
        solved_costs = [solved_solution.sum() for solved_solution in solved_solutions]
        best_solved_cost = np.min(solved_costs)
        best_solved_id = np.argmin(solved_costs)
        best_solution=solved_solutions[best_solved_id]

        if node_labels is not None:
            gt_cost = node_labels.cpu().numpy().mean()
        else:
            gt_cost = None

        guided_gap, g_best_solved_cost = -1, num_nodes# or num_nodes?
        if self.rewrite:
          g_best_solution = best_solution
          for rewrite_step in range(self.rewrite_steps):

            g_stacked_predict_labels = []
            g_x0 = torch.from_numpy(g_best_solution).long().to(device)
            if self.parallel_sampling > 1:
              g_x0 = g_x0.repeat(self.parallel_sampling)

            steps_T = int(self.diffusion_steps * self.rewrite_ratio)

            g_xt = self.diffusion.sample(g_x0, t=np.array([steps_T])).to(device)

            steps_inf = int(self.inference_diffusion_steps * self.rewrite_ratio)
            time_schedule = InferenceSchedule(inference_schedule=self.inference_schedule,
                                              T=steps_T, inference_T=steps_inf)
            for i in range(steps_inf):

              t1, t2 = time_schedule(i)
              t1 = torch.tensor([t1 ]).int().to(device)
              t2 = torch.tensor([t2 ]).int().to(device)


              g_xt = self.guided_discreteFlow_denoise_step(g_xt, t1,  edge_index,device,
                                                          graph_name=graph_name, target_t=t2,batch_index=batch_index)

            g_predict_labels = g_xt.float().cpu().detach().numpy() + 1e-6
            g_stacked_predict_labels.append(g_predict_labels)
            g_predict_labels = np.concatenate(g_stacked_predict_labels, axis=0)#

            g_splitted_predict_labels = np.split(g_predict_labels, self.parallel_sampling)
            g_solved_solutions = [fvs_decode_np(g_predict_labels, graph_data) for g_predict_labels in g_splitted_predict_labels]
            g_solved_costs = [g_solved_solution.sum() for g_solved_solution in g_solved_solutions]
            g_best_solved_cost_cur=np.min(g_solved_costs)
            if g_best_solved_cost_cur < g_best_solved_cost:
                g_best_solved_cost=g_best_solved_cost_cur
                g_best_solved_id = np.argmin(g_solved_costs)
                g_best_solution = g_solved_solutions[g_best_solved_id]

          return g_best_solution, node_labels, gt_cost, best_solved_cost/num_nodes, g_best_solved_cost/num_nodes
        else:
          return best_solution, node_labels, gt_cost, best_solved_cost/num_nodes, g_best_solved_cost/num_nodes

  def discretFlow_pred_test_step(self, data, device):
      if isinstance(data, Batch):
          ptr = data.ptr
          point_indicator = ptr[1:] - ptr[:-1]
          batch_index = data.batch
          num_graphs = point_indicator.shape[0]
      else:
          batch_index = None
          num_graphs = 1
      if data.get("graph_name"):
          graph_name = data.graph_name
      else:
          graph_name = None
      graph_data = copy.deepcopy(data)
      node_labels = graph_data.y
      edge_index = graph_data.edge_index
      num_nodes = graph_data.num_nodes

      stacked_predict_labels = []
      edge_index = edge_index.to(device).reshape(2, -1)

      discretFlow_guided_model = functools.partial(guided_cost,
                                                   edge_index=edge_index, device=device,
                                                   infer_direction=self.infer_direction,
                                                   **self.infer_direction_params)
      if self.parallel_sampling > 1:
          edge_index = self.duplicate_edge_index(self.parallel_sampling, edge_index, num_nodes, device)
          batch_index = self.duplicate_batch_index(self.parallel_sampling, batch_index,
                                                   num_graphs=num_graphs,
                                                   num_nodes=num_nodes,
                                                   device=device)

      self.infer_direction_params["fitness_map"] = fitness_map(
          diffusion_type=self.diffusion_type, fitness_func_type=self.infer_direction,
          tensor_G=graph_data.to(device),
          temperature=float(self.diffusion_steps),

          guide_model=discretFlow_guided_model if self.gradient_estimator_type == "cost_estimator" else None,
          CEP_guide_model=self.CEP_model if self.gradient_estimator_type == "CEP_estimator" else None, )
      with torch.no_grad():
          for _ in range(self.sequential_sampling):
              xt = torch.randn(num_nodes).to(device)
              if self.parallel_sampling > 1:
                  xt = xt.repeat(self.parallel_sampling)
                  xt = torch.randn_like(xt)

              xt = (xt > 0).long()
              xt = xt.reshape(-1).to(device)

              batch_size = 1
              steps = self.inference_diffusion_steps
              time_schedule = InferenceSchedule(inference_schedule=self.inference_schedule,
                                                T=self.diffusion.T, inference_T=steps)

              for i in range(steps):
                  t1, t2 = time_schedule(i)
                  t1 = torch.tensor([t1]).int().to(device)
                  t2 = torch.tensor([t2]).int().to(device)

                  xt = self.discretFlow_denoise_step(xt, t1, device, edge_index, target_t=t2, batch_index=batch_index)

              predict_labels = xt.float().cpu().detach().numpy() + 1e-6

              stacked_predict_labels.append(predict_labels)

          predict_labels = np.concatenate(stacked_predict_labels, axis=0)
          all_sampling = self.sequential_sampling * self.parallel_sampling

          if node_labels is None:
              splitted_predict_labels = np.split(predict_labels, all_sampling)
          else:
              splitted_predict_labels = torch.from_numpy(predict_labels)
              node_labels = node_labels.repeat(all_sampling)
              self.rewrite = False

          if self.rewrite:
              solved_costs = [self.infer_direction_params["fitness_map"].fitness_func(
                  torch.from_numpy(predict_labels)).cpu().detach().numpy()
                              for predict_labels in splitted_predict_labels]  #

              g_best_solved_cost, best_solved_id = np.min(solved_costs), np.argmin(solved_costs)
              g_best_solution_probs = torch.from_numpy(splitted_predict_labels[best_solved_id]).to(device)
              g_best_solution = torch.bernoulli(g_best_solution_probs.clamp(0, 1)).to(device)
              g_stacked_predict_labels = []
              for rewrite_step in range(self.rewrite_steps):
                  g_x0 = g_best_solution.clone().long().to(device)
                  if self.parallel_sampling > 1:
                      g_x0 = g_x0.repeat(self.parallel_sampling)

                  steps_T = int(self.diffusion_steps * self.rewrite_ratio)
                  steps_T = np.array([steps_T])
                  g_xt = self.diffusion.sample(g_x0, t=steps_T).to(device)

                  steps_inf = int(self.inference_diffusion_steps * self.rewrite_ratio)
                  time_schedule = InferenceSchedule(inference_schedule=self.inference_schedule,
                                                    T=steps_T, inference_T=steps_inf)
                  for i in range(steps_inf):
                      t1, t2 = time_schedule(i)
                      t1 = torch.tensor([t1]).int().to(device)
                      t2 = torch.tensor([t2]).int().to(device)

                      g_xt = self.guided_discreteFlow_denoise_step(g_xt, t1, edge_index, device,
                                                                   graph_name=graph_name, target_t=t2,
                                                                   batch_index=batch_index)

                  g_predict_labels = g_xt.float().cpu().detach().numpy() + 1e-6
                  g_splitted_predict_labels = np.split(g_predict_labels, self.parallel_sampling)
                  g_stacked_predict_labels.extend(g_splitted_predict_labels)

                  g_solved_costs = [
                      self.infer_direction_params["fitness_map"].fitness_func(
                          torch.from_numpy(g_predict_labels)).cpu().detach().numpy()
                      for g_predict_labels in g_splitted_predict_labels]
                  g_best_sloved_cost_tmp, g_best_solved_id = np.min(g_solved_costs), np.argmin(g_solved_costs)
                  if g_best_sloved_cost_tmp < g_best_solved_cost:
                      g_best_solved_cost = g_best_sloved_cost_tmp
                      g_best_solution_probs = torch.from_numpy(g_splitted_predict_labels[g_best_solved_id])
                      g_best_solution = torch.bernoulli(g_best_solution_probs.clamp(0, 1)).to(device)
                      g_stacked_predict_labels.append((rewrite_step + 1) * g_best_solution_probs.cpu().numpy())
                  else:
                      g_stacked_predict_labels.append(g_best_solution_probs.cpu().numpy())

              return g_stacked_predict_labels, node_labels
          else:
              return splitted_predict_labels, node_labels
  def pred_test_step(self, data, device):
      if isinstance(data, Batch):
          ptr = data.ptr
          point_indicator = ptr[1:] - ptr[:-1]
          batch_index = data.batch
          num_graphs = point_indicator.shape[0]
      else:
          batch_index = None
          num_graphs = 1

      if data.graph_name:
          graph_name = data.graph_name
      else:
          graph_name = None
      graph_data = copy.deepcopy(data)
      edge_index = graph_data.edge_index
      edge_index = edge_index.to(device).reshape(2, -1)
      num_nodes = data.num_nodes
      if graph_data.y is not None:
          node_labels = graph_data.y
          if self.parallel_sampling > 1:
              node_labels_repeat = node_labels.repeat(self.parallel_sampling)
          else:
              node_labels_repeat = node_labels
      else:
          node_labels = None
      stacked_predict_labels = []

      if self.parallel_sampling > 1:
          edge_index = self.duplicate_edge_index(self.parallel_sampling, edge_index, num_nodes, device)
          batch_index = self.duplicate_batch_index(self.parallel_sampling, batch_index,
                                                   num_graphs=num_graphs,
                                                   num_nodes=num_nodes,
                                                   device=device)

      self.infer_direction_params["fitness_map"] = fitness_map(diffusion_type=self.diffusion_type,
                                                               fitness_func_type=self.infer_direction,
                                                               tensor_G=graph_data.to(device),
                                                               temperature=float(self.diffusion_steps),
                                                               )


      T_local = self.diffusion.T
      with torch.no_grad():
          for _ in range(self.sequential_sampling):
              if node_labels is not None:
                  # Sample from diffusion
                  if self.diffusion_type == 'gaussian':
                      xt = self.gaussian_x0_to_xt(g_x0=node_labels_repeat, target_t=T_local)
                  elif self.diffusion_type == 'categorical':
                      xt, _ = self.categorical_x0_to_xt(g_x0=node_labels_repeat, target_t=T_local)
              else:

                  xt = torch.randn(num_nodes).to(device)
                  if self.parallel_sampling > 1:
                      xt = xt.repeat(self.parallel_sampling)
                      xt = torch.randn_like(xt)
              if self.diffusion_type == 'gaussian':
                  xt.requires_grad = True  # ??
              elif self.diffusion_type == 'categorical':
                  xt = (xt > 0).long()
              xt = xt.reshape(-1).to(device)

              batch_size = 1
              steps = self.inference_diffusion_steps
              time_schedule = InferenceSchedule(inference_schedule=self.inference_schedule,
                                                T=self.diffusion.T, inference_T=steps)

              for i in range(steps):
                  t1, t2 = time_schedule(i)
                  t1 = torch.tensor([t1 for _ in range(batch_size)]).int()
                  t2 = torch.tensor([t2 for _ in range(batch_size)]).int()

                  if self.diffusion_type == 'gaussian':
                      xt = self.gaussian_denoise_step(xt, t1, device, edge_index, target_t=t2, batch_index=batch_index)
                  elif self.diffusion_type == 'categorical':
                      xt = self.categorical_denoise_step(xt, t1, device, edge_index, target_t=t2,
                                                         batch_index=batch_index)
              if self.diffusion_type == 'gaussian':
                  predict_labels = xt.float().cpu().detach().numpy() * 0.5 + 0.5
              elif self.diffusion_type == 'categorical':
                  predict_labels = xt.float().cpu().detach().numpy() + 1e-6
              stacked_predict_labels.append(predict_labels)
          predict_labels = np.concatenate(stacked_predict_labels, axis=0)
          all_sampling = self.sequential_sampling * self.parallel_sampling
          if node_labels is None:
              splitted_predict_labels = np.split(predict_labels, all_sampling)
          else:
              splitted_predict_labels = torch.from_numpy(predict_labels)
              node_labels = node_labels.repeat(all_sampling)
              self.rewrite = False
          if self.rewrite:
              solved_costs = [guided_cost(torch.from_numpy(predict_labels), graph_data.edge_index, device=device,

                                          infer_direction=self.infer_direction, **self.infer_direction_params
                                          ).cpu().detach().numpy()
                              for predict_labels in splitted_predict_labels]  #

              g_best_solved_cost, best_solved_id = np.min(solved_costs), np.argmin(solved_costs)
              g_best_solution_probs = torch.from_numpy(splitted_predict_labels[best_solved_id]).to(device)
              g_best_solution = torch.bernoulli(g_best_solution_probs.clamp(0, 1)).to(device)
              g_stacked_predict_labels = []
              for rewrite_step in range(self.rewrite_steps):

                  g_x0 = g_best_solution.clone()
                  if self.parallel_sampling > 1:
                      g_x0 = g_x0.repeat(self.parallel_sampling)

                  steps_T = int(self.diffusion_steps * self.rewrite_ratio)
                  if self.diffusion_type == 'gaussian':
                      g_xt = self.gaussian_x0_to_xt(g_x0, target_t=steps_T)
                  elif self.diffusion_type == 'categorical':
                      g_xt, _ = self.categorical_x0_to_xt(g_x0, target_t=steps_T)
                  g_xt = g_xt.reshape(-1).to(device)
                  steps_inf = int(self.inference_diffusion_steps * self.rewrite_ratio)
                  time_schedule = InferenceSchedule(inference_schedule=self.inference_schedule,
                                                    T=steps_T, inference_T=steps_inf)
                  for i in range(steps_inf):
                      t1, t2 = time_schedule(i)
                      t1 = torch.tensor([t1]).int()
                      t2 = torch.tensor([t2]).int()

                      if self.diffusion_type == 'gaussian':
                          g_xt = self.guided_gaussian_denoise_step(g_xt, t1, edge_index, device,
                                                                   graph_name=graph_name, target_t=t2,
                                                                   batch_index=batch_index)
                      elif self.diffusion_type == 'categorical':
                          g_xt = self.guided_categorical_denoise_step(g_xt, t1, edge_index, device,
                                                                      graph_name=graph_name, target_t=t2,
                                                                      batch_index=batch_index)

                  if self.diffusion_type == 'gaussian':
                      g_predict_labels = g_xt.float().cpu().detach().numpy() * 0.5 + 0.5
                  elif self.diffusion_type == 'categorical':
                      g_predict_labels = g_xt.float().cpu().detach().numpy() + 1e-6
                  g_splitted_predict_labels = np.split(g_predict_labels, self.parallel_sampling)
                  g_stacked_predict_labels.extend(g_splitted_predict_labels)
                  # if rewrite_step < self.rewrite_steps - 1:
                  g_solved_costs = [
                      guided_cost(torch.from_numpy(g_predict_labels), graph_data.edge_index, device=device,

                                  infer_direction=self.infer_direction, **self.infer_direction_params
                                  ).cpu().detach().numpy()
                      for g_predict_labels in g_splitted_predict_labels]
                  g_best_sloved_cost_tmp, g_best_solved_id = np.min(g_solved_costs), np.argmin(g_solved_costs)
                  if g_best_sloved_cost_tmp < g_best_solved_cost:
                      g_best_solved_cost = g_best_sloved_cost_tmp
                      g_best_solution_probs = torch.from_numpy(g_splitted_predict_labels[g_best_solved_id])
                      g_best_solution = torch.bernoulli(g_best_solution_probs.clamp(0, 1)).to(device)
                      g_stacked_predict_labels.append((rewrite_step + 1) * g_best_solution_probs.cpu().numpy())
                  else:
                      g_stacked_predict_labels.append(g_best_solution_probs.cpu().numpy())

              return g_stacked_predict_labels, node_labels
          else:
              return splitted_predict_labels, node_labels

  def target_score_gaussian(self, xt, t,edge_index, device,graph_name=None,batch_index=None):
      # 目标函数可微情况下求xt对目标函数的梯度信息
      if self.gradient_estimator_type == "cost_estimator":
          with torch.inference_mode(False):
              xt.requires_grad = True
              epsilon_pred = self.forward(
                  xt,
                  t.float().to(device),
                  edge_index.long().to(device) if edge_index is not None else None,
                  batch_index
              )
              epsilon_pred = epsilon_pred.reshape(-1)
              atbar = torch.tensor(self.diffusion.alphabar[t]).float().to(device)
              x0_pred = torch.sqrt(1 / atbar) * (xt - torch.sqrt(1 - atbar) * epsilon_pred)
              # score_xt=-epsilon_pred/torch.sqrt(1-atbar)

              pred_nodes = 0.5*x0_pred.reshape(-1)+0.5  # [-1,1] -> [0,1]
              cost_est = -guided_cost(pred_nodes, edge_index, device=device,
                                     infer_direction=self.infer_direction,
                                     **self.infer_direction_params).squeeze()

              cost_est.requires_grad_(True)
              cost_est.backward()

              assert xt.grad is not None
              target_grad_xt = xt.grad.clone()  # [b*n,]
              del xt.grad
              torch.cuda.empty_cache()
      elif self.gradient_estimator_type == "CEP_estimator":
          with torch.inference_mode(False):
            xt.requires_grad=True#[-1,1]

            guided_log_energy_func= - self.CEP_model(xt, t.float.to(device)).to(device)
            guided_log_energy_func.requires_grad_(True)
            guided_log_energy_func.backward()

            assert xt.grad is not None
            target_grad_xt = xt.grad.clone()  # [b*n,]
            del xt.grad
            torch.cuda.empty_cache()
      # 目标函数不可微情况下 近似求xt对目标函数的梯度
      elif self.gradient_estimator_type == "diffusion_evolution_estimator":
          epsilon_pred = self.forward(
              xt,
              t.float().to(device),
              edge_index.long().to(device) if edge_index is not None else None,
              batch_index
          )
          epsilon_pred = epsilon_pred.reshape(-1)
          atbar = torch.tensor(self.diffusion.alphabar[t]).float().to(device)
          x0_pred = torch.sqrt(1 / atbar) * (xt - torch.sqrt(1 - atbar) * epsilon_pred)
          # score_xt = -epsilon_pred / torch.sqrt(1 - atbar)

          xt_detach = xt.detach()
          xt_detach = xt_detach.reshape(self.parallel_sampling, -1)  # [parallel_sampling_num,n]
          fitness_map = self.infer_direction_params["fitness_map"]
          fitness_map.temperature = float(t)
          alpha = atbar
          target_grad_xt = diffusion_evolution_estimator_grad(target_func=fitness_map, xt=xt_detach, alpha=alpha,
                                                              device=device)

      else:
          raise ValueError("gradient_estimator_type {} is not supported!".format(self.gradient_estimator_type))
      if self.norm is True:
          target_grad_xt = nn.functional.normalize(target_grad_xt, p=2, dim=-1)
      update_epsilon = epsilon_pred - self.target_weight * torch.sqrt(1 - atbar) * target_grad_xt
      update_epsilon=nn.functional.normalize(update_epsilon,p=2,dim=-1)
      gc.collect()
      return update_epsilon
  def multi_modal_graph_search(self, graph, n_trials=10, device="cpu",  decode_type=None):
        import random
        """
        Multi-modal graph search implementation with ov-based subgraph sampling.
    
        Args:
            graph: Input graph using torch_geometric.data.Data.
            n_trials: Number of trials for optimization.
            k_neighbors: Number of neighbors in k-NN subgraph sampling.
            omega: Threshold for stopping subgraph decomposition.
    
        Returns:
            The best solution X with the minimum cost.
        """
        num_nodes = graph.num_nodes
        subgraphs, ov = select_subgraphs(graph, num_neighbors=max(int(0.1 * num_nodes), 1000), omega=1)

        heatmaps_generator = self.heatmap_pred_step


        # Step 3-5: Generate heatmap set ℋ for each subgraph
        heatmap_sets = []
        for subgraph in subgraphs:
            # 通过并行与串行采样，得到不同噪声水平的heatmap
            heatmaps_values,_ = heatmaps_generator(subgraph, device)  # or model.pred_test_step_Consistency(subgraph,device)
            heatmaps = []
            for heatmap_value in heatmaps_values:
                subgraph_heatmap = torch.zeros(num_nodes, dtype=torch.float)
                subgraph_heatmap[subgraph.node_idx] = heatmap_value
                heatmaps.append(subgraph_heatmap)
            heatmap_sets.append(heatmaps)

        # Step 6: Initialize trial set 𝒳
        trial_set = []
        for k in range(n_trials):
            combination = [random.choice(h_set) for h_set in heatmap_sets]
            global_heatmap_values = torch.zeros(num_nodes, dtype=torch.float)
            for subgraph_heatmap in combination:
                global_heatmap_values += subgraph_heatmap
            global_heatmap = global_heatmap_values / ov.float()

            trial_x_k = decode_from_heatmap(global_heatmap,graph, decode_type)
            trial_set.append(trial_x_k)

        if decode_type is None:
            return trial_set
        else:
            costs = [torch.sum(x) for x in trial_set]
            best_trial = trial_set[torch.argmin(torch.tensor(costs))]
            return best_trial

  def categorical_x0_to_xt(self, g_x0, target_t):
      g_x0 = g_x0.unsqueeze(0)# [1,N]
      g_x0 = F.one_hot(g_x0.long(), num_classes=2).float()  # [1,N,2]
      # steps_inf = int(self.args.inference_diffusion_steps * self.args.rewrite_ratio)
      Q_bar = torch.from_numpy(self.diffusion.Q_bar[target_t]).float().to(g_x0.device)  # [2,2]
      g_xt_prob = torch.matmul(g_x0, Q_bar)  # [B, N, 2]
      g_xt = torch.bernoulli(g_xt_prob[..., 1].clamp(0, 1)).to(g_x0.device)  # [B, N]
      if self.parallel_sampling > 1:
          g_xt = g_xt.repeat(self.parallel_sampling, 1, 1)
      g_xt = (g_xt > 0).long().reshape(-1)
      return g_xt,g_xt_prob
  def gaussian_x0_to_xt(self, g_x0, target_t):
      target_t = target_t*np.ones(g_x0.shape[0],dtype=int)
      node_labels = g_x0.float() * 2 - 1
      node_labels = node_labels * (1.0 + 0.05 * torch.rand_like(node_labels))
      node_labels = node_labels.unsqueeze(1).unsqueeze(1)
      xt, epsilon = self.diffusion.sample(node_labels, target_t)
      return xt.reshape(-1)




def decode_from_heatmap(heatmap,graph, decode_type=None):
    """
    Decode a trial X_k from the global heatmap H_k.
    This is a placeholder for the actual decoding logic.
    """
    if decode_type is None:
        return heatmap
    elif decode_type == "greedy":
        return fvs_decode_np(heatmap,graph)
    else:
        raise ValueError("decode_type {}  don't support".format(decode_type))




