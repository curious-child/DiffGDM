"""Schedulers for Denoising Diffusion Probabilistic Models"""
import torch.nn as nn
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import curve_fit, least_squares
from torch.distributions import Uniform, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform, AffineTransform, ComposeTransform
from .diffusion_model_utils import cumulative_density_function,median,get_codomain,TanTransform
def log_norm(x, mu=0.0, sigma=1.0):
  """
  对数正态分布概率密度函数

  参数：
  x    : 输入值或数组（必须>0）
  mu   : 对数均值（默认0）
  sigma: 对数标准差（必须>0，默认1）

  返回：
  概率密度值，形状与x相同
  """
  # 数值稳定性处理
  x = np.asarray(x, dtype=np.float64)
  sigma = np.maximum(sigma, 1e-8)  # 防止sigma=0
  mask = x > 0  # 仅处理正数

  # 初始化结果数组
  pdf = np.zeros_like(x)

  # 核心计算公式
  log_x = np.log(x[mask])
  exponent = -0.5 * ((log_x - mu) / sigma)** 2
  denominator = sigma * np.sqrt(2 * np.pi) * x[mask]

  pdf[mask] = np.exp(exponent) / denominator
  return pdf

class GaussianDiffusion(object):
  """Gaussian Diffusion process with linear beta scheduling"""

  def __init__(self, T, schedule,loss_weight_schedule="constant"):
    # Diffusion steps
    self.T = T
    self.loss_weight_schedule = loss_weight_schedule#"constant" or "logNormal"

    # Noise schedule
    if schedule == 'linear':
      b0 = 1e-4
      bT = 2e-2
      self.beta = np.linspace(b0, bT, T)
    elif schedule == 'cosine':
      self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
          0)  # Generate an extra alpha for bT
      self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

    self.betabar = np.cumprod(self.beta)
    self.alpha = np.concatenate((np.array([1.0]), 1 - self.beta))
    self.alphabar = np.cumprod(self.alpha)

  def __cos_noise(self, t):
    offset = 0.008
    return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

  def sample(self, x0, t):
    # Select noise scales

    noise_dims = (x0.shape[0],) + tuple((1 for _ in x0.shape[1:]))
    atbar = torch.from_numpy(self.alphabar[t]).view(noise_dims).to(x0.device)
    assert len(atbar.shape) == len(x0.shape), 'Shape mismatch'

    # Sample noise and add to x0
    epsilon = torch.randn_like(x0)
    xt = torch.sqrt(atbar) * x0 + torch.sqrt(1.0 - atbar) * epsilon
    return xt, epsilon

  def get_loss_weights(self,time_step):
    if self.loss_weight_schedule == "constant":
      time_step = torch.from_numpy(time_step)
      return torch.ones_like(time_step)
    elif self.loss_weight_schedule == "logNormal":
      mu = 0.0
      sigma = 0.5
      atbar=self.alphabar[time_step]
      snr=atbar/(1-atbar)
     # print("snr",snr)
      loss_weight = log_norm(snr, mu, sigma)
     # print("loss_weight", loss_weight)
      return torch.from_numpy(loss_weight)

class CategoricalDiffusion(object):
  """Gaussian Diffusion process with linear beta scheduling"""

  def __init__(self, T, schedule,loss_weight_schedule):
    # Diffusion steps
    self.T = T
    self.loss_weight_schedule = loss_weight_schedule  # "constant" or "logNormal"
    # Noise schedule
    if schedule == 'linear':
      b0 = 1e-4
      bT = 2e-2
      self.beta = np.linspace(b0, bT, T)
    elif schedule == 'cosine':
      self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
          0)  # Generate an extra alpha for bT
      self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

    beta = self.beta.reshape((-1, 1, 1))
    eye = np.eye(2).reshape((1, 2, 2))
    ones = np.ones((2, 2)).reshape((1, 2, 2))

    self.Qs = (1 - beta) * eye + (beta / 2) * ones
    #[2,2]
    Q_bar = [np.eye(2)]
    for Q in self.Qs:
      Q_bar.append(Q_bar[-1] @ Q)
    #[T,2,2]
    self.Q_bar = np.stack(Q_bar, axis=0)

  def __cos_noise(self, t):
    offset = 0.008
    return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

  def sample(self, x0_onehot, t):
    # Select noise scales
    Q_bar = torch.from_numpy(self.Q_bar[t]).float().to(x0_onehot.device)
 #   print("Q_bar.shape[0]",Q_bar.shape[0])
    # [节点数,1,1,2]
    xt = torch.matmul(x0_onehot, Q_bar.reshape((Q_bar.shape[0], 1, 2, 2)))
  #  print("sampel xt shape:{}".format(xt.shape))
    return torch.bernoulli(xt[..., 1].clamp(0, 1))
  def get_loss_weights(self,time_step):
  #  print(self.loss_weight_schedule)
    if self.loss_weight_schedule == "constant":
      time_step=torch.from_numpy(time_step)
      return torch.ones_like(time_step)
    elif self.loss_weight_schedule == "logNormal":
      mu = 0.0
      sigma = 0.5

      atbar=self.Q_bar[time_step][:,0,0]
      snr=atbar/(1-atbar)
      loss_weight = log_norm(snr, mu, sigma)
     # print("loss_weight",loss_weight)
      return torch.from_numpy(loss_weight)

class DiscreteFlowDiffusion(object):
  def __init__(self, T,  loss_weight_schedule,discrete_classes):
    self.T = T
    self.loss_weight_schedule = loss_weight_schedule  # "constant" or "logNormal"
    self.S = discrete_classes
  def sample(self,x1, t):
    # x1 (B*nodes,): input orignal data
    # t (B*nodes,) np.array
    # Returns xt (B*nodes, )

    # uniform
    t=1-t/self.T

    t=torch.from_numpy(t).float()
    xt = x1.clone()
    uniform_noise = torch.randint(0, self.S, size=xt.shape, device=xt.device)
    corrupt_mask = torch.rand((xt.shape[0])).to(xt.device) < (1 - t).to(xt.device)
    xt[corrupt_mask] = uniform_noise[corrupt_mask]

    # masking
    # xt = x1.clone()
    # xt[torch.rand((B,D)) < (1 - t[:, None])] = S-1

    return xt
  def discreteflow_time(self,t):
    return 1-t/self.T
  def get_loss_weights(self,time_step):
    if self.loss_weight_schedule == "constant":
      time_step = torch.from_numpy(time_step)
      return torch.ones_like(time_step)
    elif self.loss_weight_schedule == "logNormal":
      mu = 0.0
      sigma = 0.5

      atbar=time_step/(self.T)+1e-9
      snr=atbar**2/(1-atbar)**2

      loss_weight = log_norm(snr, mu, sigma)
     # print("loss_weight", loss_weight)
      return torch.from_numpy(loss_weight)
  def get_rate_matrix(self,x1,xt,t):
    # Calculate R_t^*
    # For p(x_t | x_1) > 0 and p(j | x_1) > 0
    # R_t^*(x_t, j | x_1) = Relu( dtp(j | x_1) - dtp(x_t | x_1)) / (Z_t * p(x_t | x_1))
    # For p(x_t | x_1) = 0 or p(j | x_1) = 0 we have R_t^* = 0

    # We will ignore issues with diagnoal entries as later on we will set
    # diagnoal probabilities such that the row sums to one later on.
    S=2# number of classes
    dt_p_vals = dt_p_xt_g_xt(x1, t)  # (B*D, S)
    dt_p_vals_at_xt = dt_p_vals.gather(-1, xt[:, None]).squeeze(-1)  # (B*D)

    # Numerator of R_t^*
    R_t_numer = F.relu(dt_p_vals - dt_p_vals_at_xt[:, None])  # (B*D, S)

    pt_vals = p_xt_g_x1(x1, t)  # (B*D, S)
    Z_t = torch.count_nonzero(pt_vals, dim=-1)  # (B*D)
    pt_vals_at_xt = pt_vals.gather(-1, xt[:, None]).squeeze(-1)  # (B*D)

    # Denominator of R_t^*
    R_t_denom = Z_t * pt_vals_at_xt  # (B*D)

    R_t = R_t_numer / R_t_denom[:, None]  # (B*D, S)

    # Set p(x_t | x_1) = 0 or p(j | x_1) = 0 cases to zero
    R_t[(pt_vals_at_xt == 0.0)[:, None].repeat(1, S)] = 0.0
    R_t[pt_vals == 0.0] = 0.0
    return R_t #[B*D,S]
class InferenceSchedule(object):
  def __init__(self, inference_schedule="linear", T=1000, inference_T=1000):
    self.inference_schedule = inference_schedule
    self.T = T
    self.inference_T = inference_T

  def __call__(self, i):
    assert 0 <= i < self.inference_T

    if self.inference_schedule == "linear":
      t1 = self.T - int((float(i) / self.inference_T) * self.T)
      t1 = np.clip(t1, 1, self.T)

      t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
      t2 = np.clip(t2, 0, self.T - 1)
      return t1, t2
    elif self.inference_schedule == "cosine":
      t1 = self.T - int(
          np.sin((float(i) / self.inference_T) * np.pi / 2) * self.T)
      t1 = np.clip(t1, 1, self.T)

      t2 = self.T - int(
          np.sin((float(i + 1) / self.inference_T) * np.pi / 2) * self.T)
      t2 = np.clip(t2, 0, self.T - 1)
      return t1, t2
    else:
      raise ValueError("Unknown inference schedule: {}".format(self.inference_schedule))

class AdaptiveSchedule(nn.Module):
  def __init__(self, tmin, tmax, mu, sigma, height, offset):
    super().__init__()

    self.tmin, self.tmax = tmin, tmax
    self.parameters_history, self.times, self.entropy, self.medians = [], [], [], []
    self.set_parameters([mu, sigma, height, offset])

  def set_parameters(self, params):
    self.mu, self.sigma, self.height, self.offset = params
    self.medians.append(median(*params))
    self.optimal_parameters = nn.Parameter(torch.tensor(params, device='cpu'), requires_grad=False)
    self.parameters_history.append(params)

    self.transforms = ComposeTransform([
      AffineTransform(loc=-self.offset / self.height, scale=1 / self.height),
      AffineTransform(loc=-np.pi / 2, scale=np.pi),
      TanTransform(),
      AffineTransform(loc=self.mu, scale=self.sigma)
    ])

    imin, imax = get_codomain(cumulative_density_function, self.tmin, self.tmax, self.mu, self.sigma, self.height,
                              self.offset)


    self.distribution = TransformedDistribution(Uniform(imin, imax), self.transforms)

  def add_data(self, entropy, times):
    history_cutoff = int(1e5)
    self.times += times.tolist()
    self.entropy += entropy.tolist()

    if len(self.times) >= history_cutoff:
      self.times = self.times[-history_cutoff:]
      self.entropy = self.entropy[-history_cutoff:]

  def make_timesteps(self, steps, tmin=None, tmax=None, device='cpu', batch_size=None, seq_len=None,
                     padding_mask=None) :
    tmin = self.tmin if tmin is None else tmin
    tmax = self.tmax if tmax is None else tmax

    imin, imax = get_codomain(cumulative_density_function, tmin, tmax, self.mu, self.sigma, self.height, self.offset)
    indexes = torch.linspace(imax, imin, steps, device=device)

    timesteps = self.transforms(indexes)
    timesteps = torch.clamp(timesteps, self.tmin, self.tmax)

    if padding_mask is not None:
      # Ensure padding positions get max timesteps (least noise)
      expanded_timesteps = timesteps.expand(batch_size, seq_len)
      expanded_timesteps[padding_mask] = self.tmax
      return expanded_timesteps

    return timesteps

  def update_optimal_parameters(self, history=500):
    times, entropy = self.times[-history:], self.entropy[-history:]
    # Note: times and entropy already exclude padding tokens due to add_data filtering

    offset_lower_bound = -cumulative_density_function(0, self.mu, self.sigma, self.height, 0)
    bounds = (
      (0.1, 0.1, 0.1, offset_lower_bound),
      (1000, 1000, 1000, np.inf)
    )

    def residuals(params, x, y):
      """计算模型预测值与实际值的残差"""
      mu, sigma, height, offset = params
      predicted = cumulative_density_function(x, mu, sigma, height, offset)
      return predicted - y

    initial_params = np.array([self.mu, self.sigma, self.height, offset_lower_bound + 1e-3])
    try:
      # 调用 least_squares 替代 curve_fit
      result = least_squares(
        fun=residuals,
        x0=initial_params,
        args=(times, entropy),
        bounds=bounds,
        max_nfev=2000  # 可调整的最大函数评估次数
      )

      # 提取优化参数（即使未收敛）
      optimal_parameters = result.x



    except Exception as e:
      print(f"优化失败: {str(e)}")
      # 回退到初始参数或最后可用参数
      optimal_parameters = initial_params if 'result' not in locals() else result.x


    self.set_parameters(optimal_parameters.tolist())


  def sample(self, shape):

    return self.distribution.sample(shape)

def dt_p_xt_g_xt(x1, t):
    # x1 (B, D)
    # t float
    # returns (B, D, S) for varying x_t value
    S=2
    # uniform
    x1_onehot = F.one_hot(x1, num_classes=S)  # (B*D, S)
    return x1_onehot - (1 / S)

    # masking
    # x1_onehot = F.one_hot(x1, num_classes=S) # (B, D, S)
    # M_onehot = F.one_hot(torch.tensor([S-1]), num_classes=S)[None, :, :] # (1, 1, S)
    # return x1_onehot - M_onehot


def p_xt_g_x1(x1, t):
  # x1 (B, D)
  # t float
  # returns (B, D, S) for varying x_t value
  S=2
  # uniform
  x1_onehot = F.one_hot(x1, num_classes=S)  # (B*D, S)
  return t * x1_onehot + (1 - t) * (1 / S)

  # masking
  # x1_onehot = F.one_hot(x1, num_classes=S) # (B, D, S)
  # M_onehot = F.one_hot(torch.tensor([S-1]), num_classes=S)[None, :, :] # (1, 1, S)
  # return t * x1_onehot + (1-t) * M_onehot
