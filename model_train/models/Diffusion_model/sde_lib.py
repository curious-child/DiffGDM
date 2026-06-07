
import torch




#ddpm sde
class VPSDE(object):
  def __init__(self, diffusion,N):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__()
    self.N = N
    self.diffusion = diffusion
    self.beta_0 = self.diffusion.beta[0]*N
    self.beta_1 = self.diffusion.beta[-1]*N
  @property
  def T(self):
    return 1

  def sde(self, x, t):
    dt=t/self.N*self.T

    beta_t = self.beta_0 + dt * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t* x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = t
    alpha = torch.tensor(self.diffusion.alpha[timestep]).to(x.device)
    sqrt_beta = torch.sqrt(1-alpha)
    f = torch.sqrt(alpha)* x - x
    G = sqrt_beta
    return f, G


