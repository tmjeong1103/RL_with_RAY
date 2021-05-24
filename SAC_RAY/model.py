import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from memory import *
from config import *

# Set up function for computing SAC Q-losses
def compute_loss_q(data, model, target):
    o, a, r, o2, d = data['obs1'], data['acts'], data['rews'], data['obs2'], data['done']
    o = torch.Tensor(o)
    a = torch.Tensor(a)
    r = torch.Tensor(r)
    o2 = torch.Tensor(o2)
    d = torch.Tensor(d)

    q1 = model.q1(o,a)
    q2 = model.q2(o,a)

    # Bellman backup for Q functions
    with torch.no_grad():
        # Target actions come from *current* policy
        _ ,a2, logp_a2 = model.policy(o2)

        # Target Q-values
        q1_pi_targ = target.q1(o2, a2)
        q2_pi_targ = target.q2(o2, a2)
        q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
        backup = r + gamma * (1 - d) * (q_pi_targ - alpha_q * logp_a2)

    # MSE loss against Bellman backup
    loss_q1 = ((q1 - backup)**2).mean()
    loss_q2 = ((q2 - backup)**2).mean()
    loss_q = loss_q1 + loss_q2

    # Useful info for logging
    q_info = dict(Q1Vals=q1.detach().numpy(),
                    Q2Vals=q2.detach().numpy())

    return loss_q, q_info

# Set up function for computing SAC pi loss
def compute_loss_pi(data, model):
    o = data['obs1']
    o = torch.Tensor(o)

    _, pi, logp_pi = model.policy(o)
    q1_pi = model.q1(o, pi)
    q2_pi = model.q2(o, pi)
    q_pi = torch.min(q1_pi, q2_pi)

    # Entropy-regularized policy loss
    loss_pi = (alpha_pi * logp_pi - q_pi).mean()

    # Useful info for logging
    pi_info = dict(LogPi=logp_pi.detach().numpy())

    return loss_pi, pi_info

def mlp(odim=24, hdims=[256,256], actv=nn.ReLU(), output_actv=None):
    layers = []
    prev_hdim = odim
    for hdim in hdims[:-1]:
        layers.append(nn.Linear(prev_hdim, hdim, bias=True))
        layers.append(actv)
        prev_hdim = hdim
    layers.append(nn.Linear(prev_hdim, hdims[-1]))

    if output_actv is None:
        return nn.Sequential(*layers)
    else:
        layers.append(output_actv)
        return nn.Sequential(*layers)

class MLPGaussianPolicy(nn.Module):
    def __init__(self, odim=24, adim=8, hdims=[256,256], actv=nn.ReLU()):
        super().__init__()
        self.net = mlp(odim, hdims, actv, output_actv=actv) #feature
        self.mu = nn.Linear(hdims[-1], adim) #mu
        self.log_std = nn.Linear(hdims[-1], adim) #log_std

    def forward(self, o, deterministic=False, squash_action=True):
        net_ouput = self.net(o)
        mu = self.mu(net_ouput)
        log_std = self.log_std(net_ouput)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        #std = torch.exp(log_std)
        std = torch.sigmoid(log_std)

        policy = Normal(mu, std)
        if deterministic:
            pi = mu
        else:
            pi = policy.rsample()   # .sample() : no grad

        # Squash those unbounded actions
        if squash_action:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)

            # gaussian likelihood
            logp_pi = policy.log_prob(pi).sum(dim=1)
            logp_pi -= (2*(np.log(2) - pi -
                           F.softplus(-2*pi))).sum(axis=1)
        else:
            logp_pi = None
        pi = torch.tanh(pi)
        return mu, pi, logp_pi

class MLPQFunction(nn.Module):
    def __init__(self,odim=24, adim=8, hdims=[256,256], actv=nn.ReLU()):
        super().__init__()
        self.q = mlp(odim=odim+adim, hdims=hdims+[1], actv=actv)

    def forward(self, o, a):
        q = self.q(torch.cat([o, a], dim=-1))
        return torch.squeeze(q, -1)      #Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):
    def __init__(self, o, a, hdims=[256,256], actv=nn.ReLU()):
        super().__init__()

        self.policy = MLPGaussianPolicy(o, a, hdims, actv)
        self.q1 = MLPQFunction(o,a,hdims,actv)
        self.q2 = MLPQFunction(o,a,hdims,actv)

    def forward(self, o):
        with torch.no_grad():
            mu, pi, logp_pi = self.policy(o)
            return mu, pi, logp_pi
