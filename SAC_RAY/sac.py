import datetime,gym,os,pybullet_envs,time,os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from config import *

np.set_printoptions(precision=2)
gym.logger.set_level(40)    # gym logger
print("Pytorch version:[%s]."%(torch.__version__))

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, odim, adim, size):
        self.obs1_buf = np.zeros([size, odim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, odim], dtype=np.float32)
        self.acts_buf = np.zeros([size, adim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs1=self.obs1_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     acts=self.acts_buf[idxs],
                     rews=self.rews_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def get(self):
        names = ['obs1_buf','obs2_buf','acts_buf','rews_buf','done_buf',
                 'ptr','size','max_size']
        vals =[self.obs1_buf,self.obs2_buf,self.acts_buf,self.rews_buf,self.done_buf,
               self.ptr,self.size,self.max_size]
        return names,vals

    def restore(self,a):
        self.obs1_buf = a[0]
        self.obs2_buf = a[1]
        self.acts_buf = a[2]
        self.rews_buf = a[3]
        self.done_buf = a[4]
        self.ptr = a[5]
        self.size = a[6]
        self.max_size = a[7]

"""
Soft Actor Critic Model (compatible with Ray)
"""
# Dense Layer / Multi Layer Perceptron
def mlp(odim=24, hdims=hdims, actv=nn.ReLU(), output_actv=nn.ReLU()):
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


# Gaussian Policy MLP
class MLPGaussianPolicy(nn.Module):
    def __init__(self, odim=24, adim=8, hdims=hdims, actv=nn.ReLU()):
        super().__init__()
        # MLP
        self.net = mlp(odim, hdims, actv, output_actv=actv) #feature
        # mu layer
        self.mu = nn.Linear(hdims[-1], adim, bias=True)
        # std layer
        self.log_std = nn.Linear(hdims[-1], adim, bias=True)

    def forward(self, o, deterministic=False, get_logprob=True):
        net_ouput = self.net(o)
        mu = self.mu(net_ouput)
        log_std = self.log_std(net_ouput)

        LOG_STD_MIN, LOG_STD_MAX = -10.0, +2.0
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) #log_std
        #std = torch.exp(log_std)
        std = torch.sigmoid(log_std) #std

        # Pre-squash distribution and sample
        dist = Normal(mu, std)
        if deterministic:
            pi = mu
        else:
            pi = dist.rsample()    # sampled

        if get_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = dist.log_prob(pi).sum(axis=-1)    #gaussian log_likelihood # modified axis
            logp_pi -= (2 * (np.log(2) - pi - F.softplus(-2 * pi))).sum(axis=1)
        else:
            logp_pi = None
        pi = torch.tanh(pi)
        return pi, logp_pi

# Q-function mlp
class MLPQFunction(nn.Module):
    def __init__(self,odim=24, adim=8, hdims=hdims, actv=nn.ReLU()):
        super().__init__()
        self.q = mlp(odim=odim+adim, hdims=hdims+[1], actv=actv, output_actv=None)

    def forward(self, o, a):
        x = torch.cat([o, a], dim=-1)
        q = self.q(x)
        return torch.squeeze(q, -1)   #Critical to ensure q has right shape.

# ActorCritic module
class MLPActorCritic(nn.Module):
    def __init__(self, o, a, hdims=hdims, actv=nn.ReLU()):
        super().__init__()

        self.policy = MLPGaussianPolicy(odim=o, adim=a, hdims=hdims, actv=actv)
        self.q1 = MLPQFunction(odim=o, adim=a, hdims=hdims, actv=actv)
        self.q2 = MLPQFunction(odim=o, adim=a, hdims=hdims, actv=actv)

    def get_action(self, o, deterministic=False):
        with torch.no_grad():
            pi, _ = self.policy(o, deterministic, False)
            return pi

    def calc_pi_loss(self, data):
        o = data['obs1']
        pi, logp_pi = self.policy(o,)
        q1_pi = self.q1(o,pi)
        q2_pi = self.q2(o,pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        # pi losses
        pi_loss = (alpha_pi*logp_pi - min_q_pi).mean()
        return pi_loss

    def calc_q_loss(self, target, data):
        o, a, r, o2, d = data['obs1'], data['acts'], data['rews'], data['obs2'], data['done']

        # Entropy-regularized Bellman backup
        with torch.no_grad():
            # get target action from current policy
            pi_next, logp_pi_next = self.policy(o2)
            # Target value
            q1_targ = target.q1(o2, pi_next)
            q2_targ = target.q2(o2, pi_next)
            min_q_targ = torch.min(q1_targ, q2_targ)
            # Entropy-regularized Bellman backup
            q_backup = r + gamma*(1 - d)*(min_q_targ - alpha_q*logp_pi_next)
        q1 = self.q1(o, a)
        q2 = self.q2(o, a)
        # value(q) loss
        q1_loss = 0.5*F.mse_loss(q1,q_backup)           #0.5 * ((q_backup-q1)**2).mean()
        q2_loss = 0.5*F.mse_loss(q2,q_backup)          #0.5 * ((q_backup-q2)**2).mean()
        value_loss = q1_loss + q2_loss
        return value_loss