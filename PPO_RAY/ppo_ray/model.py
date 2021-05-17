import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from gym.spaces import Box, Discrete

##### Model construction #####
class MLP(nn.Module):       # def mlp in def create_ppo_model
    def __init__(self, o_dim=24, hdims=[64,64], actv=nn.ReLU(),
                 output_actv=None):
        super(MLP, self).__init__()

        self.o_dim = o_dim
        self.hdims = hdims
        self.actv = actv
        self.ouput_actv = output_actv

        self.layers = []
        prev_hdim = self.o_dim
        for hdim in self.hdims[:-1]:
            self.layers.append(nn.Linear(prev_hdim, hdim, bias=True))
            self.layers.append(actv)
            prev_hdim = hdim
        self.layers.append(nn.Linear(prev_hdim, hdims[-1]))

        self.net = nn.Sequential()
        for l_idx, layer in enumerate(self.layers):
            layer_name = "%s_%02d" % (type(layer).__name__.lower(), l_idx)
            self.net.add_module(layer_name, layer)

    def forward(self, inputs):
        x = inputs
        if self.ouput_actv is None:
            x = self.net(x)
        else:
            x = self.net(x)
            x = self.actv(x)
        return x

class CategoricalPolicy(nn.Module):
    def __init__(self, odim, adim, hdims=[64,64], actv=nn.ReLU(), output_actv=None):
        super(CategoricalPolicy, self).__init__()

        self.output_actv = output_actv
        self.net = MLP(odim, hdims=hdims, actv=actv, output_actv=output_actv)
        self.logits = nn.Linear(in_features=hdims[-1], out_features=adim)

    def forward(self, x, a=None):
        output = self.net(x)
        logits = self.logits(output)
        if self.output_actv:
            logits = self.output_actv(logits)
        prob = F.softmax(logits, dim=-1)
        dist = Categorical(probs=prob)
        pi = dist.sample()
        logp_pi = dist.log_prob(pi)
        logp = dist.log_prob(a)
        return pi, logp, logp_pi, pi

class GaussianPolicy(nn.Module):    # def mlp_gaussian_policy
    def __init__(self, odim, adim, hdims=[64,64], actv=nn.ReLU(), output_actv=None):
        super(GaussianPolicy, self).__init__()

        self.output_actv = output_actv
        self.mu = MLP(odim, hdims=hdims+[adim], actv=actv, output_actv=output_actv)
        self.log_std = nn.Parameter(-0.5*torch.ones(adim))

    def forward(self, x, a=None):
        mu = self.mu(x)
        std = self.log_std.exp()
        policy = Normal(mu, std)
        pi = policy.sample()

        # gaussian likelihood
        logp_pi = policy.log_prob(pi).sum(dim=1)
        if a is not None:
            logp = policy.log_prob(a).sum(dim=1)
        else:
            logp = None
        return pi, logp, logp_pi, mu        # 순서 ActorCritic return 값이랑 맞춤.

class ActorCritic(nn.Module):   # def mlp_actor_critic
    def __init__(self, odim, adim, hdims=[64,64], actv=nn.ReLU(),
                 output_actv=None, policy=None, action_space=None):
        super(ActorCritic,self).__init__()

        if policy is None and isinstance(action_space, Box):
            self.policy = GaussianPolicy(odim, adim, hdims, actv, output_actv)
        elif policy is None and isinstance(action_space, Discrete):
            self.policy = CategoricalPolicy(odim, adim, hdims, actv, output_actv)
        self.vf_mlp = MLP(odim, hdims=hdims+[1],
                          actv=actv, output_actv=output_actv)

    def forward(self, x, a=None):
        pi, logp, logp_pi, mu = self.policy(x, a)
        v = self.vf_mlp(x)
        return pi, logp, logp_pi, v, mu