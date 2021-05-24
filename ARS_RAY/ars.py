import numpy as np
import torch
import torch.nn as nn

##### Model construction #####
class MLP(nn.Module):       # def mlp in def create_ppo_model
    def __init__(self, o_dim=24, a_dim=8, hdims=[64,64], actv=nn.ReLU(),
                 output_actv=nn.ReLU()):
        super(MLP, self).__init__()
        self.o_dim = o_dim
        self.hdims = hdims
        self.actv = actv
        self.ouput_actv = output_actv
        self.layers = []
        prev_hdim = self.o_dim
        for hdim in self.hdims:
            linear = nn.Linear(prev_hdim, hdim)
            nn.init.trunc_normal_(linear.weight, std=0.1)
            self.layers.append(linear)
            self.layers.append(actv)
            prev_hdim = hdim
        linear_out = nn.Linear(prev_hdim, a_dim)
        nn.init.trunc_normal_(linear_out.weight, std=0.1)  # add!
        self.layers.append(linear_out)
        self.net = nn.Sequential()
        for l_idx, layer in enumerate(self.layers):
            layer_name = "%s_%02d" % (type(layer).__name__.lower(), l_idx)
            self.net.add_module(layer_name, layer)

    def forward(self, obs):
        x = obs
        if self.ouput_actv is None:
            mu = self.net(x)
        else:
            mu = self.net(x)
            mu = self.actv(mu)
        return mu

# weight와 동일하게 dictionary형태로 noise 만들고 main에서 더하는 걸로.
def get_noises_from_weights(weights, nu=0.01):
    noises = {}
    for key, value in weights.items():
        noises[key] = nu * torch.rand(value.shape)
    return noises # dictionary