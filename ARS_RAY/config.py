import torch.nn as nn

# Configuration
n_cpu = n_workers = 10
total_steps = 5000
evaluate_every = 50
print_every = 10
ep_len_rollout = 1000
num_eval = 3
max_ep_len_eval = 1e3
n_env_step = 0
hdims = [32,16]
actv = nn.Tanh()
out_actv = nn.Tanh()

alpha = 0.01
nu = 0.05 #0.03
b = (n_workers//5)      # 0.01,0.05,(n_workers//5)