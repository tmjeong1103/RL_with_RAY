import torch.nn as nn

# Configuration
n_cpu = n_workers = 50 #100
total_steps = 1000 #5000
evaluate_every = 50
print_every = 10
ep_len_rollout = 1000
num_eval = 3
max_ep_len_eval = 1000
n_env_step = 0
hdims = [256,256]
actv = nn.Tanh()
out_actv = nn.Tanh()

alpha = 0.01
nu = 0.06   #0.03 
b = (n_workers//5)      # 0.01,0.03,(n_workers//5)
