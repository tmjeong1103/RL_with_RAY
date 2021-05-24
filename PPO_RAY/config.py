# Model
hdims = [32, 32]
#Graph
clip_ratio = 0.2
pi_lr = 3e-4
vf_lr = 1e-3
epsilon = 1e-2
#Buffer
gamma = 0.99
lam = 0.95
#Update
train_pi_iters = 100
train_v_iters = 100
target_kl = 0.01
epochs = 1000
max_ep_len = 1000
#Worker
n_cpu = n_workers = 10
total_steps = 1000
evaluate_every = 50
print_every = 10
ep_len_rollout = 500
batch_size = 4096