
# # standard deviation clip
# LOG_STD_MIN = -10.0
# LOG_STD_MAX = +2.0
#
# hdims = [256,256]
#
# steps_per_epoch = 4000
# epochs = 100
# replay_size = int(1e6)
#
# n_cpu = 10
# n_workers = 10
#
# lr = 1e-3
# gamma = 0.99
# alpha_q = 0.0
# alpha_pi = 0.1
# polyak = 0.995
#
#
# total_steps = 2000
# evaluate_every = 200
# ep_len_rollout = 100
# batch_size = 128
# update_count = ep_len_rollout
# num_eval= 3
# max_ep_len_eval = 1e3
# buffer_size_long = 1e6
# buffer_size_short = 1e5
#
buffer_size = 1e6


###### SAC with Ray #######

LOG_STD_MIN = -10.0
LOG_STD_MAX = +2.0

alpha_pi = 0.1
alpha_q = 0.0 #0.1

lr = 1e-3
gamma = 0.99
polyak = 0.995
epsilon = 1e-2

hdims = [64,64]

n_cpu = n_workers = 8
total_steps = 2000
evaluate_every = 200
ep_len_rollout = 100
batch_size = 128
update_count = ep_len_rollout
num_eval = 3
max_ep_len_eval = 1e3
buffer_size_long = 1e6
buffer_size_short = 1e5
