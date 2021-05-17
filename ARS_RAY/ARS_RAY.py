import datetime,gym,os,pybullet_envs,time,os,psutil,ray
import numpy as np
import random
import torch
import torch.nn as nn
from ars import *
from config import *
from matplotlib import pyplot as plt

np.set_printoptions(precision=2)
gym.logger.set_level(40) # gym logger
print("Pytorch version:[%s]."%(torch.__version__))

RENDER_ON_EVAL = False

def get_env():
    import pybullet_envs,gym
    gym.logger.set_level(40) # gym logger
    return gym.make('AntBulletEnv-v0')

def get_eval_env():
    import pybullet_envs,gym
    gym.logger.set_level(40) # gym logger
    eval_env = gym.make('AntBulletEnv-v0')
    if RENDER_ON_EVAL:
        _ = eval_env.render(mode='human') # enable rendering
    _ = eval_env.reset()
    for _ in range(3): # dummy run for proper rendering
        a = eval_env.action_space.sample()
        o,r,d,_ = eval_env.step(a)
        time.sleep(0.01)
    return eval_env

class RolloutWorkerClass(object):
    def __init__(self,hdims=[128], actv=nn.ReLU, out_actv=nn.Tanh,seed=1):
        self.seed = seed
        self.env = get_env()
        odim, adim = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.odim = odim
        self.adim = adim

        #ARS model
        self.model = MLP(o_dim=self.odim, a_dim=self.adim,
                         hdims=hdims, actv=actv, output_actv=out_actv)
        # # model load
        # self.model.load_state_dict(torch.load('model_data/model_weights_1'))
        # print("weight load")

        # Initialize model
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def get_action(self, o):
        return self.model(torch.Tensor(o.reshape(1, -1)))

    def get_weights(self):
        weight_vals = self.model.state_dict()
        return weight_vals

    def set_weights(self, weight_vals):
        return self.model.load_state_dict(weight_vals)

@ray.remote
class RayRolloutWorkerClass(object):
    def __init__(self, worker_id=0, hdims=[128], actv=nn.ReLU,
                 out_actv=nn.Tanh, ep_len_rollout=1000):
        self.worker_id = worker_id
        self.ep_len_rollout = ep_len_rollout
        self.env = get_env()
        odim, adim = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.odim = odim
        self.adim = adim
        #ARS model
        self.model = MLP(o_dim=self.odim, a_dim=self.adim,
                         hdims=hdims, actv=actv, output_actv=out_actv)

        # Flag to initialize rollout
        self.FIRST_ROLLOUT_FLAG = True

    def get_action(self, o):
        return self.model(o)

    def set_weights(self, weight_vals, noise, noise_sign):
        weight_val_noise = {}
        for key, value in weight_vals.items():
            weight_val_noise[key] = weight_vals[key] + noise_sign*noise[key]
        return self.model.load_state_dict(weight_val_noise)

    def rollout(self):
        if self.FIRST_ROLLOUT_FLAG:
            self.FIRST_ROLLOUT_FLAG = False
            self.o = self.env.reset()  # reset environment
        # Loop
        self.o = self.env.reset()  # reset always
        r_sum, step = 0, 0
        self.a = self.get_action(torch.Tensor(self.o.reshape(1, -1)))
        for t in range(self.ep_len_rollout):
            self.a = self.get_action(torch.Tensor(self.o.reshape(1, -1)))
            self.o2, self.r, self.d, _ = self.env.step(self.a.detach().numpy()[0])
            # Save next state
            self.o = self.o2
            # Accumulate reward
            r_sum += self.r
            step += 1
            if self.d:
                break
        return r_sum, step

eval_env = get_eval_env()
adim, odim = eval_env.action_space.shape[0], eval_env.observation_space.shape[0]
print("Environment Ready. odim:[%d] adim:[%d]." % (odim, adim))

ray.init(num_cpus=n_cpu)
R = RolloutWorkerClass(hdims=hdims,actv=actv,out_actv=out_actv,seed=0)
workers = [RayRolloutWorkerClass.remote(worker_id=i,hdims=hdims,actv=actv,
                                        out_actv=out_actv,ep_len_rollout=ep_len_rollout) for i in range(n_workers)]
print("RAY initialized with [%d] cpus and [%d] workers."%(n_cpu,n_workers))

start_time = time.time()
step_list = [] # step list for visualization
ep_max_list = [] # step list for visualization
for t in range(int(total_steps)):
    # Distribute worker weights
    weights = R.get_weights()
    noises_list = []
    for _ in range(n_workers): #worker마다 noise값 다르게 가져간다.
        noises_list.append(get_noises_from_weights(weights, nu=nu))

    # Positive rollouts (noise_sign=+1)
    set_weights_list = [worker.set_weights.remote(weights, noises, noise_sign=+1)
                        for worker, noises in zip(workers, noises_list)]
    ops = [worker.rollout.remote() for worker in workers]
    res_pos = ray.get(ops)
    rollout_pos_vals, r_idx = np.zeros(n_workers), 0
    for rew, eplen in res_pos:
        rollout_pos_vals[r_idx] = rew
        r_idx = r_idx + 1
        n_env_step += eplen

    # Negative rollouts (noise_sign=-1)
    set_weights_list = [worker.set_weights.remote(weights, noises, noise_sign=-1)
                        for worker, noises in zip(workers, noises_list)]
    ops = [worker.rollout.remote() for worker in workers]
    res_neg = ray.get(ops)
    rollout_neg_vals, r_idx = np.zeros(n_workers), 0
    for rew, eplen in res_neg:
        rollout_neg_vals[r_idx] = rew
        r_idx = r_idx + 1
        n_env_step += eplen
    # Scale reward
    rollout_pos_vals, rollout_neg_vals = rollout_pos_vals / 100, rollout_neg_vals / 100

    # Reward
    rollout_concat_vals = np.concatenate((rollout_pos_vals, rollout_neg_vals))
    rollout_delta_vals = rollout_pos_vals - rollout_neg_vals  # pos-neg
    rollout_max_vals = np.maximum(rollout_pos_vals, rollout_neg_vals)
    rollout_max_val = np.max(rollout_max_vals)  # single maximum
    rollout_delta_max_val = np.max(np.abs(rollout_delta_vals))

    # Sort
    sort_idx = np.argsort(-rollout_max_vals)

    # Update
    sigma_R = np.std(rollout_concat_vals)
    weights_updated = {}
    for key, weight in weights.items():  # for each weight
        delta_weight_sum = np.zeros_like(weight)
        for k in range(b):
            idx_k = sort_idx[k]  # sorted index
            rollout_delta_k = rollout_delta_vals[idx_k]
            noises_k = noises_list[idx_k]
            noise_k = (1 / nu) * noises_k[key]  # noise for current weight
            delta_weight_sum += rollout_delta_k * noise_k.detach().numpy()
        delta_weight = (alpha / (b * sigma_R)) * delta_weight_sum
        weight = weight + delta_weight
        weights_updated[key] = weight
    # Set weight
    R.set_weights(weights_updated)

    # Print
    if (t == 0) or (((t + 1) % print_every) == 0):
        print("[%d/%d] rollout_max_val:[%.2f] rollout_delta_max_val:[%.2f] sigma_R:[%.2f] " %
              (t, total_steps, rollout_max_val, rollout_delta_max_val, sigma_R))

    # save model
    if t % 10 == 0:
        torch.save(R.get_weights(), 'model_data/model_weights_1')
        print("Weight saved")

    # Evaluate
    if (t == 0) or (((t + 1) % evaluate_every) == 0) or (t == (total_steps - 1)):
        ram_percent = psutil.virtual_memory().percent  # memory usage
        print("[Evaluate] step:[%d/%d][%.1f%%] #step:[%.1e] time:[%s] ram:[%.1f%%]." %
              (t + 1, total_steps, t / total_steps * 100,
               n_env_step,
               time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
               ram_percent)
              )
        step_list.append(t) #for visualization
        ep_ret_list = []  #for visualization
        for eval_idx in range(num_eval):
            o, d, ep_ret, ep_len = eval_env.reset(), False, 0, 0
            if RENDER_ON_EVAL:
                _ = eval_env.render(mode='human')
            while not (d or (ep_len == max_ep_len_eval)):
                a = R.get_action(o)
                o, r, d, _ = eval_env.step(a.detach().numpy()[0])
                if RENDER_ON_EVAL:
                    _ = eval_env.render(mode='human')
                ep_ret += r  # compute return
                ep_len += 1
                ep_ret_list.append(ep_ret)  #for visualization
            print(" [Evaluate] [%d/%d] ep_ret:[%.4f] ep_len:[%d]"
                  % (eval_idx, num_eval, ep_ret, ep_len))
        ep_max_list.append(max(ep_ret_list)) #for visualization

# make ep_ret-total step graph
plt.plot(step_list,ep_max_list,marker='o')
plt.xlabel('step')
plt.ylabel('ep_return')
plt.title("alpha:[%.4f] nu:[%.4f]"%(alpha,nu))
plt.grid(True, linestyle='--')
plt.show()
plt.savefig('ARS_result.png',dpi=100)

print("Done.")
eval_env.close()
ray.shutdown()
