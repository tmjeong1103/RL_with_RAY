import datetime,gym,os,pybullet_envs,time,os,psutil,ray
import random
import torch
from memory import *
from model_modified import *
from config_ray import *

np.set_printoptions(precision=2)
gym.logger.set_level(40) # gym logger
print("Pytorch version:[%s]."%(torch.__version__))

# Rollout Worker
def get_env():
    return gym.make('AntBulletEnv-v0')

def get_eval_env():
    eval_env = gym.make('AntBulletEnv-v0')
    #_ = eval_env.render(mode='human') # enable rendering
    _ = eval_env.reset()
    for _ in range(3): # dummy run for proper rendering
        a = eval_env.action_space.sample()
        o,r,d,_ = eval_env.step(a)
        time.sleep(0.01)
    return eval_env

class RolloutWorkerClass(object):
    def __init__(self, seed=1):
        self.seed = seed
        self.env = get_env()
        odim, adim = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.odim = odim
        self.adim = adim

        # Initialize PPO
        # Actor-critic model
        ac_kwargs = dict()
        ac_kwargs['action_space'] = self.env.action_space
        self.model = ActorCritic(odim, adim, hdims, **ac_kwargs)

        # Initialize model
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Optimizers
        self.train_pi = torch.optim.Adam(self.model.policy.parameters(), lr=pi_lr)
        self.train_v = torch.optim.Adam(self.model.vf_mlp.parameters(), lr=vf_lr)

    def get_weights(self):
        weight_vals = self.model.state_dict()
        return weight_vals

    def set_weights(self, weight_vals):
        return self.model.load_state_dict(weight_vals)

@ray.remote
class RayRolloutWorkerClass(object):
    """
    Rollout Worker with RAY
    """
    def __init__(self, worker_id=0, ep_len_rollout=1000):
        # Parse
        self.worker_id = worker_id
        self.ep_len_rollout = ep_len_rollout


        # Each worker should maintain its own environment
        import pybullet_envs, gym
        gym.logger.set_level(40)  # gym logger
        self.env = get_env()
        odim, adim = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.odim = odim
        self.adim = adim

        # Replay buffers to pass
        self.o_buffer = np.zeros((self.ep_len_rollout, self.odim))
        self.a_buffer = np.zeros((self.ep_len_rollout, self.adim))
        self.r_buffer = np.zeros((self.ep_len_rollout))
        self.v_t_buffer = np.zeros((self.ep_len_rollout))
        self.logp_t_buffer = np.zeros((self.ep_len_rollout))

        # Create PPO model
        # Actor-critic model
        ac_kwargs = dict()
        ac_kwargs['action_space'] = self.env.action_space
        self.model = ActorCritic(odim, adim, hdims, **ac_kwargs)
        # Buffer
        self.buf = PPOBuffer(odim=self.odim, adim=self.adim,
                             size=self.ep_len_rollout, gamma=gamma, lam=lam)
        # Flag to initialize rollout
        self.FIRST_ROLLOUT_FLAG = True

    def set_weights(self, weight_vals):
        return self.model.load_state_dict(weight_vals)

    def rollout(self):
        if self.FIRST_ROLLOUT_FLAG:
            self.FIRST_ROLLOUT_FLAG = False
            self.o = self.env.reset()  # reset environment
        # Loop
        for t in range(self.ep_len_rollout):
            a, _, logp_t, v_t, _ = self.model(torch.Tensor(self.o.reshape(1, -1)))  # pi, logp, logp_pi, v, mu
            o2, r, d, _ = self.env.step(a.detach().numpy()[0])
            # save and log
            self.buf.store(self.o, a, r, v_t, logp_t)
            # Update obs (critical!)
            self.o = o2
            if d:
                self.buf.finish_path(last_val=0.0)
                self.o = self.env.reset()  # reset when done
        last_val = self.model.vf_mlp(torch.Tensor(self.o.reshape(1, -1))).item()
        self.buf.finish_path(last_val)
        return self.buf.get()

# Initialize PyBullet Ant Environment
eval_env = get_eval_env()
adim,odim = eval_env.action_space.shape[0],eval_env.observation_space.shape[0]
print("Environment Ready. odim:[%d] adim:[%d]."%(odim,adim))

# Initialize Workers
ray.init(num_cpus=n_cpu,
         _memory = 5*1024*1024*1024,
         object_store_memory = 10*1024*1024*1024,
         _driver_object_store_memory = 1*1024*1024*1024)

R = RolloutWorkerClass(seed=0)
workers = [RayRolloutWorkerClass.remote(worker_id=i,ep_len_rollout=ep_len_rollout)
           for i in range(int(n_workers))]
print("RAY initialized with [%d] cpus and [%d] workers."%
       (n_cpu, n_workers))
time.sleep(1)
# Loop
start_time = time.time()
n_env_step = 0  # number of environment steps

for t in range(int(total_steps)):
    esec = time.time() - start_time
    # 1. Synchronize worker weights
    weights = R.get_weights()
    set_weights_list = [worker.set_weights.remote(weights) for worker in workers]
    # 2. Make rollout and accumulate to Buffers
    t_start = time.time()
    ops = [worker.rollout.remote() for worker in workers]
    rollout_vals = ray.get(ops)
    sec_rollout = time.time() - t_start
    # 3. Update
    t_start = time.time()  # tic
    # Mini-batch type of update
    for r_idx, rval in enumerate(rollout_vals):
        obs_buf, act_buf, adv_buf, ret_buf, logp_buf = \
            rval[0], rval[1], rval[2], rval[3], rval[4]
        if r_idx == 0:
            obs_bufs, act_bufs, adv_bufs, ret_bufs, logp_bufs = \
                obs_buf, act_buf, adv_buf, ret_buf, logp_buf
        else:
            obs_bufs = np.concatenate((obs_bufs, obs_buf), axis=0)
            act_bufs = np.concatenate((act_bufs, act_buf), axis=0)
            adv_bufs = np.concatenate((adv_bufs, adv_buf), axis=0)
            ret_bufs = np.concatenate((ret_bufs, ret_buf), axis=0)
            logp_bufs = np.concatenate((logp_bufs, logp_buf), axis=0)
    n_val_total = obs_bufs.shape[0]
    for pi_iter in range(int(train_pi_iters)):
        rand_idx = np.random.permutation(n_val_total)[:batch_size]
        buf_batches = [obs_bufs[rand_idx], act_bufs[rand_idx], adv_bufs[rand_idx],
                       ret_bufs[rand_idx], logp_bufs[rand_idx]]
        obs, act, adv, ret, logp_a_old = [torch.Tensor(x) for x in buf_batches]
        ent = (-logp_a_old).mean()
        _, logp_a, _, _ = R.model.policy(obs, act)
        # PPO objectives
        ratio = (logp_a - logp_a_old).exp()
        min_adv = torch.where(adv > 0, (1 + clip_ratio) * adv, (1 - clip_ratio) * adv)
        pi_loss = -(torch.min(ratio * adv, min_adv)).mean()
        R.train_pi.zero_grad(set_to_none=True)
        pi_loss.backward()
        R.train_pi.step()
        # a sample estimate for KL-divergence
        kl = torch.mean(logp_a_old - logp_a)
        if kl > 1.5 * target_kl:
            #print("  pi_iter:[%d] kl(%.3f) is higher than 1.5x(%.3f)" % (pi_iter, kl, target_kl))
            break
    # Value gradient step
    for _ in range(int(train_v_iters)):
        rand_idx = np.random.permutation(n_val_total)[:batch_size]
        buf_batches = [obs_bufs[rand_idx], act_bufs[rand_idx], adv_bufs[rand_idx],
                       ret_bufs[rand_idx], logp_bufs[rand_idx]]
        obs, act, adv, ret, logp = [torch.Tensor(x) for x in buf_batches]
        v = R.model.vf_mlp(obs).squeeze()
        v_loss = F.mse_loss(v, ret)
        R.train_v.zero_grad(set_to_none=True)
        v_loss.backward()
        R.train_v.step()
    sec_update = time.time() - t_start  # toc
    # Print
    if (t == 0) or (((t + 1) % print_every) == 0):
        print("[%d/%d] rollout:[%.1f]s pi_iter:[%d/%d] update:[%.1f]s kl:[%.4f] target_kl:[%.4f]." %
              (t + 1, total_steps, sec_rollout, pi_iter, train_pi_iters, sec_update, kl, target_kl))
        print("   pi_loss:[%.4f], entropy:[%.4f]" %
              (pi_loss, ent))
    # Evaluate
    if (t == 0) or (((t + 1) % evaluate_every) == 0):
        ram_percent = psutil.virtual_memory().percent  # memory usage
        print("[Eval. start] step:[%d/%d][%.1f%%] #step:[%.1e] time:[%s] ram:[%.1f%%]." %
              (t + 1, total_steps, t / total_steps * 100,
               n_env_step,
               time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
               ram_percent)
              )
        o, d, ep_ret, ep_len = eval_env.reset(), False, 0, 0
        #_ = eval_env.render(mode='human')

        while not (d or (ep_len == max_ep_len)):
            a, _, _, _ = R.model.policy(torch.Tensor(o.reshape(1, -1)))
            o, r, d, _ = eval_env.step(a.detach().numpy()[0])
            #_ = eval_env.render(mode='human')
            ep_ret += r  # compute return
            ep_len += 1
        print("[Evaluate] ep_ret:[%.4f] ep_len:[%d]" % (ep_ret, ep_len))
print("Done.")

# Close
eval_env.close()
ray.shutdown()