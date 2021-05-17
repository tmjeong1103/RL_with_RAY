import datetime,gym,os,pybullet_envs,time,os,psutil,ray
import random
from memory import *
from model import *

print("Pytorch version:[%s]."%(torch.__version__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:[%s]."%(device))

np.set_printoptions(precision=2)
gym.logger.set_level(40) # gym logger
print("Pytorch version:[%s]."%(torch.__version__))

# Rollout Worker
def get_env():
    import pybullet_envs,gym
    gym.logger.set_level(40) # gym logger
    return gym.make('AntBulletEnv-v0')

def get_eval_env():
    import pybullet_envs,gym
    gym.logger.set_level(40) # gym logger
    eval_env = gym.make('AntBulletEnv-v0')
    _ = eval_env.render(mode='human') # enable rendering
    _ = eval_env.reset()
    for _ in range(3): # dummy run for proper rendering
        a = eval_env.action_space.sample()
        o,r,d,_ = eval_env.step(a)
        time.sleep(0.01)
    return eval_env

class Config:
    def __init__(self):
        # Model
        self.hdims = [64, 64]
        #Graph
        self.clip_ratio = 0.2
        self.pi_lr = 3e-4
        self.vf_lr = 1e-3
        self.epsilon = 1e-2
        #Buffer
        self.gamma = 0.99
        self.lam = 0.95
        #Update
        self.train_pi_iters = 100
        self.train_v_iters = 100
        self.target_kl = 0.01
        self.epochs = 1000
        self.max_ep_len = 1000
        #Worker
        self.n_cpu = self.n_workers = 10
        self.total_steps = 1000
        self.evaluate_every = 50
        self.print_every = 10
        self.ep_len_rollout = 500
        self.batch_size = 4096

class RolloutWorkerClass(object):
    """
    Worker without RAY (for update purposes)
    """
    def __init__(self, seed=1):
        self.seed = seed
        # Each worker should maintain its own environment
        import pybullet_envs, gym

        gym.logger.set_level(40)  # gym logger
        self.env = get_env()
        odim, adim = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.odim = odim
        self.adim = adim
        self.config = Config()

        # Initialize PPO
        # Actor-critic model
        ac_kwargs = dict()
        ac_kwargs['action_space'] = self.env.action_space
        self.model = ActorCritic(odim, adim, self.config.hdims,**ac_kwargs)

        # Initialize model
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Optimizers
        self.train_pi = torch.optim.Adam(self.model.policy.parameters(), lr=self.config.pi_lr)
        self.train_v = torch.optim.Adam(self.model.vf_mlp.parameters(), lr=self.config.vf_lr)

        ## Flag to initialize assign operations for 'set_weights()'
        #self.FIRST_SET_FLAG = True

    def get_action(self, o, deterministic=False):
        pi, _, _, mu = self.model.policy(torch.Tensor(o.reshape(1, -1)))
        if deterministic:
            return mu
        else:
            return pi

    def get_weights(self):
        """
        Get weights
        """
        weight_vals = self.model.state_dict()
        return weight_vals

    def set_weights(self, weight_vals):
        """
        Set weights without memory leakage
        """
        return self.model.load_state_dict(weight_vals)

@ray.remote
class RayRolloutWorkerClass(object):
    """
    Rollout Worker with RAY
    """
    def __init__(self, worker_id=0, ep_len_rollout=1000):
        self.config = Config()
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
        self.model = ActorCritic(odim, adim, self.config.hdims,**ac_kwargs)

        # Buffer
        self.buf = PPOBuffer(odim=self.odim, adim=self.adim,
                             size=self.ep_len_rollout, gamma=self.config.gamma, lam=self.config.lam)

        ## Flag to initialize assign operations for 'set_weights()'
        #self.FIRST_SET_FLAG = True

        # Flag to initialize rollout
        self.FIRST_ROLLOUT_FLAG = True

    def get_action(self, o, deterministic=False):
        pi, _, _, mu = self.model.policy(torch.Tensor(o.reshape(1, -1)))
        if deterministic:
            return mu
        else:
            return pi

    def set_weights(self, weight_vals):
        """
        Set weights without memory leakage
        """
        return self.model.load_state_dict(weight_vals)

    def rollout(self):
        """
        Rollout
        """
        if self.FIRST_ROLLOUT_FLAG:
            self.FIRST_ROLLOUT_FLAG = False
            self.o = self.env.reset()  # reset environment
        # Loop
        for t in range(self.ep_len_rollout):
            a, _, logp_t, v_t, _ = self.model(
                torch.Tensor(self.o.reshape(1, -1)))  # pi, logp, logp_pi, v, mu

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
print ("Environment Ready. odim:[%d] adim:[%d]."%(odim,adim))
config = Config()

# Initialize Workers
ray.init(num_cpus=config.n_cpu,
         _memory = 5*1024*1024*1024,
         object_store_memory = 10*1024*1024*1024,
         _driver_object_store_memory = 1*1024*1024*1024)

R = RolloutWorkerClass(seed=0)
workers = [RayRolloutWorkerClass.remote(worker_id=i,ep_len_rollout=config.ep_len_rollout)
           for i in range(int(config.n_workers))]
print ("RAY initialized with [%d] cpus and [%d] workers."%
       (config.n_cpu,config.n_workers))

time.sleep(1)

# Loop
start_time = time.time()
n_env_step = 0  # number of environment steps

for t in range(int(config.total_steps)):
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
    for pi_iter in range(int(config.train_pi_iters)):
        rand_idx = np.random.permutation(n_val_total)[:config.batch_size]
        buf_batches = [obs_bufs[rand_idx], act_bufs[rand_idx], adv_bufs[rand_idx],
                       ret_bufs[rand_idx], logp_bufs[rand_idx]]

        obs, act, adv, ret, logp = [torch.Tensor(x) for x in buf_batches]
        ent = (-logp).mean()

        obs = torch.FloatTensor(obs).to(device)
        act = torch.FloatTensor(act).to(device)
        adv = torch.FloatTensor(adv).to(device)
        ret = torch.FloatTensor(ret).to(device)
        logp_a_old = torch.FloatTensor(logp).to(device)

        _, logp_a, _, _ = R.model.policy(obs, act)

        # PPO objectives
        ratio = (logp_a - logp_a_old).exp()
        min_adv = torch.where(adv > 0, (1 + config.clip_ratio) * adv,
                              (1 - config.clip_ratio) * adv)
        pi_loss = -(torch.min(ratio * adv, min_adv)).mean()

        R.train_pi.zero_grad()
        pi_loss.backward()
        R.train_pi.step()

        # a sample estimate for KL-divergence
        kl = torch.mean(logp_a_old - logp_a)
        if kl > 1.5 * config.target_kl:
            #print("  pi_iter:[%d] kl(%.3f) is higher than 1.5x(%.3f)"%(pi_iter,kl,config.target_kl))
            break

    # Value gradient step
    for _ in range(int(config.train_v_iters)):
        rand_idx = np.random.permutation(n_val_total)[:config.batch_size]
        buf_batches = [obs_bufs[rand_idx], act_bufs[rand_idx], adv_bufs[rand_idx],
                       ret_bufs[rand_idx], logp_bufs[rand_idx]]

        obs, act, adv, ret, logp = [torch.Tensor(x) for x in buf_batches]

        v = R.model.vf_mlp(obs).squeeze()
        v_loss = F.mse_loss(v, ret)

        R.train_v.zero_grad()
        v_loss.backward()
        R.train_v.step()

    sec_update = time.time() - t_start  # toc

    # Print
    if (t == 0) or (((t + 1) % config.print_every) == 0):
        print("[%d/%d] rollout:[%.1f]s pi_iter:[%d/%d] update:[%.1f]s kl:[%.4f] target_kl:[%.4f]." %
              (t + 1, config.total_steps, sec_rollout, pi_iter, config.train_pi_iters, sec_update, kl, config.target_kl))
        print("   pi_loss:[%.4f], entropy:[%.4f]" %
              (pi_loss, ent))

    # Evaluate
    if (t == 0) or (((t + 1) % config.evaluate_every) == 0):
        ram_percent = psutil.virtual_memory().percent  # memory usage
        print("[Eval. start] step:[%d/%d][%.1f%%] #step:[%.1e] time:[%s] ram:[%.1f%%]." %
              (t + 1, config.total_steps, t / config.total_steps * 100,
               n_env_step,
               time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
               ram_percent)
              )
        o, d, ep_ret, ep_len = eval_env.reset(), False, 0, 0
        _ = eval_env.render(mode='human')
        while not (d or (ep_len == config.max_ep_len)):
            a, _, _, _ = R.model.policy(torch.Tensor(o.reshape(1, -1)))
            o, r, d, _ = eval_env.step(a.detach().numpy()[0])
            _ = eval_env.render(mode='human')
            ep_ret += r  # compute return
            ep_len += 1
        print("[Evaluate] ep_ret:[%.4f] ep_len:[%d]" % (ep_ret, ep_len))

print("Done.")

# Close
eval_env.close()
ray.shutdown()