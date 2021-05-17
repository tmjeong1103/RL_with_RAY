import datetime,gym,os,pybullet_envs,time,os,psutil,ray
import scipy.signal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from gym.spaces import Box, Discrete

print("Pytorch version:[%s]."%(torch.__version__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:[%s]."%(device))

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x
    Args:
        x: An array containing samples of the scalar to produce statistics for.
        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = np.sum(x), len(x)
    mean = global_sum / global_n
    global_sum_sq = np.sum((x - mean)**2)
    std = np.sqrt(global_sum_sq / global_n)  # compute global std
    if with_min_and_max:
        global_min = (np.min(x) if len(x) > 0 else np.inf)
        global_max = (np.max(x) if len(x) > 0 else -np.inf)
        return mean, std, global_min, global_max
    return mean, std

def discount_cumsum(x, discount):
    """
    Compute discounted cumulative sums of vectors.
    input:
        vector x, [x0, x1, x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, odim, adim, size=5000, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, odim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, adim), dtype=np.float32)
        self.act_old_buf = np.zeros(combined_shape(size, adim), dtype=np.float32) # added
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf]

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
        return pi, logp_pi, logp, pi

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
        return pi, logp, logp_pi, mu

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
        pi, mu, logp_pi, logp = self.policy(x, a)
        v = self.vf_mlp(x)
        return pi, logp, logp_pi, v, mu

# Configuration, set model parameter
class Config:
    def __init__(self):
        # Model
        self.hdims = [256, 256]
        #Graph
        self.clip_ratio = 0.2
        self.pi_lr = 3e-4
        self.vf_lr = 1e-3
        #Buffer
        self.steps_per_epoch = 5000
        self.gamma = 0.99
        self.lam = 0.95
        #Update
        self.train_pi_iters = 100
        self.train_v_iters = 100
        self.target_kl = 0.01
        self.epochs = 1000
        self.max_ep_len = 1000
        self.print_every = 10
        self.evaluate_every = 10

class PPOAgent():
    def __init__(self):
        self.config = Config()
        self.env, self.eval_env = get_envs()
        odim = self.env.observation_space.shape[0]
        adim = self.env.action_space.shape[0]

        # Actor-critic model
        ac_kwargs = dict()
        ac_kwargs['action_space'] = self.env.action_space
        self.actor_critic = ActorCritic(odim, adim, self.config.hdims,**ac_kwargs)
        self.buf = PPOBuffer(odim=odim,adim=adim,size=self.config.steps_per_epoch,
                             gamma=self.config.gamma,lam=self.config.lam)

        # Optimizers
        self.train_pi = torch.optim.Adam(self.actor_critic.policy.parameters(), lr=self.config.pi_lr)
        self.train_v = torch.optim.Adam(self.actor_critic.vf_mlp.parameters(), lr=self.config.vf_lr)

        # model load
        self.actor_critic.load_state_dict(torch.load('model_weights2'))

    def update_ppo(self):
        self.actor_critic.train()
        #self.actor_critic.cuda()

        obs, act, adv, ret, logp = [torch.Tensor(x) for x in self.buf.get()]

        obs = torch.FloatTensor(obs).to(device)
        act = torch.FloatTensor(act).to(device)
        adv = torch.FloatTensor(adv).to(device)
        ret = torch.FloatTensor(ret).to(device)
        logp_a_old = torch.FloatTensor(logp).to(device)

        # Policy gradient step
        for i in range(self.config.train_pi_iters):
            _, logp_a, _, _ = self.actor_critic.policy(obs, act)
            # pi, logp, logp_pi, mu

            # PPO objectives
            ratio = (logp_a - logp_a_old).exp()
            min_adv = torch.where(adv > 0, (1 + self.config.clip_ratio) * adv,
                                  (1 - self.config.clip_ratio) * adv)
            pi_loss = -(torch.min(ratio * adv, min_adv)).mean()

            self.train_pi.zero_grad()
            pi_loss.backward()
            self.train_pi.step()

            kl = (logp_a_old - logp_a).mean()
            if kl > 1.5 * self.config.target_kl:
                break

        # Value gradient step
        for _ in range(self.config.train_v_iters):
            v = self.actor_critic.vf_mlp(obs).squeeze()
            v_loss = F.mse_loss(v, ret)

            self.train_v.zero_grad()
            v_loss.backward()
            self.train_v.step()

    def main(self):
        start_time = time.time()
        o, r, d, ep_ret, ep_len, n_env_step = self.env.reset(), 0, False, 0, 0, 0

        self.actor_critic.eval()
        #self.actor_critic.cpu()

        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(self.config.epochs):
            if (epoch == 0) or (((epoch + 1) % self.config.print_every) == 0):
                print("[%d/%d]" % (epoch + 1, self.config.epochs))
            for t in range(self.config.steps_per_epoch):
                a, _, logp_t, v_t, _ = self.actor_critic(
                    torch.Tensor(o.reshape(1, -1)))  # pi, logp, logp_pi, v, mu

                o2, r, d, _ = self.env.step(a.detach().numpy()[0])
                ep_ret += r
                ep_len += 1
                n_env_step += 1

                # save and log  def store(self, obs, act, rew, val, logp):
                self.buf.store(o, a, r, v_t, logp_t)

                # Update obs (critical!)
                o = o2

                terminal = d or (ep_len == self.config.max_ep_len)
                if terminal or (t == (self.config.steps_per_epoch - 1)):
                    # if trajectory didn't reach terminal state, bootstrap value target
                    last_val = 0 if d else self.actor_critic.vf_mlp(torch.Tensor(o.reshape(1, -1))).item()
                    self.buf.finish_path(last_val)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Perform PPO update!
            self.update_ppo()

            # # save model
            # if epoch % 10 == 0:
            #     torch.save(self.actor_critic.state_dict(), 'model_weights2')
            #     print("Weight saved")

            # Evaluate
            self.actor_critic.eval()
            if (epoch == 0) or (((epoch + 1) % self.config.evaluate_every) == 0):
                ram_percent = psutil.virtual_memory().percent  # memory usage
                print("[Eval. start] step:[%d/%d][%.1f%%] #step:[%.1e] time:[%s] ram:[%.1f%%]." %
                      (epoch + 1, self.config.epochs, epoch / self.config.epochs * 100,
                       n_env_step,
                       time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
                       ram_percent)
                      )
                o, d, ep_ret, ep_len = self.eval_env.reset(), False, 0, 0
                _ = self.eval_env.render(mode='human')
                while not (d or (ep_len == self.config.max_ep_len)):
                    a, _, _, _ = self.actor_critic.policy(torch.Tensor(o.reshape(1, -1)))
                    o, r, d, _ = self.eval_env.step(a.detach().numpy()[0])
                    _ = self.eval_env.render(mode='human')
                    ep_ret += r  # compute return
                    ep_len += 1
                print("[Evaluate] ep_ret:[%.4f] ep_len:[%d]" % (ep_ret, ep_len))

        print("Done.")

        self.env.close()
        self.eval_env.close()

    def test(self):
        gym.logger.set_level(40)
        _, eval_env = get_envs()
        o, d, ep_ret, ep_len = eval_env.reset(), False, 0, 0
        _ = eval_env.render(mode='human')
        while not (d or (ep_len == self.config.max_ep_len)):
            a, _, _, _ = self.actor_critic.policy(torch.Tensor(o.reshape(1, -1)))
            o, r, d, _ = eval_env.step(a.detach().numpy()[0])
            _ = eval_env.render(mode='human')
            ep_ret += r  # compute return
            ep_len += 1
        print("[Evaluate] ep_ret:[%.4f] ep_len:[%d]"
              % (ep_ret, ep_len))
        eval_env.close()  # close env


def get_envs():
    env_name = 'AntBulletEnv-v0'
    env,eval_env = gym.make(env_name),gym.make(env_name)
    _ = eval_env.render(mode='human') # enable rendering on test_env
    _ = eval_env.reset()
    for _ in range(3): # dummy run for proper rendering
        a = eval_env.action_space.sample()
        o,r,d,_ = eval_env.step(a)
        time.sleep(0.01)
    return env,eval_env

agent = PPOAgent()
agent.main()
agent.test()