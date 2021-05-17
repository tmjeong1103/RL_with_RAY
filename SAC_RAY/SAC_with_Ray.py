import datetime,gym,os,pybullet_envs,time,os,psutil,ray
import random
from copy import deepcopy
import itertools
from sac import *
from config import *
from matplotlib import pyplot as plt

gym.logger.set_level(40) # gym logger
print("Pytorch version:[%s]."%(torch.__version__))

RENDER_ON_EVAL = False

def get_env():
    import pybullet_envs, gym
    gym.logger.set_level(40) # gym logger
    return gym.make('AntBulletEnv-v0')

def get_eval_env():
    import pybullet_envs, gym
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
    """
    Worker without RAY (for update purposes)
    """
    def __init__(self, seed=1):
        self.seed = seed
        # Each worker should maintain its own environment
        self.env = get_env()
        odim, adim = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.odim = odim
        self.adim = adim

        # Create SAC model and target networks
        self.model = MLPActorCritic(self.odim, self.adim, hdims)

        # # model load
        # self.model.load_state_dict(torch.load('model_data/model_weights_1'))
        # print("weight load")

        self.target = deepcopy(self.model)

        # Freeze target networks with respect to optimizers
        # (only update via polyak averaging)
        for p in self.target.parameters():
            p.requires_grad = False

        # Initialize model
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # parameter chain [q1 + q2]
        self.q_vars = itertools.chain(self.model.q1.parameters(), self.model.q2.parameters())

        # Optimizers
        self.train_pi_op = torch.optim.Adam(self.model.policy.parameters(), lr=lr)
        self.train_q_op = torch.optim.Adam(self.q_vars, lr=lr)

    def get_action(self, o, deterministic=False):
        return self.model.get_action(torch.Tensor(o.reshape(1, -1)), deterministic)

    # get weihts from model and target layer
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
        # import pybullet_envs, gym
        # gym.logger.set_level(40)  # gym logger
        self.env = get_env()
        self.odim = self.env.observation_space.shape[0]
        self.adim = self.env.action_space.shape[0]

        # Replay buffers to pass
        self.o_buffer = np.zeros((self.ep_len_rollout, self.odim))
        self.a_buffer = np.zeros((self.ep_len_rollout, self.adim))
        self.r_buffer = np.zeros((self.ep_len_rollout))
        self.o2_buffer = np.zeros((self.ep_len_rollout, self.odim))
        self.d_buffer = np.zeros((self.ep_len_rollout))

        # Create SAC model and target networks
        self.model = MLPActorCritic(self.odim, self.adim, hdims)
        self.target = deepcopy(self.model)
        print("Ray Worker [%d] Ready." % (self.worker_id))

        # Flag to initialize rollout
        self.FIRST_ROLLOUT_FLAG = True

    def get_action(self, o, deterministic=False):
        return self.model.get_action(torch.Tensor(o.reshape(1, -1)), deterministic)

    def set_weights(self, weight_vals):
        weights = self.model.load_state_dict(weight_vals)
        return weights

    def rollout(self):
        """
        Rollout
        """
        if self.FIRST_ROLLOUT_FLAG:
            self.FIRST_ROLLOUT_FLAG = False
            self.o = self.env.reset()  # reset environment
        # Loop
        for t in range(self.ep_len_rollout):
            self.a = self.get_action(self.o, deterministic=False)
            self.o2, self.r, self.d, _ = self.env.step(self.a.detach().numpy()[0])

            # Append
            self.o_buffer[t, :] = self.o
            self.a_buffer[t, :] = self.a
            self.r_buffer[t] = self.r
            self.o2_buffer[t, :] = self.o2
            self.d_buffer[t] = self.d

            # Save next state
            self.o = self.o2
            if self.d:
                self.o = self.env.reset()  # reset when done
        return self.o_buffer, self.a_buffer, self.r_buffer, self.o2_buffer, self.d_buffer

print("Rollout worker classes (with and without RAY) ready.")

# Initialize PyBullet Ant Environment
eval_env = get_eval_env()
adim = eval_env.action_space.shape[0]
odim = eval_env.observation_space.shape[0]
print("Environment Ready. odim:[%d] adim:[%d]."%(odim,adim))

ray.init(num_cpus=n_cpu,
         _memory = 5*1024*1024*1024,
         object_store_memory = 10*1024*1024*1024,
         _driver_object_store_memory = 1*1024*1024*1024)

R = RolloutWorkerClass(seed=0)
workers = [RayRolloutWorkerClass.remote(worker_id=i, ep_len_rollout=ep_len_rollout)
           for i in range(n_workers)]
print("RAY initialized with [%d] cpus and [%d] workers."%(n_cpu,n_workers))

time.sleep(1)

replay_buffer_long = ReplayBuffer(odim=odim,adim=adim,size=int(buffer_size_long))
replay_buffer_short = ReplayBuffer(odim=odim,adim=adim,size=int(buffer_size_short))

start_time = time.time()
n_env_step = 0  # number of environment steps
step_list = [] # step list for visualization
ep_max_list = [] # step list for visualization
for t in range(int(total_steps)):
    esec = time.time() - start_time

    # Synchronize worker weights
    weights = R.get_weights()
    set_weights_list = [worker.set_weights.remote(weights) for worker in workers]

    # Make rollout and accumulate to Buffers
    ops = [worker.rollout.remote() for worker in workers]
    rollout_vals = ray.get(ops)
    for rollout_val in rollout_vals:
        o_buffer, a_buffer, r_buffer, o2_buffer, d_buffer = rollout_val
        for i in range(ep_len_rollout):
            o, a, r, o2, d = o_buffer[i, :], a_buffer[i, :], r_buffer[i], o2_buffer[i, :], d_buffer[i]
            replay_buffer_long.store(o, a, r, o2, d)
            replay_buffer_short.store(o, a, r, o2, d)
            n_env_step += 1

    # Update
    for _ in range(int(update_count)):

        batch = replay_buffer_long.sample_batch(batch_size//2)
        batch_short = replay_buffer_short.sample_batch(batch_size // 2)

        replay_buffer = dict(obs1=torch.cat((batch['obs1'], batch_short['obs1'])),
                             obs2=torch.cat((batch['obs2'], batch_short['obs2'])),
                             acts=torch.cat((batch['acts'], batch_short['acts'])),
                             rews=torch.cat((batch['rews'], batch_short['rews'])),
                             done=torch.cat((batch['done'], batch_short['done'])))

        # Value train op
        val_loss = R.model.calc_q_loss(target=R.target, data=replay_buffer)
        R.train_q_op.zero_grad()
        val_loss.backward()
        R.train_q_op.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in R.q_vars:
            p.requires_grad = False
        pi_loss = R.model.calc_pi_loss(data=replay_buffer)
        R.train_pi_op.zero_grad()
        pi_loss.backward()
        R.train_pi_op.step()
        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in R.q_vars:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        """moving average update of target networks"""
        with torch.no_grad():
            for v_main, v_targ in zip(R.model.parameters(), R.target.parameters()):
                v_targ.data.copy_(polyak*v_targ.data + (1-polyak)*v_main.data)

    # # save model
    # if t % 200 == 0:
    #     torch.save(R.get_weights(), 'model_data/model_weights_1')
    #     print("Weight saved")

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
                a = R.get_action(o, deterministic=False)
                o, r, d, _ = eval_env.step(a.detach().numpy()[0])
                if RENDER_ON_EVAL:
                    _ = eval_env.render(mode='human')
                ep_ret += r  # compute return
                ep_len += 1
                ep_ret_list.append(ep_ret)  #for visualization
            print("[Evaluate] [%d/%d] ep_ret:[%.4f] ep_len:[%d]"
                  % (eval_idx, num_eval, ep_ret, ep_len))
        ep_max_list.append(max(ep_ret_list)) #for visualization

# make ep_ret-total step graph
plt.plot(step_list,ep_max_list,marker='o')
plt.xlabel('step')
plt.ylabel('ep_return')
plt.title("hdim:%s alpha_pi:[%.4f] alpha_q:[%.4f]\npolyak:[%.4f] gamma:[%.4f] eps:[%.4f] lr:[%.4f]"
          %(hdims,alpha_pi,alpha_q,polyak,gamma,epsilon,lr))
plt.grid(True, linestyle='--')
#plt.show()
plt.savefig('SAC_result.png', dpi=100)

print("Done.")
eval_env.close()
ray.shutdown()
