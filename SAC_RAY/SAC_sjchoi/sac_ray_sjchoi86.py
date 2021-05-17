import datetime,gym,os,pybullet_envs,time,os,psutil,ray
import numpy as np
import tensorflow as tf
from util import gpu_sess,suppress_tf_warning
from sac import ReplayBuffer,create_sac_model,create_sac_graph,\
    save_sac_model_and_buffers,restore_sac_model_and_buffers
np.set_printoptions(precision=2)
suppress_tf_warning() # suppress warning
gym.logger.set_level(40) # gym logger
print ("Packaged loaded. TF version is [%s]."%(tf.__version__))

RENDER_ON_EVAL = True

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
    """
    Worker without RAY (for update purposes)
    """

    def __init__(self, lr=1e-3, gamma=0.99, alpha_q=0.1, alpha_pi=0.1, polyak=0.995, seed=1):
        self.seed = seed
        # Each worker should maintain its own environment
        # import pybullet_envs,gym
        from util import suppress_tf_warning
        suppress_tf_warning()  # suppress TF warnings
        # gym.logger.set_level(40) # gym logger
        # self.env = gym.make('AntBulletEnv-v0')
        self.env = get_env()
        odim, adim = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.odim = odim
        self.adim = adim

        # Create SAC model and computational graph
        self.model, self.sess = create_sac_model(odim=self.odim, adim=self.adim)
        self.step_ops, self.target_init = \
            create_sac_graph(self.model, lr=lr, gamma=gamma, alpha_q=alpha_q, alpha_pi=alpha_pi, polyak=polyak)

        # Initialize model
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init)

        # Flag to initialize assign operations for 'set_weights()'
        self.FIRST_SET_FLAG = True

    def get_action(self, o, deterministic=False):
        act_op = self.model['mu'] if deterministic else self.model['pi']
        return self.sess.run(act_op, feed_dict={self.model['o_ph']: o.reshape(1, -1)})[0]

    def get_weights(self):
        """
        Get weights
        """
        weight_vals = self.sess.run(self.model['main_vars'] + self.model['target_vars'])
        return weight_vals

    def set_weights(self, weight_vals):
        """
        Set weights without memory leakage
        """
        if self.FIRST_SET_FLAG:
            self.FIRST_SET_FLAG = False
            self.assign_placeholders = []
            self.assign_ops = []
            for w_idx, weight_tf_var in enumerate(self.model['main_vars'] + self.model['target_vars']):
                a = weight_tf_var
                assign_placeholder = tf.placeholder(a.dtype, shape=a.get_shape())
                assign_op = a.assign(assign_placeholder)
                self.assign_placeholders.append(assign_placeholder)
                self.assign_ops.append(assign_op)
        for w_idx, weight_tf_var in enumerate(self.model['main_vars'] + self.model['target_vars']):
            self.sess.run(self.assign_ops[w_idx],
                          {self.assign_placeholders[w_idx]: weight_vals[w_idx]})


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
        # import pybullet_envs,gym
        from util import suppress_tf_warning
        suppress_tf_warning()  # suppress TF warnings
        # gym.logger.set_level(40) # gym logger
        # self.env = gym.make('AntBulletEnv-v0')
        self.env = get_env()
        odim, adim = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.odim = odim
        self.adim = adim

        # Replay buffers to pass
        self.o_buffer = np.zeros((self.ep_len_rollout, self.odim))
        self.a_buffer = np.zeros((self.ep_len_rollout, self.adim))
        self.r_buffer = np.zeros((self.ep_len_rollout))
        self.o2_buffer = np.zeros((self.ep_len_rollout, self.odim))
        self.d_buffer = np.zeros((self.ep_len_rollout))

        # Create SAC model
        self.model, self.sess = create_sac_model(odim=self.odim, adim=self.adim)
        self.sess.run(tf.global_variables_initializer())
        print("Ray Worker [%d] Ready." % (self.worker_id))

        # Flag to initialize assign operations for 'set_weights()'
        self.FIRST_SET_FLAG = True

        # Flag to initialize rollout
        self.FIRST_ROLLOUT_FLAG = True

    def get_action(self, o, deterministic=False):
        act_op = self.model['mu'] if deterministic else self.model['pi']
        return self.sess.run(act_op, feed_dict={self.model['o_ph']: o.reshape(1, -1)})[0]

    def set_weights(self, weight_vals):
        """
        Set weights without memory leakage
        """
        if self.FIRST_SET_FLAG:
            self.FIRST_SET_FLAG = False
            self.assign_placeholders = []
            self.assign_ops = []
            for w_idx, weight_tf_var in enumerate(self.model['main_vars'] + self.model['target_vars']):
                a = weight_tf_var
                assign_placeholder = tf.placeholder(a.dtype, shape=a.get_shape())
                assign_op = a.assign(assign_placeholder)
                self.assign_placeholders.append(assign_placeholder)
                self.assign_ops.append(assign_op)
        for w_idx, weight_tf_var in enumerate(self.model['main_vars'] + self.model['target_vars']):
            self.sess.run(self.assign_ops[w_idx],
                          {self.assign_placeholders[w_idx]: weight_vals[w_idx]})

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
            self.o2, self.r, self.d, _ = self.env.step(self.a)
            # Append
            self.o_buffer[t, :] = self.o
            self.a_buffer[t, :] = self.a
            self.r_buffer[t] = self.r
            self.o2_buffer[t, :] = self.o2
            self.d_buffer[t] = self.d
            # Save next state
            self.o = self.o2
            if self.d: self.o = self.env.reset()  # reset when done
        return self.o_buffer, self.a_buffer, self.r_buffer, self.o2_buffer, self.d_buffer


print("Rollout worker classes (with and without RAY) ready.")


eval_env = get_eval_env()
adim,odim = eval_env.action_space.shape[0],eval_env.observation_space.shape[0]
print ("Environment Ready. odim:[%d] adim:[%d]."%(odim,adim))

n_cpu = n_workers = 9
total_steps,evaluate_every = 2000,200 # 2000,200
ep_len_rollout = 100
batch_size,update_count = 128,ep_len_rollout
num_eval,max_ep_len_eval = 3,1e3
buffer_size_long,buffer_size_short = 1e6, 1e5

ray.init(num_cpus=n_cpu,
         _memory = 5*1024*1024*1024,
         object_store_memory = 10*1024*1024*1024,
         _driver_object_store_memory = 1*1024*1024*1024)
tf.reset_default_graph()
R = RolloutWorkerClass(lr=1e-3,gamma=0.99,alpha_q=0.0,alpha_pi=0.1,polyak=0.995,seed=0)
workers = [RayRolloutWorkerClass.remote(worker_id=i,ep_len_rollout=ep_len_rollout)
           for i in range(n_workers)]
print ("RAY initialized with [%d] cpus and [%d] workers."%
       (n_cpu,n_workers))


time.sleep(1)

replay_buffer_long = ReplayBuffer(odim=odim,adim=adim,size=int(buffer_size_long))
replay_buffer_short = ReplayBuffer(odim=odim,adim=adim,size=int(buffer_size_short))

start_time = time.time()
n_env_step = 0  # number of environment steps
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
        batch = replay_buffer_long.sample_batch(batch_size // 2)
        batch_short = replay_buffer_short.sample_batch(batch_size // 2)
        feed_dict = {R.model['o_ph']: np.concatenate((batch['obs1'], batch_short['obs1'])),
                     R.model['o2_ph']: np.concatenate((batch['obs2'], batch_short['obs2'])),
                     R.model['a_ph']: np.concatenate((batch['acts'], batch_short['acts'])),
                     R.model['r_ph']: np.concatenate((batch['rews'], batch_short['rews'])),
                     R.model['d_ph']: np.concatenate((batch['done'], batch_short['done']))
                     }
        outs = R.sess.run(R.step_ops, feed_dict)

    # Evaluate
    if (t == 0) or (((t + 1) % evaluate_every) == 0) or (t == (total_steps - 1)):
        ram_percent = psutil.virtual_memory().percent  # memory usage
        print("[Evaluate] step:[%d/%d][%.1f%%] #step:[%.1e] time:[%s] ram:[%.1f%%]." %
              (t + 1, total_steps, t / total_steps * 100,
               n_env_step,
               time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
               ram_percent)
              )
        for eval_idx in range(num_eval):
            o, d, ep_ret, ep_len = eval_env.reset(), False, 0, 0
            if RENDER_ON_EVAL:
                _ = eval_env.render(mode='human')
            while not (d or (ep_len == max_ep_len_eval)):
                a = R.get_action(o, deterministic=True)
                o, r, d, _ = eval_env.step(a)
                if RENDER_ON_EVAL:
                    _ = eval_env.render(mode='human')
                ep_ret += r  # compute return
                ep_len += 1
            print("[Evaluate] [%d/%d] ep_ret:[%.4f] ep_len:[%d]"
                  % (eval_idx, num_eval, ep_ret, ep_len))

        #     # Save current SAC model and replay buffers
        # npz_path = '../data/net/sac_ant/model_and_buffers.npz'
        # save_sac_model_and_buffers(npz_path, R, replay_buffer_long, replay_buffer_short,
        #                            VERBOSE=False, IGNORE_BUFFERS=True)

print("Done.")

eval_env.close()
ray.shutdown()

# # Path to save the npz file
# npz_path = '../data/net/sac_ant/model_and_buffers_final.npz'
# save_sac_model_and_buffers(npz_path,R,replay_buffer_long,replay_buffer_short,
#                            VERBOSE=False,IGNORE_BUFFERS=True)
# R.sess.run(tf.global_variables_initializer())
#
# # Load npz
# npz_path = '../data/net/sac_ant/model_and_buffers_final.npz'
# restore_sac_model_and_buffers(npz_path,R,replay_buffer_long,replay_buffer_short,VERBOSE=True)
#
# eval_env = get_eval_env()
# o,d,ep_ret,ep_len = eval_env.reset(),False,0,0
# if RENDER_ON_EVAL:
#     _ = eval_env.render(mode='human')
# while not(d or (ep_len == max_ep_len_eval)):
#     a = R.get_action(o,deterministic=True)
#     o,r,d,_ = eval_env.step(a)
#     if RENDER_ON_EVAL:
#         _ = eval_env.render(mode='human')
#     ep_ret += r # compute return
#     ep_len += 1
# print ("[Evaluate] ep_ret:[%.4f] ep_len:[%d]"
#     %(ep_ret,ep_len))
# eval_env.close() # close env

