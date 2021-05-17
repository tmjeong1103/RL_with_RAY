import datetime,gym,os,pybullet_envs,time,os
import numpy as np
import tensorflow as tf
np.set_printoptions(precision=2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
print ("Packaged loaded. TF version is [%s]."%(tf.__version__))

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, odim, adim, size):
        self.obs1_buf = np.zeros([size, odim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, odim], dtype=np.float32)
        self.acts_buf = np.zeros([size, adim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


def create_sac_model(odim=10, adim=2, hdims=[256, 256]):
    """
    Soft Actor Critic Model (compatible with Ray)
    """
    import tensorflow as tf  # make it compatible with Ray actors

    def mlp(x, hdims=[256, 256], actv=tf.nn.relu, out_actv=tf.nn.relu):
        ki = tf.truncated_normal_initializer(stddev=0.1)
        for hdim in hdims[:-1]:
            x = tf.layers.dense(x, units=hdim, activation=actv, kernel_initializer=ki)
        return tf.layers.dense(x, units=hdims[-1], activation=out_actv, kernel_initializer=ki)

    def gaussian_loglik(x, mu, log_std):
        EPS = 1e-8
        pre_sum = -0.5 * (
                ((x - mu) / (tf.exp(log_std) + EPS)) ** 2 +
                2 * log_std + np.log(2 * np.pi)
        )

    def mlp_gaussian_policy(o, adim=2, hdims=[256, 256], actv=tf.nn.relu):
        net = mlp(x=o, hdims=hdims, actv=actv, out_actv=actv)  # feature
        mu = tf.layers.dense(net, adim, activation=None)  # mu
        log_std = tf.layers.dense(net, adim, activation=None)  # log_std
        LOG_STD_MIN, LOG_STD_MAX = -10.0, +2.0
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)  # std
        pi = mu + tf.random_normal(tf.shape(mu)) * std  # sampled
        logp_pi = gaussian_loglik(x=pi, mu=mu, log_std=log_std)  # log lik
        return mu, pi, logp_pi

    def squash_action(mu, pi, logp_pi):
        # Squash those unbounded actions
        logp_pi -= tf.reduce_sum(2 * (np.log(2) - pi - tf.nn.softplus(-2 * pi)), axis=1)
        mu, pi = tf.tanh(mu), tf.tanh(pi)
        return mu, pi, logp_pi

    def mlp_actor_critic(o, a, hdims=[256, 256], actv=tf.nn.relu, out_actv=None,
                         policy=mlp_gaussian_policy):
        adim = a.shape.as_list()[-1]
        with tf.variable_scope('pi'):  # policy
            mu, pi, logp_pi = policy(o=o, adim=adim, hdims=hdims, actv=actv)
            mu, pi, logp_pi = squash_action(mu=mu, pi=pi, logp_pi=logp_pi)

        def vf_mlp(x): return tf.squeeze(
            mlp(x=x, hdims=hdims + [1], actv=actv, out_actv=None), axis=1)

        with tf.variable_scope('q1'): q1 = vf_mlp(tf.concat([o, a], axis=-1))
        with tf.variable_scope('q2'): q2 = vf_mlp(tf.concat([o, a], axis=-1))
        return mu, pi, logp_pi, q1, q2

    def placeholder(dim=None):
        return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))

    def placeholders(*args):
        """
        Usage: a_ph,b_ph,c_ph = placeholders(adim,bdim,None)
        """
        return [placeholder(dim) for dim in args]

    def get_vars(scope):
        return [x for x in tf.compat.v1.global_variables() if scope in x.name]

    # Have own session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Placeholders
    o_ph, a_ph, o2_ph, r_ph, d_ph = placeholders(odim, adim, odim, None, None)
    # Actor critic
    ac_kwargs = {'hdims': hdims, 'actv': tf.nn.relu, 'out_actv': None, 'policy': mlp_gaussian_policy}
    with tf.variable_scope('main'):
        mu, pi, logp_pi, q1, q2 = mlp_actor_critic(o=o_ph, a=a_ph, **ac_kwargs)
    with tf.variable_scope('main', reuse=True):
        _, _, _, q1_pi, q2_pi = mlp_actor_critic(o=o_ph, a=pi, **ac_kwargs)
        _, pi_next, logp_pi_next, _, _ = mlp_actor_critic(o=o2_ph, a=a_ph, **ac_kwargs)
    # Target value
    with tf.variable_scope('target'):
        _, _, _, q1_targ, q2_targ = mlp_actor_critic(o=o2_ph, a=pi_next, **ac_kwargs)

    # Get variables
    main_vars, q_vars, pi_vars, target_vars = \
        get_vars('main'), get_vars('main/q'), get_vars('main/pi'), get_vars('target')

    model = {'o_ph': o_ph, 'a_ph': a_ph, 'o2_ph': o2_ph, 'r_ph': r_ph, 'd_ph': d_ph,
             'mu': mu, 'pi': pi, 'logp_pi': logp_pi, 'q1': q1, 'q2': q2,
             'q1_pi': q1_pi, 'q2_pi': q2_pi,
             'pi_next': pi_next, 'logp_pi_next': logp_pi_next,
             'q1_targ': q1_targ, 'q2_targ': q2_targ,
             'main_vars': main_vars, 'q_vars': q_vars, 'pi_vars': pi_vars, 'target_vars': target_vars}

    return model, sess


def create_sac_graph(model, lr=1e-3, gamma=0.98, alpha=0.1, polyak=0.995):
    """
    SAC Computational Graph
    """
    # Double Q-learning
    min_q_pi = tf.minimum(model['q1_pi'], model['q2_pi'])
    min_q_targ = tf.minimum(model['q1_targ'], model['q2_targ'])

    # Entropy-regularized Bellman backup
    q_backup = tf.stop_gradient(
        model['r_ph'] +
        gamma * (1 - model['d_ph']) * (min_q_targ - alpha * model['logp_pi_next'])
    )

    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(alpha * model['logp_pi'] - min_q_pi)
    q1_loss = 0.5 * tf.reduce_mean((q_backup - model['q1']) ** 2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - model['q2']) ** 2)
    value_loss = q1_loss + q2_loss

    # Policy train op
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=model['pi_vars'])

    # Value train op
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=model['q_vars'])

    # Polyak averaging for target variables
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                  for v_main, v_targ in
                                  zip(model['main_vars'], model['target_vars'])]
                                 )

    # All ops to call during one training step
    step_ops = [pi_loss, q1_loss, q2_loss, model['q1'], model['q2'], model['logp_pi'],
                train_pi_op, train_value_op, target_update]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                            for v_main, v_targ in
                            zip(model['main_vars'], model['target_vars'])]
                           )

    return step_ops, target_init


def get_action(model, sess, o, deterministic=False):
    act_op = model['mu'] if deterministic else model['pi']
    return sess.run(act_op, feed_dict={model['o_ph']: o.reshape(1, -1)})[0]


print("SAC model ready.")

gym.logger.set_level(40)
env_name = 'AntBulletEnv-v0'
env,test_env = gym.make(env_name),gym.make(env_name)
_ = test_env.render(mode='human') # enable rendering on test_env
_ = test_env.reset()
for _ in range(3): # dummy run for proper rendering
    a = test_env.action_space.sample()
    o,r,d,_ = test_env.step(a)
    time.sleep(0.01)
print ("[%s] ready."%(env_name))
observation_space = env.observation_space
action_space = env.action_space # -1.0 ~ +1.0
odim,adim = observation_space.shape[0],action_space.shape[0]
print ("odim:[%d] adim:[%d]."%(odim,adim))


tf.reset_default_graph()
model,sess = create_sac_model(odim=odim,adim=adim)
step_ops,target_init = create_sac_graph(model,lr=1e-3,gamma=0.98,alpha=0.1,polyak=0.995)
# Replay buffers
replay_buffer = ReplayBuffer(odim=odim,adim=adim,size=int(1e6))
replay_buffer_short = ReplayBuffer(odim=odim,adim=adim,size=int(1e5))

# Training configuration
total_steps,start_steps = 1e6,1e4
update_every,update_count,batch_size,max_ep_len_train = 1,2,128,1e3
evaluate_every,num_eval,max_ep_len_test = 1e4,3,1e3

# Fix random seed and initialize the model
seed = 0
tf.set_random_seed(seed)
np.random.seed(seed)
sess.run(tf.global_variables_initializer())
sess.run(target_init)

start_time = time.time()
o, ep_ret, ep_len = env.reset(), 0, 0
for t in range(int(total_steps)):
    zero_to_one = (t / total_steps)
    one_to_zero = 1.0 - zero_to_one
    esec = time.time() - start_time

    # Get action
    if t > start_steps:
        a = get_action(model, sess, o, deterministic=False)
    else:
        a = env.action_space.sample()

    # Step the env
    o2, r, d, _ = env.step(a)
    ep_ret += r
    ep_len += 1
    d = False if ep_len == max_ep_len_train else d  # ignore done if it maxed out

    # Store experience to replay buffers
    replay_buffer.store(o, a, r, o2, d)  # save obs, action, reward, next obs
    replay_buffer_short.store(o, a, r, o2, d)  # save obs, action, reward, next obs
    o = o2  # easy to overlook

    # End of trajectory handling - reset env
    if d or (ep_len == max_ep_len_train):
        o, ep_ret, ep_len = env.reset(), 0, 0

    # Update
    if (t >= start_steps) and (t % update_every == 0):
        for _ in range(update_count):
            batch = replay_buffer.sample_batch(batch_size // 2)
            batch_short = replay_buffer_short.sample_batch(batch_size // 2)
            feed_dict = {model['o_ph']: np.concatenate((batch['obs1'], batch_short['obs1'])),
                         model['o2_ph']: np.concatenate((batch['obs2'], batch_short['obs2'])),
                         model['a_ph']: np.concatenate((batch['acts'], batch_short['acts'])),
                         model['r_ph']: np.concatenate((batch['rews'], batch_short['rews'])),
                         model['d_ph']: np.concatenate((batch['done'], batch_short['done']))
                         }
            outs = sess.run(step_ops, feed_dict=feed_dict)  # train
            q1_val, q2_val = outs[3], outs[4]

    # Evaluate
    if (((t + 1) % evaluate_every) == 0):
        print("[Evaluate] step:[%d/%d][%.1f%%] time:%s." %
              (t + 1, total_steps, zero_to_one * 100,
               time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
              )
        for eval_idx in range(num_eval):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            _ = test_env.render(mode='human')
            while not (d or (ep_len == max_ep_len_test)):
                a = get_action(model, sess, o, deterministic=True)
                o, r, d, _ = test_env.step(a)
                _ = test_env.render(mode='human')
                ep_ret += r  # compute return
                ep_len += 1
            print("[Evaluate] [%d/%d] ep_ret:[%.4f] ep_len:[%d]"
                  % (eval_idx, num_eval, ep_ret, ep_len))

print("Done.")

env.close()
test_env.close()

# gym.logger.set_level(40)
# env_name = 'AntBulletEnv-v0'
# test_env = gym.make(env_name)
# _ = test_env.render(mode='human') # enable rendering on test_env
# _ = test_env.reset()
# for _ in range(3): # dummy run for proper rendering
#     a = test_env.action_space.sample()
#     o,r,d,_ = test_env.step(a)
#     time.sleep(0.01)
# print ("[%s] ready."%(env_name))
# o,d,ep_ret,ep_len = test_env.reset(),False,0,0
# _ = test_env.render(mode='human')
# while not(d or (ep_len == max_ep_len_test)):
#     a = get_action(model,sess,o,deterministic=True)
#     o,r,d,_ = test_env.step(a)
#     _ = test_env.render(mode='human')
#     ep_ret += r # compute return
#     ep_len += 1
# print ("[Evaluate] ep_ret:[%.4f] ep_len:[%d]"
#     %(ep_ret,ep_len))
# test_env.close() # close env
#
# from IPython.display import Video
# Video('../vid/SAC_PyBullet_Ant.mp4')