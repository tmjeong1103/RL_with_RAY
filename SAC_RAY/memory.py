import numpy as np


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

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
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def get(self):
        names = ['obs1_buf', 'obs2_buf', 'acts_buf', 'rews_buf', 'done_buf',
                 'ptr', 'size', 'max_size']
        vals = [self.obs1_buf, self.obs2_buf, self.acts_buf, self.rews_buf, self.done_buf,
                self.ptr, self.size, self.max_size]
        return names, vals

    def restore(self, a):
        self.obs1_buf = a[0]
        self.obs2_buf = a[1]
        self.acts_buf = a[2]
        self.rews_buf = a[3]
        self.done_buf = a[4]
        self.ptr = a[5]
        self.size = a[6]
        self.max_size = a[7]
