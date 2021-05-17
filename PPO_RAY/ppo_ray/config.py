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
