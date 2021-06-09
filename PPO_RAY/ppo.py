import gym,pybullet_envs,time, psutil, torch
from memory import *
from model import *
from config import *

print("Pytorch version:[%s]."%(torch.__version__))
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("device:[%s]."%(device))

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
        #self.actor_critic.load_state_dict(torch.load('model_data/model_weights'))

    def update_ppo(self):
        self.actor_critic.train()

        obs, act, adv, ret, logp = [torch.Tensor(x) for x in self.buf.get()]

        obs = torch.FloatTensor(obs)
        act = torch.FloatTensor(act)
        adv = torch.FloatTensor(adv)
        ret = torch.FloatTensor(ret)
        logp_a_old = torch.FloatTensor(logp)

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

            kl = torch.mean(logp_a_old - logp_a)
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
            #     torch.save(self.actor_critic.state_dict(), 'model_data/model_weights')
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
    env,eval_env = gym.make(env_name), gym.make(env_name)
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
