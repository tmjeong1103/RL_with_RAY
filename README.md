# RL_with_RAY

> Implementation of reinforcement learning algorithms that is applied RAY in PyTorch.
> 
> In this repo, we implemented this RL_with_RAY code only use RAY CORE from scratch, didn't use RLLib.
> 

## Dependencies
1. Python 3.6.x
2. PyTorch 1.8.1
3. RAY 1.3.0
4. pybullet-gym
5. numpy


### Install Requirements
```
pip install -r requirements.txt
```

## Contents

### PPO with RAY
#### Hyperparameter

### SAC with RAY
#### Hyperparameter

### ARS with RAY
#### Hyperparameter
```
- n_cpu = n_workers = 5
- hdims = [256,256]
- actv = nn.Tanh()
- out_actv = nn.Tanh()
```

## Run
### Set configuration
1. n_worker
2. total_steps 
3. ep_len_rollout
4. hdims

## Reference
1. https://docs.ray.io/en/master/index.html, RAY Documentation
2. https://arxiv.org/abs/1707.06347, PPO Paper
3. https://arxiv.org/abs/1801.01290, SAC Paper
4. https://arxiv.org/abs/1803.07055, ARS Paper
