import torch
import torch.nn as nn


class LinearPolicy(nn.Module):
  def __init__(self, input_s, output_s):
    super(LinearPolicy, self).__init__()
    self.net = nn.Sequential(
          nn.Linear(input_s, output_s, bias=False),
          nn.Tanh())
    
  def forward(self, x):
    return self.net(x)


class MlpPolicy(nn.Module):
  def __init__(self, input_s, hidden_s, output_s):
    super(MlpPolicy, self).__init__()
    self.first_fc = nn.Linear(input_s, hidden_s[0])
    if len(hidden_s) > 1:
      self.hid_fcs = nn.ModuleList([nn.Linear(hidden_s[ind], hidden_s[ind+1]) for ind in range(len(hidden_s)-1)])
    else:
      self.hid_fcs = None
    self.last_fc = nn.Linear(hidden_s[-1], output_s)

  def forward(self, x):
    x = torch.tanh(self.first_fc(x))
    if self.hid_fcs is not None:
      for idx, l in enumerate(self.hid_fcs):
        x = torch.tanh(l(x))
    x = torch.tanh(self.last_fc(x))
    return x


def model_name(env, quality, seed, separate_mean_var):
  name = f'{env}-{quality}'
  if separate_mean_var:
    name += '_smv'
  name += f'_{seed}'
  return name

def get_model_env(seed=0, 
                  env='halfcheetah', 
                  quality='mixed', 
                  hidden_dim=200, 
                  num_networks=7, 
                  num_elites=5, 
                  deterministic=False,
                  model_type='mlp', 
                  separate_mean_var=True, 
                  batch_size=256, 
                  holdout_ratio=0.2, 
                  max_epochs=0, 
                  p_coeff=0.0):
  import os
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  import gym
  import d4rl
  gym.logger.setLevel(40)
  import numpy as np
  # import tensorflow as tf
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
  
  from models.constructor import construct_model
  from models.fake_env import FakeEnv

  tf.set_random_seed(seed)

  env_1 = gym.make(f'{env}-random-v2')
  dataset = env_1.get_dataset()
  obs_dim = dataset['observations'].shape[1]
  act_dim = dataset['actions'].shape[1]

  # index 5 out of 7 models
  model_idxs = {'halfcheetah-random':[3, 4, 5, 2, 1], 
                'halfcheetah-medium':[6, 2, 0, 1, 3], 
                'halfcheetah-expert':[6, 0, 5, 4, 2],
                'halfcheetah-medium-replay':[4, 3, 6, 1, 2], 
                'halfcheetah-medium-expert':[1, 2, 3, 5, 6], 
                'hopper-random':[0, 1, 3, 5, 2],
                'hopper-medium':[0, 3, 5, 4, 6], 
                'hopper-expert':[5, 0, 2, 6, 1], 
                'hopper-medium-replay':[4, 6, 2, 0, 3],
                'hopper-medium-expert':[4, 5, 2, 0, 3], 
                'walker2d-random':[1, 5, 2, 6, 0], 
                'walker2d-medium':[0, 3, 4, 6, 5],
                'walker2d-expert':[1, 0, 6, 5, 2], 
                'walker2d-medium-replay':[1, 4, 2, 0, 3], 
                'walker2d-medium-expert':[5, 6, 3, 4, 1]}
  
  model_dir = '/home/yijunyan/Data/Datasets/d4rl/models_v3'
  model = construct_model(obs_dim=obs_dim, 
                          act_dim=act_dim, 
                          hidden_dim=hidden_dim,
                          num_networks=num_networks, 
                          num_elites=num_elites,
                          model_type=model_type, 
                          separate_mean_var=separate_mean_var,
                          name=model_name(env, quality, seed, separate_mean_var), 
                          load_dir=model_dir, 
                          deterministic=deterministic)
  
  model_idx = model_idxs['{}-{}'.format(env, quality)]
  model.train(inputs=None, 
              targets=None, 
              model_idx=model_idx,
              batch_size=batch_size, 
              holdout_ratio=holdout_ratio,
              max_epochs=max_epochs, 
              max_t=None, 
              load_model=True)
    
  if env == 'halfcheetah':
    from static.halfcheetah import StaticFns
  elif env == 'hopper':
    from static.hopper import StaticFns
  elif env == 'walker2d':
    from static.walker2d import StaticFns
  else:
    raise NotImplementedError
  
  static_fns = StaticFns()
  fake_env = FakeEnv(model, static_fns, penalty_coeff=p_coeff)

  return fake_env


if __name__=="__main__":

  tasks = ['halfcheetah', 'hopper', 'walker2d']
  datasets = ['random', 'medium', 'expert', 'medium-replay', 'medium-expert']

  for t in tasks:
    for d in datasets:
      model = get_model_env(seed=0, env=t, quality=d)
      print(t, d, 'load success!')
  