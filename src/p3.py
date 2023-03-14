import os
import time
import pickle
import parser
from copy import deepcopy

import gym
import ray
import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import logz
from p3_utils import compute_kl_similarity, batched_weighted_sum, \
  compute_centered_ranks, fast_calculate_pareto_fronts, FrankWolfeSolver
from shared_noise import *
from modelhub import LinearPolicy, MlpPolicy, get_model_env
from filter import get_filter
from numpy.random import default_rng
from static.infos import get_scores
from optimizers import get_optimizer


@ray.remote
class Worker(object):
  """ 
  Object class for parallel rollout generation.
  """
  def __init__(self,
               worker_seed,
               env_name='',
               fake_env_name=None,
               quality=None,
               mean_obs=None,
               std_obs=None,
               policy_params=None,
               deltas=None,
               rollout_length=1000,
               repeat_num=3,
               delta_std=0.02,
               init_obs=None):

    # initialize MuJoCo environment on each worker for evaluating
    self.env = gym.make(env_name)
    self.env.seed(worker_seed)
    obs_dim = self.env.observation_space.shape[0]
    act_dim = self.env.action_space.shape[0]
    # state normalizer for each work
    self.obs_filter = get_filter(policy_params['ob_filter'], 
                                 shape=(obs_dim,), 
                                 mean=mean_obs, 
                                 std=std_obs)
    # each worker gets access to the shared noise table
    # with independent random streams for sampling
    # from the shared noise table. 
    self.noises = SharedNoiseTable(deltas, worker_seed + 7)
    
    # initialize policy 
    if policy_params['type'] == 'linear':
      self.policy = LinearPolicy(obs_dim, act_dim)
    elif policy_params['type'] == 'mlp':
      self.policy = MlpPolicy(obs_dim, policy_params['hidden'], act_dim)
    else:
      raise NotImplementedError
      
    self.delta_std = delta_std
    self.rollout_length = rollout_length

    # create fake environment
    if fake_env_name is not None:
      self.fake_env = get_model_env(env=fake_env_name, quality=quality)
      print("created model environment.")
    else:
      self.fake_env = None
    
    self.rng = default_rng(worker_seed)
    self.init_obs = init_obs
    self.repeat_num = repeat_num
    
    self.r_filter = lambda x: np.clip(
      np.where(np.logical_or(np.isnan(x), np.isinf(x)), 0, x),
      0,
      20.0
    )
    self.u_filter = lambda x: np.clip(
      np.where(np.logical_or(np.isnan(x), np.isinf(x)), 25, x), 
      0, 
      25.0
    )

  def rollout(self, shift = 0., rollout_length = None, evaluate = False):
    """
    Performs one rollout of maximum length rollout_length. 
    At each time-step it substracts shift from the reward.
    """
    if rollout_length is None:
      rollout_length = self.rollout_length
    
    episodic_r = 0.
    episodic_u = 0.
    steps = 0
    if self.fake_env is not None and not evaluate:
      for _ in range(self.repeat_num):
        
        idx = self.rng.integers(0, self.init_obs.shape[0])
        ob = self.init_obs[idx]
        for _ in range(rollout_length):
          ob_norm = self.obs_filter(ob, update=self.update_filter)
          action = self.policy(torch.from_numpy(ob_norm).float()).detach().numpy()
          ob, reward, done, inf = self.fake_env.step(ob, action)
          
          # clipping           
          ob = np.clip(ob, -100, 100)
          reward = self.r_filter(reward)
          uncertainty = self.u_filter(inf['penalty'][0][0])
          
          episodic_r += (reward - shift)
          episodic_u += np.exp(-uncertainty / 1.5)
          steps += 1
          if done:
            break
      episodic_r /= self.repeat_num
      episodic_u /= self.repeat_num
      
    else:
      ob = self.env.reset()
      for _ in range(rollout_length):
        
        ob_norm = self.obs_filter(ob, update=self.update_filter)
        action = self.policy(torch.from_numpy(ob_norm).float()).detach().numpy()
        ob, reward, done, _ = self.env.step(action)
        steps += 1
        episodic_r += (reward - shift)
        episodic_u += np.exp(0 / 1.5)
        if done:
          break
      
    return episodic_r, episodic_u, steps

  def do_rollouts(self, 
                  w_policy, 
                  num_rollouts=1, 
                  shift=1, 
                  evaluate=False, 
                  real_eval=True):
    """ 
    Generate multiple rollouts with the policy parametrized by w_policy.
    """
    steps = 0
    collect_r, collect_u, deltas_idx = [], [], []
    
    for _ in range(num_rollouts):

      if evaluate:
        vector_to_parameters(torch.from_numpy(np.array(w_policy)).float(), 
                             self.policy.parameters())
        deltas_idx.append(-1)
        
        # set to false so that evaluation rollouts are 
        # not used for updating state statistics
        self.update_filter = False

        # for evaluation we do not shift the rewards (shift = 0) and 
        # we use the default rollout length (1000 for the MuJoCo tasks)
        if real_eval:
          shift4eval = 0.0
        else:
          shift4eval = shift
        episodic_r, episodic_u, _ = self.rollout(shift=shift4eval, 
                                                 rollout_length=1000, 
                                                 evaluate=real_eval)
        collect_r.append(episodic_r)
        collect_u.append(episodic_u)
        
      else:
        idx, delta = self.noises.get_delta(w_policy.size)
        delta = (self.delta_std * delta).reshape(w_policy.shape)
        deltas_idx.append(idx)

        # set to true so that state statistics are updated 
        self.update_filter = True

        # compute reward and number of timesteps 
        # used for positive perturbation rollout
        vector_to_parameters(torch.from_numpy(w_policy + delta).float(), 
                             self.policy.parameters())
        pos_ep_r, pos_ep_u, pos_steps  = self.rollout(shift = shift)

        # compute reward and number of timesteps 
        # used for negative pertubation rollout
        vector_to_parameters(torch.from_numpy(w_policy - delta).float(), 
                             self.policy.parameters())
        neg_ep_r, neg_ep_u, neg_steps = self.rollout(shift = shift)
        steps += (pos_steps + neg_steps)

        # not_update = False
        # if np.isnan(pos_ep_r) or np.isnan(neg_ep_r) or pos_ep_r>2e5 or neg_ep_r>2e5:
        #   not_update = True
        # if np.isnan(pos_ep_u) or np.isnan(neg_ep_u) or pos_ep_u<-2e5 or neg_ep_u<-2e5:
        #   not_update = True
        collect_r.append([pos_ep_r, neg_ep_r])
        collect_u.append([pos_ep_u, neg_ep_u])
              
    return {'deltas_idx': deltas_idx, 
            'collect_r': collect_r, 
            'collect_u': collect_u, 
            'steps' : steps}


def aggre_dataset(env, quality):
  # new dataset loading from the repository of MOREL
  dir_dataset = '/home/yijunyan/Data/PyCode/ModelARS/datasets/{}-{}-v2.pickle'.format(env, quality)
  paths = pickle.load(open(dir_dataset, 'rb'))
  init_obs = np.array([p['observations'][0] for p in paths])
  obs = np.concatenate([p['observations'] for p in paths])
  ats = np.concatenate([p['actions'] for p in paths])
  assert obs.shape[0] == ats.shape[0]
  return obs, init_obs, ats

  
class P3Learner(object):
  """ 
  Object class implementing the P3 algorithm.
  """
  def __init__(self, 
               policy_params=None,
               num_workers=32, 
               num_deltas=320, 
               delta_std=0.02, 
               logdir=None, 
               rollout_length=1000,
               step_size=0.01,
               shift='constant zero',
               psi=-1e-3,
               params=None,
               seed=123):

    # logging
    self.logz = logz
    self.logz.configure_output_dir(logdir)
    self.logz.save_params(params)
    
    env = gym.make(params['env_name'])
    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    
    self.timesteps = 0
    self.num_deltas = num_deltas
    self.rollout_length = rollout_length
    self.step_size = step_size
    self.delta_std = delta_std
    self.logdir = logdir
    self.shift = shift
    self.psi = psi
    self.params = params
    self.num_pareto = params['num_pareto']

    # create shared table for storing noise
    print("Creating deltas table.")
    deltas_id = create_shared_noise.remote()
    self.deltas = SharedNoiseTable(ray.get(deltas_id), seed = seed + 3)

    # initialize policy 
    if policy_params['type'] == 'linear':
      self.policy = LinearPolicy(obs_dim, act_dim)
    elif policy_params['type'] == 'mlp':
      self.policy = MlpPolicy(obs_dim, policy_params['hidden'], act_dim)
    else:
      raise NotImplementedError
    
    # initialize optimization algorithm
    self.flat_weights = parameters_to_vector(self.policy.parameters()).detach().numpy()
    self.optimizer = get_optimizer(params['optim'], self.flat_weights, self.step_size)

    if params['filter'] == 'FixedFilter':
      obs, init_obs, _ = aggre_dataset(params['fake_env_name'], params['quality'])
      mean_obs = np.mean(obs, 0)
      std_obs = np.std(obs, 0)
    else:
      raise NotImplementedError

    init_obs_id = ray.put(init_obs)
    # initialize workers with different random seeds
    print('Initializing workers.')
    self.num_workers = num_workers
    self.workers = [Worker.remote(seed + 7 * i,
                                  env_name=params['env_name'],
                                  fake_env_name=params['fake_env_name'],
                                  quality=params['quality'],
                                  mean_obs=mean_obs,
                                  std_obs=std_obs,
                                  deltas=deltas_id,
                                  delta_std=delta_std,
                                  policy_params=policy_params,
                                  rollout_length=rollout_length,
                                  repeat_num=params['smooth_avg'],
                                  init_obs=init_obs_id) for i in range(num_workers)]

  def aggregate_rollouts(self, 
                         ref_vector,
                         num_rollouts=None, 
                         evaluate=False, 
                         real_eval=True, 
                         eval_policy=None):
    """ 
    Aggregate update step from rollouts generated in parallel.
    """
    if num_rollouts is None:
      num_deltas = self.num_deltas
    else:
      num_deltas = num_rollouts
      
    # put policy weights in the object store
    if eval_policy is not None:
      policy_id = ray.put(eval_policy)
    else:
      policy_id = ray.put(self.flat_weights)

    num_rollouts = int(num_deltas / self.num_workers)

    # parallel generation of rollouts
    rollout_ids_one = [
      worker.do_rollouts.remote(policy_id,
                                num_rollouts=num_rollouts,
                                shift=self.shift,
                                evaluate=evaluate,
                                real_eval=real_eval) 
      for worker in self.workers
    ]

    rollout_ids_two = [
      worker.do_rollouts.remote(policy_id,
                                num_rollouts=1,
                                shift=self.shift,
                                evaluate=evaluate,
                                real_eval=real_eval) 
      for worker in self.workers[:(num_deltas % self.num_workers)]
    ]

    # gather results 
    results_one = ray.get(rollout_ids_one)
    results_two = ray.get(rollout_ids_two)

    collect_r, collect_u, deltas_idx, = [], [], []

    for result in results_one:
      if not evaluate:
        self.timesteps += result["steps"]
      deltas_idx += result['deltas_idx']
      collect_r += result['collect_r']
      collect_u += result['collect_u']

    for result in results_two:
      if not evaluate:
        self.timesteps += result["steps"]
      deltas_idx += result['deltas_idx']
      collect_r += result['collect_r']
      collect_u += result['collect_u']

    deltas_idx = np.array(deltas_idx)
    collect_r = np.array(collect_r)
    collect_u = np.array(collect_u)
    collect_c = compute_kl_similarity(collect_r, collect_u, ref_vector)
    
    avg_return = collect_r.mean()
    avg_uncert = collect_u.mean()
    avg_const = collect_c.mean()

    if evaluate:
      if real_eval:
        return collect_r
      else:
        return collect_r, collect_u, collect_c
    else:
      point_results = self.workers[0].do_rollouts.remote(policy_id,
                                                         num_rollouts=1,
                                                         shift=self.shift,
                                                         evaluate=True,
                                                         real_eval=False)
      point_results = ray.get(point_results)
      point_r = np.array(point_results['collect_r'])
      point_u = np.array(point_results['collect_u'])
      point_c = compute_kl_similarity(point_r, point_u, ref_vector)

      # rank-based scale shaping
      collect_r = compute_centered_ranks(collect_r)
      collect_u = compute_centered_ranks(collect_u)
      collect_c = compute_centered_ranks(collect_c)

      # aggregate rollouts to form g_hat, the gradient used to compute update step
      g_r, _ = batched_weighted_sum(collect_r[:,0] - collect_r[:,1],
                                    (self.deltas.get(idx, self.flat_weights.size)
                                     for idx in deltas_idx),
                                    batch_size = 500)
      g_u, _ = batched_weighted_sum(collect_u[:,0] - collect_u[:,1],
                                    (self.deltas.get(idx, self.flat_weights.size)
                                     for idx in deltas_idx),
                                    batch_size = 500)
      g_c, _ = batched_weighted_sum(collect_c[:,0] - collect_c[:,1],
                                    (self.deltas.get(idx, self.flat_weights.size)
                                     for idx in deltas_idx),
                                    batch_size = 500)
      g_r /= deltas_idx.size
      g_u /= deltas_idx.size
      g_c /= deltas_idx.size

    if point_c < self.psi:
      if (point_r[0] / point_u[0]) < (ref_vector[0] / ref_vector[1]):
        g_hat = g_r
        stamp = 'r'
        alphas = [1,0,0]
      else:
        g_hat = g_u
        stamp = 'u'
        alphas = [0,1,0]
    else:
      g_stack = np.stack((g_r, g_u, g_c))
      alphas = FrankWolfeSolver(g_stack)
      g_hat = alphas[0] * g_r + alphas[1] * g_u + alphas[2] * g_c
      stamp = 'r + u + c'

    print(stamp, ' Alpha:', alphas)
    return g_hat, avg_return, avg_uncert, avg_const

  def train_step(self, ref_vector):
    """ 
    Perform one update step of the policy weights.
    """
    g_hat, avg_return, avg_uncert, avg_const = self.aggregate_rollouts(ref_vector)
    print(f"Norm of grads {np.linalg.norm(g_hat):.4f}")                  
    self.flat_weights -= self.optimizer._compute_step(g_hat)

    return avg_return, avg_uncert, avg_const

  def train(self, num_refs, iter_g, iter_l):
    # load reference scores:
    random_score, expert_score = get_scores(self.params['fake_env_name'])
    start = time.time()
    policy_pool = []
    fitness_pool = []
    returns = np.zeros(self.num_pareto)

    iters = 0
    num_p = 0
    for i in range(num_refs):
      # initialize policy network's weights
      if self.params['bc_init']:
        if params['filter'] == 'FixedFilter':
          bc_weights = np.load(
            "/home/yijunyan/Data/PyCode/ModelARS/clone_policies_v3/{}_{}_{}_BCpolicy.npy".format(self.params['fake_env_name'], 
            self.params['quality'], self.params['policy_type'])
          )
        else:
          raise NotImplementedError
        self.flat_weights = deepcopy(bc_weights)
      else:
        # zeroize weights
        self.flat_weights = np.zeros_like(self.flat_weights, dtype=np.float32)

      tau_c = (0.9 - 0.1) / (num_refs - 1)
      w_1 = 0.9 - i * tau_c
      w_2 = 0.1 + i * tau_c
      ref_vector = np.array([w_1, w_2])
      ref_vector *= np.array([expert_score-self.shift*self.params['rollout_length'], 
                              self.params['rollout_length']])
      
      # Diverse Pareto Policies
      for _ in range(iter_g):
        t1 = time.time()
        avg_return, avg_uncert, avg_const = self.train_step(ref_vector)
        t2 = time.time()
        print(f"Iter {iters+1}, time for one step {t2 - t1:.2f} s")
        
        iters += 1

      # Local Pareto Extension
      for v in range(2):
        ref_vector = np.array([w_1, w_2]) + np.array([-5e-2 * (-1)**v, 
                                                      5e-2 * (-1)**v])
        ref_vector *= np.array([expert_score-self.shift*self.params['rollout_length'], 
                                self.params['rollout_length']])
        for _ in range(iter_l):
          t1 = time.time()
          avg_return, avg_uncert, avg_const = self.train_step(ref_vector)
          t2 = time.time()
          print(f"Iter {iters+1}, time for one step {t2 - t1:.2f} s")
          
          # Store Pareto policies into the pool
          best_policy = np.copy(self.flat_weights)
          point_r, point_u, point_c = self.aggregate_rollouts(ref_vector,
                                                              num_rollouts=3, 
                                                              evaluate=True, 
                                                              real_eval=False, 
                                                              eval_policy=best_policy)
          if num_p >= self.num_pareto:
            current_ret = self.aggregate_rollouts(ref_vector,
                                                  num_rollouts=10, 
                                                  evaluate=True, 
                                                  real_eval=True, 
                                                  eval_policy=best_policy)
            # stacking
            policy_array = np.concatenate((policy_array,
                                           np.expand_dims(best_policy, axis=0)), axis=0)
            fitnesses = np.concatenate((fitnesses, 
                                        np.array([[np.mean(point_r), np.mean(point_u)]])))
            returns = np.append(returns, np.mean(current_ret))
            assert policy_array.shape[0] == self.num_pareto+1 and len(returns) == self.num_pareto+1
            
            # refine pareto front via nondominated sorting
            fronts = fast_calculate_pareto_fronts(fitnesses)

            # select fixed number of pareto policies
            flat_fronts = np.array([item for sublist in fronts for item in sublist])
            policy_array = policy_array[flat_fronts[:self.num_pareto],:]
            fitnesses = fitnesses[flat_fronts[:self.num_pareto],:]
            returns = returns[flat_fronts[:self.num_pareto]]
            assert returns.shape[0] == policy_array.shape[0]
        
            # argmax
            rewards = np.max(returns)
            
          else:
            policy_pool.append(best_policy)
            fitness_pool.append([np.mean(point_r), np.mean(point_u)])
            fitnesses = np.array(fitness_pool)
            policy_array = np.array(policy_pool)
            # evaluation
            current_ret = self.aggregate_rollouts(ref_vector,
                                                  num_rollouts=10, 
                                                  evaluate=True, 
                                                  real_eval=True, 
                                                  eval_policy=best_policy)
            rewards = current_ret
            returns[num_p] = np.mean(current_ret)

          self.logz.log_tabular("Time", time.time() - start)
          self.logz.log_tabular("Iteration", iters + 1)
          self.logz.log_tabular("NormScore", 100*(np.mean(rewards)-random_score)/(expert_score-random_score))
          self.logz.log_tabular("AverageRrturn", np.mean(rewards))
          self.logz.log_tabular("AverageCurRet", np.mean(current_ret))
          self.logz.log_tabular("ave_mean_return", np.mean(point_r))
          self.logz.log_tabular("ave_mean_uncert", np.mean(point_u))
          self.logz.log_tabular("ave_mean_const", np.mean(point_c))
          self.logz.log_tabular("ref_angle", np.arctan(w_2/w_1)/np.pi*180)
          self.logz.log_tabular("StdReturn", np.std(rewards))
          self.logz.log_tabular("MaxReturnRollout", np.max(rewards))
          self.logz.log_tabular("MinReturnRollout", np.min(rewards))
          self.logz.log_tabular("timesteps", self.timesteps)
          self.logz.log_tabular("ave_pops_return", avg_return)
          self.logz.log_tabular("ave_pops_uncert", avg_uncert)
          self.logz.log_tabular("ave_pops_consim", avg_const)
          self.logz.dump_tabular()
          
          iters += 1
          num_p += 1   

      # re-initialize optimization algorithm
      self.optimizer = get_optimizer(params['optim'], self.flat_weights, self.step_size)
                
    return 

def run_ars(params):

  if params['dir_path'] is not None:
    dir_path = params['dir_path']

    if not(os.path.exists(dir_path)):
      os.makedirs(dir_path)
    logdir = dir_path + '/{}_{}/{}_{}/seed_{}'.format(params['env_name'], params['quality'], 
                                                      params['policy_type'], 
                                                      params['rollout_length'], params['seed'])
    if not(os.path.exists(logdir)):
      os.makedirs(logdir)
  else:
    logdir = None

  env = gym.make(params['env_name'])
  obs_dim = env.observation_space.shape[0]
  act_dim = env.action_space.shape[0]

  # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
  policy_params={'type':params['policy_type'],
                 'ob_filter':params['filter'],
                 'ob_dim':obs_dim,
                 'ac_dim':act_dim,
                 'hidden':[32, 32]}

  learner = P3Learner(policy_params=policy_params,
                   num_workers=params['n_workers'], 
                   num_deltas=params['n_directions'],
                   step_size=params['step_size'],
                   delta_std=params['delta_std'], 
                   logdir=logdir,
                   rollout_length=params['rollout_length'],
                   shift=params['shift'],
                   psi=params['psi'],
                   params=params,
                   seed=params['seed'])
    
  learner.train(params['num_refs'], params['iter_g'], params['iter_l'])
     
  return 


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--env_name', type=str, default='Hopper-v3')
  parser.add_argument('--fake_env_name', type=str, default='hopper')
  parser.add_argument('--quality', type=str, default='medium-expert')
  parser.add_argument('--num_pareto', type=int, default=100)
  parser.add_argument('--num_refs', type=int, default=5)
  parser.add_argument('--iter_g', '-n', type=int, default=15)
  parser.add_argument('--iter_l', type=int, default=5)
  parser.add_argument('--optim', type=str, default='Adam')

  parser.add_argument('--step_size', '-s', type=float, default=0.02)
  parser.add_argument('--delta_std', '-std', type=float, default=0.03)
  parser.add_argument('--psi', type=float, default=-1e-3)

  parser.add_argument('--n_directions', '-nd', type=int, default=30)
  parser.add_argument('--n_workers', '-e', type=int, default=30)

  parser.add_argument('--rollout_length', '-r', type=int, default=1000)
  parser.add_argument('--smooth_avg', type=int, default=3)

  # for Swimmer-v1 and HalfCheetah-v1 use shift = 0
  # for Hopper-v1, Walker2d-v1, and Ant-v1 use shift = 1
  # for Humanoid-v1 used shift = 5
  parser.add_argument('--shift', type=float, default=1.0)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--policy_type', type=str, default='mlp')
  parser.add_argument('--bc_init', type=bool, default=True)
  parser.add_argument('--dir_path', type=str, default='logs')

  parser.add_argument('--filter', type=str, default='FixedFilter')

  ray.init()
  
  args = parser.parse_args()
  params = vars(args)
  s = time.time()
  run_ars(params)
  e = time.time()
  print(f"total training time: {e-s:.2f} s")
  ray.shutdown()

