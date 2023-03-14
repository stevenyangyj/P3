# Part of code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


def itergroups(items, group_size):
  assert group_size >= 1
  group = []
  for x in items:
    group.append(x)
    if len(group) == group_size:
      yield tuple(group)
      del group[:]
  if group:
    yield tuple(group)


def batched_weighted_sum(weights, vecs, batch_size):
  total = 0
  num_items_summed = 0
  for batch_weights, batch_vecs in zip(itergroups(weights, batch_size),
                     itergroups(vecs, batch_size)):
    assert len(batch_weights) == len(batch_vecs) <= batch_size
    total += np.dot(np.asarray(batch_weights, dtype=np.float64),
            np.asarray(batch_vecs, dtype=np.float64))
    num_items_summed += len(batch_weights)
  return total, num_items_summed


def compute_ranks(x):
  """Returns ranks in [0, len(x))
  Note: This is different from scipy.stats.rankdata, which returns ranks in
  [1, len(x)].
  """
  assert x.ndim == 1
  ranks = np.empty(len(x), dtype=int)
  ranks[x.argsort()] = np.arange(len(x))
  return ranks

def compute_centered_ranks(x):
  y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
  y /= (x.size - 1)
  y -= 0.5
  return y


def calculate_domination_matrix(fitnesses):    
  
  pop_size = fitnesses.shape[0]
  num_objectives = fitnesses.shape[1]
  
  # print(pop_size, num_objectives)
  fitness_grid_x = np.zeros([pop_size,pop_size,num_objectives])
  fitness_grid_y = np.zeros([pop_size,pop_size,num_objectives])
  
  for i in range(pop_size):
    fitness_grid_x[i,:,:] = fitnesses[i]
    fitness_grid_y[:,i,:] = fitnesses[i]
  
  larger_or_equal = fitness_grid_x >= fitness_grid_y
  larger = fitness_grid_x > fitness_grid_y
  
  return np.logical_and(np.all(larger_or_equal,axis=2),np.any(larger,axis=2))


def fast_calculate_pareto_fronts(fitnesses):
  '''
  https://github.com/adam-katona/NSGA_2_tutorial/blob/master/NSGA_2_tutorial.ipynb
  '''
  # Calculate dominated set for each individual
  domination_sets = []
  domination_counts = []
  
  domination_matrix = calculate_domination_matrix(fitnesses)
  pop_size = fitnesses.shape[0]
  
  for i in range(pop_size):
    current_dimination_set = set()
    domination_counts.append(0)
    for j in range(pop_size):
      if domination_matrix[i,j]:
        current_dimination_set.add(j)
      elif domination_matrix[j,i]:
        domination_counts[-1] += 1
        
    domination_sets.append(current_dimination_set)

  domination_counts = np.array(domination_counts)
  fronts = []
  while True:
    current_front = np.where(domination_counts==0)[0]
    if len(current_front) == 0:
      #print("Done")
      break
    # print("Front: ",current_front)
    fronts.append(current_front)

    for individual in current_front:
      domination_counts[individual] = -1 # this individual is already accounted for, make it -1 so  ==0 will not find it anymore
      dominated_by_current_set = domination_sets[individual]
      for dominated_by_current in dominated_by_current_set:
        domination_counts[dominated_by_current] -= 1
      
  return fronts


# https://github.com/swisscom/ai-research-mamo-framework/blob/master/copsolver/analytical_solver.py
def cop_solver(gradients):

  if gradients is None:
    raise TypeError('Argument: gradients type cannot be None')
  if (len(gradients) != 2):
    raise ValueError('Argument: The number of gradients must be equal to 2')
  if (len(gradients[0]) != len(gradients[1])):
    raise ValueError('Argument: The gradients must have the same length')
  if (gradients[0] == gradients[1]).all():
    return [0.5,0.5]

  alpha = ((gradients[1] - gradients[0]) @ gradients[1]) \
    / ((gradients[0] - gradients[1]) @ (gradients[0] - gradients[1]))

  if alpha < 0:
    alpha = 0
  if alpha > 1:
    alpha = 1

  return [alpha, 1-alpha]

def FrankWolfeSolver(gradients, max_iter=200, min_change=1e-3):
  """ FrankWolfeSolver class. Inherits from the COPSolver
  class.
  FrankWolfeSolver is used to calculate the numerical solutions
  for the QCOP for 2 or more gradients
  Attributes:
  max_iter: max number of iterations for the algorithm
  min_change: minimum change stopping criterion. The algorithms
  stop when the difference between iterations is lower than
  min_change
  """

  if max_iter < 0:
    raise ValueError('Argument, max_iter must be positive')
  if min_change < 0:
    raise ValueError('Arguement: min_change must be positive')

  if gradients is None:
    raise TypeError('Argument: gradients must be set.')
  for gradient in gradients:
    if(len(gradient) != len(gradients[0])):
      raise ValueError('Argument: gradients must have the same length')

  # number of objectives
  n = len(gradients)

  # if there is only 1 gradient, alpha is equal to 1
  if (n == 1):
    return np.ones(1)

  # initialize alphas
  alphas_list = np.ones(n) / n

  # precompute gradient products
  M = gradients @ gradients.T

  for i in range(max_iter):
    # find objective whose gradient gives the smallest product
    # (min row in M)
    min_objective = np.argmin(alphas_list @ M)

    # find min gamma
    # v1 = alphas_list @ gradients   <- combined gradient
    # v2 = gradients[min_objective]  <- min gradient
    v1_v1 = alphas_list @ M @ alphas_list
    v1_v2 = alphas_list @ M[min_objective]
    v2_v2 = M[min_objective, min_objective]
    min_gamma = __min_norm_2(v1_v1, v1_v2, v2_v2)

    # update alpha
    new_alphas_list = min_gamma * alphas_list
    new_alphas_list[min_objective] += 1 - min_gamma

    # if update is smaller than min_change stop
    if np.sum(np.abs(new_alphas_list - alphas_list)) \
        < min_change:
      return new_alphas_list
    else:
      alphas_list = new_alphas_list
    # debug
    # sum_g = 0
    # for idg in range(n):
    #     sum_g += gradients[idg]*alphas_list[idg]
    # print("L2 norm of gradients:", np.linalg.norm(sum_g))

  return alphas_list

def __min_norm_2(v1_v1, v1_v2, v2_v2):
  """Helper function for the Frank Wolve Solver, compute the min alpha of
  the norm squared between two gradients
  .. math::
    \alpha = \frac{(\overline{\theta} - \theta)^{T} \overline{\theta}}
            {\|\theta - \overline{\theta}\|_{2}^{2}}
    where v1_v1 = \theta^{T}\theta
      v1_v2 = \theta^{T}\overline{\theta}
      v2_v2 = \overline{\theta}^{T}\overline{\theta}
  source: https://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization.pdf
  (page 5, alg 1)
  """
  if v1_v1 <= v1_v2:
    return 1.0
  if v2_v2 <= v1_v2:
    return 0.0
  # calculate and return alpha
  return (v2_v2 - v1_v2) / (v1_v1 + v2_v2 - 2 * v1_v2)

def compute_kl_similarity(rollouts_r, rollouts_u, ref_vector):
  def kl(v, v_ref):
    v = v * (1./v_ref)
    v = v / np.expand_dims(np.sum(v, axis=1), axis=1)
    y = []
    for i in range(v.shape[0]):
      if np.all(v[i]>0):
        y.append(-np.sum(v[i] * np.log(v[i] / np.array([0.5, 0.5]))))
      else:
        y.append(-1e5)
    assert len(y) == v.shape[0]
    return np.array(y)
    
  if len(rollouts_r.shape) > 1:
    col_vector_one = np.column_stack((rollouts_r[:,0], rollouts_u[:,0]))
    col_vector_two = np.column_stack((rollouts_r[:,1], rollouts_u[:,1]))

    kl_sim_one = kl(col_vector_one, ref_vector)
    kl_sim_two = kl(col_vector_two, ref_vector)
    return np.column_stack((kl_sim_one, kl_sim_two))
  else:
    col_vector = np.column_stack((rollouts_r, rollouts_u))
    kl_sim = kl(col_vector, ref_vector)
    return kl_sim


if __name__=="__main__":

  r = np.array([[10,8000], [12,-1000], [5000,8000]])
  u = np.array([[0.7,500],[1,900],[360,500]])
  v = np.array([1200/1300, 100/1300])

  print(compute_kl_similarity(r,u,v))

  # test one column
  r = np.array([6])
  u = np.array([1])

  print(compute_kl_similarity(r,u,v))
  # a = np.array([10000, 700])
  # theta = 180 * np.arccos(np.inner(a, v) / (np.linalg.norm(a)*np.linalg.norm(v))) / np.pi
  # dist = np.linalg.norm(a-v)
  # print(np.exp(-theta*dist/tau))