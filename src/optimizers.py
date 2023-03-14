# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# OPTIMIZERS FOR MINIMIZING OBJECTIVES
class Optimizer(object):
  def __init__(self, w_policy):
    self.dim = w_policy.size
    self.t = 0

  def _compute_step(self, globalg):
    raise NotImplementedError


class BasicSGD(Optimizer):
  def __init__(self, pi, stepsize):
    Optimizer.__init__(self, pi)
    self.stepsize = stepsize

  def _compute_step(self, globalg):
    self.t += 1
    step = -self.stepsize * globalg
    return step


class SGD(Optimizer):
  def __init__(self, pi, stepsize, momentum=0.9):
    Optimizer.__init__(self, pi)
    self.v = np.zeros(self.dim, dtype=np.float32)
    self.stepsize = stepsize
    self.momentum = momentum

  def _compute_step(self, globalg):
    self.t += 1
    self.v = self.momentum * self.v + (1.-self.momentum) * globalg
    step = -self.stepsize * self.v
    return step


class Adam(Optimizer):
  def __init__(self, pi, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08):
    Optimizer.__init__(self, pi)
    self.stepsize = stepsize
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.m = np.zeros(self.dim, dtype=np.float32)
    self.v = np.zeros(self.dim, dtype=np.float32)

  def _compute_step(self, globalg):
    self.t += 1
    a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
    self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
    self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
    step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
    return step


def get_optimizer(opt_name, parameters, lr):
  if opt_name == 'BasicSGD':
      optimizer = BasicSGD(parameters, lr)
  elif opt_name == 'SGD':
      optimizer = SGD(parameters, lr)
  elif opt_name == 'Adam':
      optimizer = Adam(parameters, lr)
  else:
      raise NotImplementedError
  return optimizer

  

