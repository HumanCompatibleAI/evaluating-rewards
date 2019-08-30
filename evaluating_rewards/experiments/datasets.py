# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Methods to create dataset generators.

These are methods that yield batches of observation-action-next observation
triples, implicitly used to define a distribution which distance metrics
can be taken with respect to.
"""

import math
from typing import Callable, Iterator, Union

from evaluating_rewards import rewards
from evaluating_rewards.envs import point_mass
import gym
from imitation.envs import resettable_env
from imitation.util import rollout
import numpy as np
from stable_baselines.common import base_class
from stable_baselines.common import policies


def dummy_env_and_dataset(dims: int = 5):
  """Make a simple fake environment with rollouts."""
  obs_space = gym.spaces.Box(low=np.repeat(0.0, dims),
                             high=np.repeat(1.0, dims))
  act_space = gym.spaces.Box(low=np.repeat(0.0, dims),
                             high=np.repeat(1.0, dims))

  def dataset_generator(total_timesteps, batch_size):
    nbatches = math.ceil(total_timesteps / batch_size)
    for _ in range(nbatches):
      old_obs = np.array([obs_space.sample() for _ in range(batch_size)])
      actions = np.array([act_space.sample() for _ in range(batch_size)])
      new_obs = (old_obs + actions).clip(0.0, 1.0)
      yield rewards.Batch(old_obs=old_obs,
                          actions=actions,
                          new_obs=new_obs)

  return {
      "observation_space": obs_space,
      "action_space": act_space,
      "dataset_generator": dataset_generator,
  }


BatchCallable = Callable[[int, int], Iterator[rewards.Batch]]


def rollout_generator(env: gym.Env,
                      policy: Union[base_class.BaseRLModel,
                                    policies.BasePolicy],
                     ) -> BatchCallable:
  """Generator returning rollouts from a policy in a given environment."""
  def f(total_timesteps: int, batch_size: int) -> Iterator[rewards.Batch]:
    nbatch = math.ceil(total_timesteps / batch_size)
    for _ in range(nbatch):
      res = rollout.generate_transitions(policy, env, n_timesteps=batch_size,
                                         truncate=True)
      old_obs, actions, new_obs, _ = res
      yield rewards.Batch(old_obs=old_obs,
                          actions=actions,
                          new_obs=new_obs)
  return f


def random_generator(env: resettable_env.ResettableEnv) -> BatchCallable:
  """Randomly samples state and action and computes next state from dynamics.

  This is one of the weakest possible priors, with broad support. It is similar
  to `rollout_generator` with a random policy, with two key differences.
  First, adjacent timesteps are independent from each other, as a state
  is randomly sampled at the start of each transition. Second, the initial
  state distribution is ignored. WARNING: This can produce physically impossible
  states, if there is no path from a feasible initial state to a sampled state.

  Args:
    env: A model-based environment.

  Returns:
    A function that, when called with timesteps and batch size, will perform
    the sampling process described above.
  """
  def f(total_timesteps: int, batch_size: int) -> Iterator[rewards.Batch]:
    """Helper function."""
    nbatch = math.ceil(total_timesteps / batch_size)
    for _ in range(nbatch):
      old_obses = []
      acts = []
      new_obses = []
      for _ in range(batch_size):
        old_state = env.state_space.sample()
        old_obs = env.obs_from_state(old_state)
        act = env.action_space.sample()
        new_state = env.transition(old_state, act)  # may be non-deterministic
        new_obs = env.obs_from_state(new_state)

        old_obses.append(old_obs)
        acts.append(act)
        new_obses.append(new_obs)
      yield rewards.Batch(old_obs=np.array(old_obses),
                          actions=np.array(acts),
                          new_obs=np.array(new_obses))
  return f


def make_pm(env_name="evaluating_rewards/PointMassLine-v0"):
  """Make Point Mass environment and dataset generator."""
  env = gym.make(env_name)
  obs_space = env.observation_space
  act_space = env.action_space

  pm = point_mass.PointMassPolicy(env)
  dataset_generator = rollout_generator(env, pm)

  return {
      "observation_space": obs_space,
      "action_space": act_space,
      "dataset_generator": dataset_generator,
  }
