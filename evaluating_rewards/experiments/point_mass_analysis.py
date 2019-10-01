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

"""Helper methods for experiments involving evaluating_rewards.envs.point_mass.

See Colab notebooks for use cases.
"""

import itertools
from typing import List, Tuple

from evaluating_rewards import rewards
from evaluating_rewards.envs import point_mass
from evaluating_rewards.experiments import datasets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray


def plot_reward(rew: xarray.DataArray,
                cmap: str = "RdBu",
                **kwargs,
               ) -> plt.Figure:
  """Visualizes a 3D reward for evaluating_rewards/PointMassLine-v0.

  Arguments:
    rew: A 3D-tensor containing reward values from a given state (2D) and
        action (1D).
    cmap: The color palette to use.
    **kwargs: Passed through to `rew.plot`.

  Returns:
    A figure containing a facet of heatmaps visualizing the reward.
  """
  rew = rew.rename({dim: dim.capitalize() for dim in rew.dims})
  rew = rew.rename({"Position": "Pos"})  # tight on space in title
  facet = rew.plot(x="Acceleration", y="Velocity", col="Pos",
                   cmap=cmap, **kwargs)
  return facet.fig


def mesh_input(env: point_mass.PointMassEnv,
               goal: np.ndarray,
               pos_lim: float = 1.0,
               pos_density: int = 9,
               vel_lim: float = 1.0,
               act_lim: float = 1.0,
               density: int = 21,
              ) -> Tuple[List[List[int]], rewards.Batch]:
  """Computes a grid dataset of observation, actions and next observations.

  Specifically, it computes a grid of position, velocity and actions
  with the corresponding limits and density. It uses a fixed, specified goal.
  It then computes the next observation for each possible combination
  under the environment transition dynamics.

  Arguments:
    env: The PointMass environment.
    goal: A goal position, an (env.ndim)-dimensional vector.
    pos_lim: Position limit: the mesh will include range [-pos_lim, pos_lim].
    pos_density: The number of points in the position axis.
    vel_lim: Velocity limit: the mesh will include range [-vel_lim, vel_lim].
    act_lim: Action limit: the mesh will include range [-act_lim, act_lim].
    density: The number of points in the velocity and acceleration axes.

  Returns:
    Indexes (before turning into a grid) and a batch of resulting
    observation, action and next-observation triples.
  """
  n = env.ndim

  ranges = [(pos_lim, pos_density), (vel_lim, density), (act_lim, density)]
  idxs = [np.linspace(-lim, lim, density)
          for lim, density in ranges]
  idxs = list(itertools.chain(*[[idx for _ in range(n)] for idx in idxs]))
  mesh = np.meshgrid(*idxs, indexing="ij")

  pos = np.stack([x.flatten() for x in mesh[0:n]], axis=-1)
  vel = np.stack([x.flatten() for x in mesh[n:2*n]], axis=-1)
  goal_obs = np.broadcast_to(goal, (pos.shape[0], n))
  obs = np.concatenate((pos, vel, goal_obs), axis=-1)
  actions = np.stack([x.flatten() for x in mesh[2*n:3*n]], axis=-1)

  states = env.state_from_obs(obs)
  next_states = env.transition(states, actions)
  next_obs = env.obs_from_state(next_states)

  dataset = rewards.Batch(obs=obs,
                          actions=actions,
                          next_obs=next_obs)
  return idxs, dataset


def evaluate_reward_model(env: point_mass.PointMassEnv,
                          model: rewards.RewardModel,
                          goal: np.ndarray,
                          **kwargs) -> xarray.DataArray:
  """Computes the reward predicted by model on environment.

  Arguments:
    env: A point mass environment.
    model: A reward model.
    goal: The position of the goal in env.
    **kwargs: Passed through to `mesh_input`.

  Returns:
    A 3D-tensor of rewards, indexed by position, velocity and acceleration.
  """
  assert model.observation_space == env.observation_space
  assert model.action_space == env.action_space
  idxs, dataset = mesh_input(env, goal=goal, **kwargs)
  reward = rewards.evaluate_models({"m": model}, dataset)["m"]
  reward = reward.reshape(*[len(idx) for idx in idxs])
  reward = xarray.DataArray(reward, coords=idxs,
                            dims=["position", "velocity", "acceleration"])
  return reward


def plot_state_density(dataset_generator: datasets.BatchCallable,
                       nsamples: int = 2 ** 12,
                       **kwargs):
  """Plots the density of a state distribution.

  Arguments:
    dataset_generator: A generator implicitly defining the distribution.
    nsamples: The number of points to sample.
    **kwargs: Passed through to `sns.jointplot`.
  """
  batch = next(dataset_generator(nsamples, nsamples))
  obs, _, _ = batch
  sns.jointplot(y=obs[:, 0], x=obs[:, 1], kind="hex",
                xlim=(-1, 1), ylim=(-1, 1), **kwargs)
  plt.xlabel("Velocity")
  plt.ylabel("Position")
