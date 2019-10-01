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
import math
from typing import Any, Dict, Optional, List, Tuple

from evaluating_rewards import rewards
from evaluating_rewards.envs import point_mass
from evaluating_rewards.experiments import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray


def closest_elt(xs, target):
  return min(xs, key=lambda x: abs(x - target))


def reward_heatmap(rew: pd.DataFrame,
                   goal: np.ndarray,
                   ax=None,
                   **kwargs):
  """Plots a heatmap of a 2D reward for evaluating_rewards/PointMassLine-v0.

  This could be used for visualizing a state-only reward function, or for
  plotting a 2D-slice of a state-action reward function.

  Arguments:
    rew: The 2D reward.
    goal: A 1D-vector describing the goal position on the line.
    ax: A matplotlib axis to plot on.
    **kwargs: Passed through to `sns.heatmap`.

  Returns:
    The axis plotted on.
  """
  if ax is None:
    ax = plt.gca()

  assert goal.ndim == 1

  rounded_goal = closest_elt(rew.index, goal)
  annot = pd.DataFrame("", index=rew.index, columns=rew.index)
  annot.loc[rounded_goal, :] = "G"

  xticklabels = [f"{x:.2f}" for x in rew.columns]
  yticklabels = [f"{y:.2f}" for y in rew.index]

  return sns.heatmap(rew, annot=annot, fmt="",
                     xticklabels=xticklabels, yticklabels=yticklabels,
                     ax=ax, **kwargs)


def plot_reward(rew: xarray.DataArray,
                goal: np.ndarray,
                zaxis: str,
                ncols: int = 4,
                subplot_kwargs: Optional[Dict[str, Any]] = None,
                **kwargs,
               ) -> plt.Figure:
  """Visualizes a 3D reward for evaluating_rewards/PointMassLine-v0.

  Repeatedly calls plot_rewards_2d on slices of `rew` with a fixed value
  of `zaxis`, tieing them together into a single plot containing many heatmaps.

  Arguments:
    rew: A 3D-tensor containing reward values from a given state (2D) and
        action (1D).
    goal: A 1D-vector describing the goal position on the line.
    zaxis: One of "acceleration", "velocity" or "position".
    ncols: The number of heatmaps in each row.
    subplot_kwargs: Passed through to `plt.subplots`.
    **kwargs: Passed through to `reward_heatmap`.

  Returns:
    A figure containing many heatmaps visualizing the reward.
  """
  coords = rew.coords[zaxis]
  nrows = math.ceil(len(coords) / ncols)
  our_subplot_kwargs = dict(figsize=(20, 8), gridspec_kw={"hspace": 0.5})
  if subplot_kwargs is not None:
    our_subplot_kwargs.update(subplot_kwargs)
  fig, axs = plt.subplots(nrows, ncols, squeeze=False, **our_subplot_kwargs)
  axs_flatten = list(itertools.chain(*axs))

  vmin = rew.min()
  vmax = rew.max()
  for z, ax in zip(coords, axs_flatten):
    df = rew.loc[{zaxis: z}].to_pandas()
    ax.set_title(f"{zaxis} = {float(z):.3}")
    reward_heatmap(df, goal, vmin=vmin, vmax=vmax, ax=ax, **kwargs)

  return fig


def mesh_input(env: point_mass.PointMassEnv,
               goal: np.ndarray,
               pos_lim: float = 1.0,
               vel_lim: float = 1.0,
               act_lim: float = 1.0,
               density: float = 21,
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
    vel_lim: Velocity limit: the mesh will include range [-vel_lim, vel_lim].
    act_lim: Action limit: the mesh will include range [-act_lim, act_lim].
    density: The number of points to include in each range. The total number
        of points will be `density**(env.ndim*3)`.

  Returns:
    Indexes (before turning into a grid) and a batch of resulting
    observation, action and next-observation triples.
  """
  n = env.ndim

  idxs = [np.linspace(-lim, lim, density)
          for lim in [pos_lim, vel_lim, act_lim]]
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
