# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
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
from unittest import mock

from imitation.data import types
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray

from evaluating_rewards import datasets
from evaluating_rewards.envs import point_mass
from evaluating_rewards.rewards import base


def no_op(*args, **kwargs):
    """No-operation: I do nothing, really."""
    del args, kwargs


def plot_reward(rew: xarray.DataArray, cmap: str = "RdBu", **kwargs) -> plt.Figure:
    """Visualizes a 3D reward for evaluating_rewards/PointMassLine-v0.

    Arguments:
        rew: A 3D-tensor containing reward values from a given state (2D) and
                action (1D).
        cmap: The color palette to use.
        **kwargs: Passed through to `rew.plot`.

    Returns:
        A figure containing a facet of heatmaps visualizing the reward.
    """
    # xarray insists on calling tight_layout(). We would prefer to tweak figure spacing
    # manually to ensure font sizes remain the same for publication-quality figures.
    with mock.patch("matplotlib.figure.Figure.tight_layout", new=no_op):
        rew = rew.rename({dim: dim.capitalize() for dim in rew.dims})
        rew = rew.rename(
            {"Acceleration": "Accel.", "Position": "Pos."}  # abbreviate to save space in figure
        )
        # By default xarray ignores figsize and does its own size calculation. Override.
        figsize = mpl.rcParams.get("figure.figsize")
        facet = rew.plot(x="Accel.", y="Velocity", col="Pos.", cmap=cmap, figsize=figsize, **kwargs)

        if "row" in kwargs:
            # xarray adds row labels in a hard-to-spot far-right side.
            # Remove them and put it in a better place.

            # Remove annotations on right-side axes (should just be labels)
            for ax in facet.axes[:, -1]:
                for child in ax.get_children():
                    if isinstance(child, mpl.text.Annotation):
                        child.remove()

            # Add more noticeable annotations on left-side axes
            row_dim = kwargs["row"]
            labels = [str(coord.values) for coord in rew.coords[row_dim]]
            pad = 30
            for ax, label in zip(facet.axes[:, 0], labels):
                ax.annotate(
                    label,
                    xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label,
                    textcoords="offset points",
                    size="large",
                    fontweight="bold",
                    ha="center",
                    va="baseline",
                )

        col_dim = "Pos."
        labels = [float(coord.values) for coord in rew.coords[col_dim]]
        for ax, label in zip(facet.axes[0, :], labels):
            ax.set_title(f"{col_dim} = {label:.4}", fontweight="bold")

    return facet.fig


def mesh_input(
    env: point_mass.PointMassEnv,
    goal: np.ndarray,
    pos_lim: float = 1.0,
    pos_density: int = 9,
    vel_lim: float = 1.0,
    act_lim: float = 1.0,
    density: int = 21,
) -> Tuple[List[List[int]], types.Transitions]:
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
    idxs = [np.linspace(-lim, lim, density) for lim, density in ranges]
    idxs = list(itertools.chain(*[[idx for _ in range(n)] for idx in idxs]))
    mesh = np.meshgrid(*idxs, indexing="ij")

    pos = np.stack([x.flatten() for x in mesh[0:n]], axis=-1)
    vel = np.stack([x.flatten() for x in mesh[n : 2 * n]], axis=-1)
    goal_obs = np.broadcast_to(goal, (pos.shape[0], n))
    obs = np.concatenate((pos, vel, goal_obs), axis=-1).astype(env.observation_space.dtype)
    actions = np.stack([x.flatten() for x in mesh[2 * n : 3 * n]], axis=-1)

    states = env.state_from_obs(obs)
    next_states = env.transition(states, actions)
    next_obs = env.obs_from_state(next_states)

    dones = np.zeros(len(obs), dtype=np.bool)
    dataset = types.Transitions(obs=obs, acts=actions, next_obs=next_obs, dones=dones, infos=None)
    return idxs, dataset


def evaluate_reward_model(
    env: point_mass.PointMassEnv, model: base.RewardModel, goal: np.ndarray, **kwargs
) -> xarray.DataArray:
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
    reward = base.evaluate_models({"m": model}, dataset)["m"]
    reward = reward.reshape(*[len(idx) for idx in idxs])
    reward = xarray.DataArray(reward, coords=idxs, dims=["position", "velocity", "acceleration"])
    return reward


def plot_state_density(
    dataset_generator: datasets.TransitionsCallable, nsamples: int = 2 ** 12, **kwargs
):
    """Plots the density of a state distribution.

    Arguments:
        dataset_generator: A generator implicitly defining the distribution.
        nsamples: The number of points to sample.
        **kwargs: Passed through to `sns.jointplot`.
    """
    batch = dataset_generator(nsamples)
    obs = batch.obs
    sns.jointplot(y=obs[:, 0], x=obs[:, 1], kind="hex", xlim=(-1, 1), ylim=(-1, 1), **kwargs)
    plt.xlabel("Velocity")
    plt.ylabel("Position")
