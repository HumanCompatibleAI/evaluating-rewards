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

"""CLI script to visualize a reward model for PointMassLine*-v0.

Note most environments are too high-dimensional to directly visualize reward.
"""

import os
from typing import Any, Mapping, Sequence, Tuple

import gym
from imitation import util
import numpy as np
import sacred
from stable_baselines.common import vec_env
import xarray as xr

from evaluating_rewards import serialize
from evaluating_rewards.experiments import point_mass_analysis, visualize
from evaluating_rewards.scripts import script_utils

visualize_pm_reward_ex = sacred.Experiment("visualize_pm_reward")


@visualize_pm_reward_ex.config
def default_config():
    """Default configuration values."""
    # Reward parameters
    env_name = "evaluating_rewards/PointMassLine-v0"
    # Simple method: specify one model
    reward_type = "evaluating_rewards/PointMassSparse-v0"
    reward_path = "dummy"
    # Complex method: specify multiple models
    models = None

    # Mesh parameters
    density = 20  # number of points along velocity and acceleration axis
    pos_density = 9  # number of points along position axis = number of plots
    lim = 1.0  # points span [-lim, lim]
    pos_lim = lim  # position point range
    vel_lim = lim  # velocity point range
    act_lim = lim  # action point range

    # Figure parameters
    ncols = 3  # number of heatmaps per row
    width = 5  # in
    height = 4  # in
    cbar_kwargs = {"fraction": 0.1, "pad": 0.05}
    fmt = "pdf"  # file type
    _ = locals()  # quieten flake8 unused variable warning
    del _


script_utils.add_logging_config(visualize_pm_reward_ex, "visualize_pm_reward")


@visualize_pm_reward_ex.config
def logging_config(log_root, models, reward_type, reward_path):
    if models is None:
        save_path = os.path.join(
            log_root, reward_type.replace("/", "_"), reward_path.replace("/", "_")
        )
    else:
        save_path = util.make_unique_timestamp()
    _ = locals()  # quieten flake8 unused variable warning
    del _


@visualize_pm_reward_ex.config
def reward_config(models, reward_type, reward_path):
    if models is None:
        models = [("singleton", reward_type, reward_path)]
    _ = locals()  # quieten flake8 unused variable warning
    del _


STRIP_CONFIG = dict(pos_density=7, ncols=7, width=9.5, height=1.5)


@visualize_pm_reward_ex.named_config
def strip():
    locals().update(**STRIP_CONFIG)
    cbar_kwargs = {"aspect": 3, "pad": 0.03}  # noqa: F841  pylint:disable=unused-variable


@visualize_pm_reward_ex.named_config
def dense_no_ctrl_sparsified():
    """PointMassDenseNoCtrl along with sparsified and ground-truth sparse reward."""
    locals().update(**STRIP_CONFIG)
    height = 4.5
    pos_lim = 0.15
    # Use lists of tuples rather than OrderedDict as Sacred reorders dictionaries
    models = [
        ("Dense", "evaluating_rewards/PointMassDenseNoCtrl-v0", "dummy"),
        (
            "Sparsified",
            "evaluating_rewards/RewardModel-v0",
            os.path.join(
                script_utils.get_output_dir(),
                "model_comparison",
                "evaluating_rewards_PointMassLine-v0",
                "20190921_190606_58935eb0a51849508381daf1055d0360",
                "model",
            ),
        ),
        ("Sparse", "evaluating_rewards/PointMassSparseNoCtrl-v0", "dummy"),
    ]
    _ = locals()  # quieten flake8 unused variable warning
    del _


@visualize_pm_reward_ex.named_config
def fast():
    """Small config, intended for tests / debugging."""
    density = 5  # noqa: F841  pylint:disable=unused-variable


@visualize_pm_reward_ex.main
def visualize_pm_reward(
    env_name: str,
    models: Sequence[Tuple[str, str, str]],
    # Mesh parameters
    pos_lim: float,
    pos_density: int,
    vel_lim: float,
    act_lim: float,
    density: int,
    # Figure parameters
    ncols: int,
    width: float,
    height: float,
    cbar_kwargs: Mapping[str, Any],
    save_path: str,
    fmt: str,
) -> xr.DataArray:
    """Entry-point into script to visualize a reward model for point mass."""
    env = gym.make(env_name)
    venv = vec_env.DummyVecEnv([lambda: env])
    goal = np.array([0.0])

    rewards = {}
    with util.make_session():
        for model_name, reward_type, reward_path in models:
            model = serialize.load_reward(reward_type, reward_path, venv)
            reward = point_mass_analysis.evaluate_reward_model(
                env,
                model,
                goal=goal,
                pos_lim=pos_lim,
                pos_density=pos_density,
                vel_lim=vel_lim,
                act_lim=act_lim,
                density=density,
            )
            rewards[model_name] = reward

    if len(rewards) == 1:
        reward = next(iter(rewards.values()))
        kwargs = {"col_wrap": ncols}
    else:
        reward = xr.Dataset(rewards).to_array("model")
        kwargs = {"row": "Model"}

    fig = point_mass_analysis.plot_reward(
        reward, figsize=(width, height), cbar_kwargs=cbar_kwargs, **kwargs
    )
    visualize.save_fig(save_path, fig, fmt=fmt)

    return reward


if __name__ == "__main__":
    script_utils.experiment_main(visualize_pm_reward_ex, "visualize_pm_reward")
