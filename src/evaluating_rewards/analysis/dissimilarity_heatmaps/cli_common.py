# Copyright 2020 Adam Gleave
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

"""Configurations for dissimilarity_heatmaps heatmaps.

Shared between `evaluating_rewards.analysis.{plot_epic_heatmap,plot_canon_heatmap}`.
"""

import functools
import itertools
import os
from typing import Iterable, Mapping, Tuple

import gym
import numpy as np
import pandas as pd
import sacred
from stable_baselines.common import vec_env

from evaluating_rewards import rewards, serialize, tabular
from evaluating_rewards.analysis.dissimilarity_heatmaps import heatmaps, reward_masks

RewardCfg = Tuple[str, str]  # (type, path)


def canonicalize_reward_cfg(reward_cfg: Iterable[RewardCfg], data_root: str) -> Iterable[RewardCfg]:
    """Canonicalize paths and cast configs to tuples.

    Sacred has a bad habit of converting tuples in the config into lists, which makes
    the rewards no longer `RewardCfg` -- we put this right by casting to tuples.

    We also join paths with the `data_root`, unless they are the special "dummy" path.

    Args:
        reward_cfg: Iterable of configurations to canonicailze.
        data_root: The root to join paths to.

    Returns:
        Iterable of canonicalized configurations.
    """
    res = []
    for kind, path in reward_cfg:
        if path != "dummy":
            path = os.path.join(data_root, path)
        res.append((kind, path))
    return res


def load_models(
    env_name: str, reward_cfgs: Iterable[RewardCfg], discount: float,
) -> Mapping[RewardCfg, rewards.RewardModel]:
    """Load models specified by the `reward_cfgs`.

    Args:
        - env_name: The environment name in the Gym registry of the rewards to compare.
        - reward_cfgs: Iterable of reward configurations.
        - discount: Discount to use for reward models (mostly for shaping).

    Returns:
         A mapping from reward configurations to the loaded reward model.
    """
    venv = vec_env.DummyVecEnv([lambda: gym.make(env_name)])
    return {
        (kind, path): serialize.load_reward(kind, path, venv, discount)
        for kind, path in reward_cfgs
    }


def dissimilarity_mapping_to_series(
    dissimilarity: Mapping[Tuple[RewardCfg, RewardCfg], float]
) -> pd.Series:
    """Converts dissimilarity mapping to a MultiIndex series.

    Args:
        dissimilarity: A mapping from pairs of configurations to a float.

    Returns:
        A Series with a multi-index based on the configurations.
    """
    dissimilarity = {
        (xtype, xpath, ytype, ypath): v
        for ((xtype, xpath), (ytype, ypath)), v in dissimilarity.items()
    }
    dissimilarity = pd.Series(dissimilarity)
    dissimilarity.index.names = [
        "target_reward_type",
        "target_reward_path",
        "source_reward_type",
        "source_reward_path",
    ]
    return dissimilarity


def bootstrap_ci(vals: Iterable[float], n_bootstrap: int, alpha: float) -> Mapping[str, float]:
    bootstrapped = tabular.bootstrap(np.array(vals), stat_fn=np.mean, n_samples=n_bootstrap)
    lower, middle, upper = tabular.empirical_ci(bootstrapped, alpha)
    return {"lower": lower, "middle": middle, "upper": upper}


MUJOCO_STANDARD_ORDER = [
    "ForwardNoCtrl",
    "ForwardWithCtrl",
    "BackwardNoCtrl",
    "BackwardWithCtrl",
]


POINT_MASS_KINDS = [
    f"evaluating_rewards/PointMass{label}-v0"
    for label in ["SparseNoCtrl", "SparseWithCtrl", "DenseNoCtrl", "DenseWithCtrl", "GroundTruth"]
]


def _norm(args: Iterable[str]) -> bool:
    return any(reward_masks.match("evaluating_rewards/PointMassGroundTruth-v0")(args))


def _hopper_activity(args: Iterable[str]) -> bool:
    pattern = r"evaluating_rewards/(.*)(GroundTruth|Backflip)(.*)"
    repl = reward_masks.replace(pattern, r"\1\2")(args)
    return len(set(repl)) > 1 and reward_masks.no_ctrl(args)


def _hardcoded_model_cfg(kinds: Iterable[str]) -> Iterable[RewardCfg]:
    return [(kind, "dummy") for kind in kinds]


def make_config(
    experiment: sacred.Experiment,
):  # pylint: disable=unused-variable,too-many-statements
    """Adds configs and named configs to `experiment`.

    The standard config parameters it defines are:
        - env_name (str): The environment name in the Gym registry of the rewards to compare.
        - x_reward_cfgs (Iterable[RewardCfg]): tuples of reward_type and reward_path for x-axis.
        - y_reward_cfgs (Iterable[RewardCfg]): tuples of reward_type and reward_path for y-axis.
        - log_root (str): the root directory to log; subdirectory path automatically constructed.
        - n_bootstrap (int): the number of bootstrap samples to take.
        - heatmap_kwargs (dict): passed through to `analysis.compact_heatmaps`.
        - styles (Iterable[str]): styles to apply from `evaluating_rewards.analysis.stylesheets`.
        - save_kwargs (dict): passed through to `analysis.save_figs`.
    """

    @experiment.config
    def default_config():
        """Default configuration values."""
        env_name = "evaluating_rewards/PointMassLine-v0"
        kinds = POINT_MASS_KINDS
        log_root = serialize.get_output_dir()  # where results are read from/written to
        n_bootstrap = 1000  # number of bootstrap samples
        alpha = 95  # percentile confidence interval

        # Reward configurations: models to compare
        x_reward_cfgs = None
        y_reward_cfgs = None

        _ = locals()
        del _

    @experiment.config
    def reward_config(kinds, x_reward_cfgs, y_reward_cfgs):
        """Default reward configuration: hardcoded reward model types from kinds."""
        if kinds is not None:
            if x_reward_cfgs is None:
                x_reward_cfgs = _hardcoded_model_cfg(kinds)
            if y_reward_cfgs is None:
                y_reward_cfgs = _hardcoded_model_cfg(kinds)
        _ = locals()
        del _

    @experiment.config
    def figure_config(kinds):
        """Defaults for figure parameters."""
        heatmap_kwargs = {
            "masks": {"all": [reward_masks.always_true]},
            "after_plot": heatmaps.horizontal_ticks,
        }
        if kinds and "order" not in heatmap_kwargs:
            heatmap_kwargs["order"] = kinds
        styles = ["paper", "heatmap", "heatmap-1col", "tex"]
        save_kwargs = {
            "fmt": "pdf",
        }
        _ = locals()
        del _

    @experiment.named_config
    def large():
        """Large output size, high precision."""
        styles = ["paper", "heatmap", "heatmap-2col", "tex"]
        heatmap_kwargs = {
            "fmt": heatmaps.short_e,
            "cbar_kws": dict(fraction=0.05),
        }
        _ = locals()
        del _

    @experiment.named_config
    def point_mass():
        """Heatmaps for evaluating_rewards/PointMass* environments."""
        env_name = "evaluating_rewards/PointMassLine-v0"
        kinds = POINT_MASS_KINDS
        heatmap_kwargs = {}
        heatmap_kwargs["masks"] = {
            "diagonal": [reward_masks.zero, reward_masks.same],
            "control": [reward_masks.zero, reward_masks.control],
            "dense_vs_sparse": [reward_masks.zero, reward_masks.sparse_or_dense],
            "norm": [reward_masks.zero, reward_masks.same, _norm],
            "all": [reward_masks.always_true],
        }
        _ = locals()
        del _

    @experiment.named_config
    def point_maze():
        """Heatmaps for imitation/PointMaze{Left,Right}-v0 environments."""
        env_name = "imitation/PointMazeLeft-v0"
        kinds = [
            "imitation/PointMazeGroundTruthWithCtrl-v0",
            "imitation/PointMazeGroundTruthNoCtrl-v0",
        ]
        heatmap_kwargs = {
            "masks": {"all": [reward_masks.always_true]},  # "all" is still only 2x2
        }
        _ = locals()
        del _

    @experiment.named_config
    def point_maze_learned():
        """Compare rewards learned in PointMaze to the ground-truth reward."""
        env_name = "imitation/PointMazeLeftVel-v0"
        x_reward_cfgs = [
            ("evaluating_rewards/PointMazeGroundTruthWithCtrl-v0", "dummy"),
            ("evaluating_rewards/PointMazeGroundTruthNoCtrl-v0", "dummy"),
        ]
        y_reward_cfgs = [
            (
                "imitation/RewardNet_unshaped-v0",
                "transfer_point_maze/reward/irl_state_only/checkpoints/final/discrim/reward_net/",
            ),
            (
                "imitation/RewardNet_unshaped-v0",
                "transfer_point_maze/reward/irl_state_action/checkpoints/final/discrim/reward_net/",
            ),
            ("evaluating_rewards/RewardModel-v0", "transfer_point_maze/reward/preferences/model/"),
            ("evaluating_rewards/RewardModel-v0", "transfer_point_maze/reward/regress/model/"),
        ]
        kinds = None
        _ = locals()
        del _

    @experiment.named_config
    def half_cheetah():
        """Heatmaps for HalfCheetah-v3."""
        env_name = "seals/HalfCheetah-v0"
        kinds = [
            f"evaluating_rewards/HalfCheetahGroundTruth{suffix}-v0"
            for suffix in MUJOCO_STANDARD_ORDER
        ]
        heatmap_kwargs = {
            "masks": {
                "diagonal": [reward_masks.zero, reward_masks.same],
                "control": [reward_masks.zero, reward_masks.control],
                "direction": [reward_masks.zero, reward_masks.direction],
                "no_ctrl": [reward_masks.zero, reward_masks.no_ctrl],
                "all": [reward_masks.always_true],
            },
        }
        _ = locals()
        del _

    @experiment.named_config
    def hopper():
        """Heatmaps for Hopper-v3."""
        env_name = "seals/Hopper-v0"
        activities = ["GroundTruth", "Backflip"]
        kinds = [
            f"evaluating_rewards/Hopper{prefix}{suffix}-v0"
            for prefix, suffix in itertools.product(activities, MUJOCO_STANDARD_ORDER)
        ]
        del activities
        heatmap_kwargs = {}
        heatmap_kwargs["masks"] = {
            "diagonal": [reward_masks.zero, reward_masks.same],
            "control": [reward_masks.zero, reward_masks.control],
            "direction": [reward_masks.zero, reward_masks.direction],
            "no_ctrl": [reward_masks.zero, reward_masks.no_ctrl],
            "different_activity": [reward_masks.zero, _hopper_activity],
            "all": [reward_masks.always_true],
        }
        heatmap_kwargs["fmt"] = functools.partial(heatmaps.short_e, precision=0)
        _ = locals()
        del _
