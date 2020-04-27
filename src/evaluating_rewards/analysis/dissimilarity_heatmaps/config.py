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
from typing import Iterable, Optional, Tuple

import sacred

from evaluating_rewards import serialize
from evaluating_rewards.analysis.dissimilarity_heatmaps import heatmaps, reward_masks

RewardCfg = Tuple[str, Optional[str]]  # (type, path)

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
    return [(kind, None) for kind in kinds]


def make_config(
    experiment: sacred.Experiment,
):  # pylint: disable=unused-variable,too-many-statements
    """Adds configs and named configs to `experiment`.

    The standard config parameters it defines are:
        - env_name (str): The environment name in the Gym registry of the rewards to compare.
        - x_reward_cfgs (Iterable[RewardCfg]): tuples of reward_type and reward_path for x-axis.
        - y_reward_cfgs (Iterable[RewardCfg]): tuples of reward_type and reward_path for y-axis.
        - log_root (str): the root directory to log; subdirectory path automatically constructed.
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
    def half_cheetah():
        """Heatmaps for HalfCheetah-v3."""
        env_name = "evaluating_rewards/HalfCheetah-v3"
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
        env_name = "evaluating_rewards/Hopper-v3"
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
