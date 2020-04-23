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

"""Configurations for dissimilarity heatmaps.

Shared between `evaluating_rewards.analysis.{plot_epic_heatmap,plot_canon_heatmap}`.
"""

# TODO(adam): should this also be integrated into plot_gridworld_divergence?

import functools
import itertools
from typing import Iterable, Tuple

import sacred

from evaluating_rewards import serialize
from evaluating_rewards.analysis import visualize
from evaluating_rewards.analysis.visualize import horizontal_ticks

MUJOCO_STANDARD_ORDER = [
    "ForwardNoCtrl",
    "ForwardWithCtrl",
    "BackwardNoCtrl",
    "BackwardWithCtrl",
]


def _norm(args: Iterable[str]) -> bool:
    return any(visualize.match("evaluating_rewards/PointMassGroundTruth-v0")(args))


def _hopper_activity(args: Iterable[str]) -> bool:
    pattern = r"evaluating_rewards/(.*)(GroundTruth|Backflip)(.*)"
    repl = visualize.replace(pattern, r"\1\2")(args)
    return len(set(repl)) > 1 and visualize.no_ctrl(args)


def _hardcoded_model_cfg(kinds: Iterable[str]) -> Iterable[Tuple[str, str]]:
    return [(kind, "dummy") for kind in kinds]


def make_config(
    experiment: sacred.Experiment,
):  # pylint: disable=unused-variable,too-many-statements
    """Adds configs and named configs to `experiment`.

    Assumes Sacred experiment functions use `heatmap_kwargs`, `styles` and `save_kwargs`
    like `plot_epic_heatmap` and `plot_canon_heatmap`.
    """

    @experiment.config
    def default_config():
        """Default configuration values."""
        env_name = "evaluating_rewards/Hopper-v3"
        log_root = serialize.get_output_dir()  # where results are read from/written to

        # Reward configurations: models to compare
        kinds = None
        x_reward_cfgs = None
        y_reward_cfgs = None

        _ = locals()
        del _

    @experiment.config
    def reward_config(kinds, x_reward_cfgs, y_reward_cfgs):
        """Default to hardcoded reward model types."""
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
            "masks": {"all": [visualize.always_true]},
            "after_plot": horizontal_ticks,
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
            "fmt": visualize.short_e,
            "cbar_kws": dict(fraction=0.05),
        }
        _ = locals()
        del _

    @experiment.named_config
    def point_mass():
        """Heatmaps for evaluating_rewards/PointMass* environments."""
        env_name = "evaluating_rewards/PointMassLine-v0"
        kinds = ["SparseNoCtrl", "SparseWithCtrl", "DenseNoCtrl", "DenseWithCtrl", "GroundTruth"]
        kinds = [f"evaluating_rewards/PointMass{label}-v0" for label in kinds]
        heatmap_kwargs = {}
        heatmap_kwargs["masks"] = {
            "diagonal": [visualize.zero, visualize.same],
            "control": [visualize.zero, visualize.control],
            "dense_vs_sparse": [visualize.zero, visualize.sparse_or_dense],
            "norm": [visualize.zero, visualize.same, _norm],
            "all": [visualize.always_true],
        }
        heatmap_kwargs["after_plot"] = horizontal_ticks
        _ = locals()
        del _

    @experiment.named_config
    def point_maze():
        """Heatmaps for imitation/PointMaze{Left,Right}-v0 environments."""
        env_name = "evaluating_rewards/PointMazeLeft-v0"
        kinds = [
            "imitation/PointMazeGroundTruthWithCtrl-v0",
            "imitation/PointMazeGroundTruthNoCtrl-v0",
            "evaluating_rewards/Zero-v0",
        ]
        heatmap_kwargs = {
            "masks": {"all": [visualize.always_true]},  # "all" is still only 3x3
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
                "diagonal": [visualize.zero, visualize.same],
                "control": [visualize.zero, visualize.control],
                "direction": [visualize.zero, visualize.direction],
                "no_ctrl": [visualize.zero, visualize.no_ctrl],
                "all": [visualize.always_true],
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
            "diagonal": [visualize.zero, visualize.same],
            "control": [visualize.zero, visualize.control],
            "direction": [visualize.zero, visualize.direction],
            "no_ctrl": [visualize.zero, visualize.no_ctrl],
            "different_activity": [visualize.zero, _hopper_activity],
            "all": [visualize.always_true],
        }
        heatmap_kwargs["after_plot"] = horizontal_ticks
        heatmap_kwargs["fmt"] = functools.partial(visualize.short_e, precision=0)
        _ = locals()
        del _
