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

import logging
from typing import Iterable

import sacred

from evaluating_rewards.analysis.dissimilarity_heatmaps import heatmaps, reward_masks
from evaluating_rewards.scripts.distance import common

logger = logging.getLogger("evaluating_rewards.analysis.dissimilarity_heatmaps.cli_common")


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


def make_config(
    experiment: sacred.Experiment,
):  # pylint: disable=unused-variable,too-many-statements
    """Adds configs and named configs to `experiment`.

    The standard config parameters it defines are:
        - vals_path (Optional[str]): path to precomputed values to plot.
        - x_reward_cfgs (Iterable[RewardCfg]): tuples of reward_type and reward_path for x-axis.
        - y_reward_cfgs (Iterable[RewardCfg]): tuples of reward_type and reward_path for y-axis.
        - log_root (str): the root directory to log; subdirectory path automatically constructed.
        - heatmap_kwargs (dict): passed through to `analysis.compact_heatmaps`.
        - styles (Iterable[str]): styles to apply from `evaluating_rewards.analysis.stylesheets`.
        - save_kwargs (dict): passed through to `analysis.save_figs`.
    """

    @experiment.config
    def figure_config():
        """Defaults for figure parameters."""
        heatmap_kwargs = {
            "masks": {"all": [reward_masks.always_true]},
            "after_plot": heatmaps.horizontal_ticks,
        }
        styles = ["paper", "heatmap", "heatmap-3col", "tex"]
        styles_for_env = []
        save_kwargs = {
            "fmt": "pdf",
        }
        _ = locals()
        del _

    @experiment.named_config
    def large():
        """Large output size, high precision."""
        styles = ["paper", "heatmap", "heatmap-1col", "tex"]
        heatmap_kwargs = {
            "fmt": heatmaps.short_e,
            "cbar_kws": dict(fraction=0.05),
        }
        _ = locals()
        del _

    @experiment.named_config
    def point_mass():
        """Heatmaps for evaluating_rewards/PointMass* environments."""
        locals().update(**common.COMMON_CONFIGS["point_mass"])
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
        locals().update(**common.COMMON_CONFIGS["point_maze"])
        heatmap_kwargs = {
            "masks": {"all": [reward_masks.always_true]},  # "all" is still only 2x2
        }
        _ = locals()
        del _

    @experiment.named_config
    def point_maze_learned():
        """Compare rewards learned in PointMaze to the ground-truth reward."""
        # Analyzes models generated by `runners/transfer_point_maze.sh`.
        locals().update(**common.COMMON_CONFIGS["point_maze_learned"])

    @experiment.named_config
    def half_cheetah():
        """Heatmaps for seals/HalfCheetah-v0."""
        locals().update(**common.COMMON_CONFIGS["half_cheetah"])
        heatmap_kwargs = {
            "masks": {
                "diagonal": [reward_masks.zero, reward_masks.same],
                "control": [reward_masks.zero, reward_masks.control],
                "direction": [reward_masks.zero, reward_masks.direction],
                "no_ctrl": [reward_masks.zero, reward_masks.no_ctrl],
                "all": [reward_masks.always_true],
            },
        }
        styles_for_env = ["small-labels"]  # downscale emoji labels slightly
        _ = locals()
        del _

    @experiment.named_config
    def hopper():
        """Heatmaps for seals/Hopper-v0."""
        locals().update(**common.COMMON_CONFIGS["hopper"])
        heatmap_kwargs = {}
        heatmap_kwargs["masks"] = {
            "diagonal": [reward_masks.zero, reward_masks.same],
            "control": [reward_masks.zero, reward_masks.control],
            "direction": [reward_masks.zero, reward_masks.direction],
            "no_ctrl": [reward_masks.zero, reward_masks.no_ctrl],
            "different_activity": [reward_masks.zero, _hopper_activity],
            "all": [reward_masks.always_true],
        }
        styles_for_env = ["tiny-font"]
        _ = locals()
        del _
