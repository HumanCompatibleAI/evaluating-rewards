"""Configurations for dissimilarity heatmaps.

Shared between `evaluating_rewards.analysis.{plot_epic_heatmap,plot_canon_heatmap}`.
"""

# TODO(adam): should this also be integrated into plot_gridworld_divergence?

import functools
import itertools
from typing import Iterable

import sacred

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


# pylint:disable=unused-variable


def make_config(experiment: sacred.Experiment):
    """Adds configs and named configs to `experiment`.

    Assumes Sacred experiment functions use `heatmap_kwargs`, `styles` and `save_kwargs`
    like `plot_epic_heatmap` and `plot_canon_heatmap`.
    """

    @experiment.config
    def default_config():
        """Default configuration values."""
        # Figure parameters
        heatmap_kwargs = {
            "masks": {"all": [visualize.always_true]},
            "after_plot": horizontal_ticks,
        }
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
        heatmap_kwargs = {}
        heatmap_kwargs["masks"] = {
            "diagonal": [visualize.zero, visualize.same],
            "control": [visualize.zero, visualize.control],
            "dense_vs_sparse": [visualize.zero, visualize.sparse_or_dense],
            "norm": [visualize.zero, visualize.same, _norm],
            "all": [visualize.always_true],
        }
        order = ["SparseNoCtrl", "SparseWithCtrl", "DenseNoCtrl", "DenseWithCtrl", "GroundTruth"]
        heatmap_kwargs["order"] = [f"evaluating_rewards/PointMass{label}-v0" for label in order]
        del order
        heatmap_kwargs["after_plot"] = horizontal_ticks
        _ = locals()
        del _

    @experiment.named_config
    def point_maze():
        """Heatmaps for imitation/PointMaze{Left,Right}-v0 environments."""
        env_name = "evaluating_rewards/PointMazeLeft-v0"
        heatmap_kwargs = {
            "masks": {"all": [visualize.always_true]},  # "all" is still only 3x3
            "order": [
                "imitation/PointMazeGroundTruthWithCtrl-v0",
                "imitation/PointMazeGroundTruthNoCtrl-v0",
                "evaluating_rewards/Zero-v0",
            ],
        }
        _ = locals()
        del _

    @experiment.named_config
    def half_cheetah():
        """Heatmaps for HalfCheetah-v3."""
        env_name = "evaluating_rewards/HalfCheetah-v3"
        heatmap_kwargs = {
            "masks": {
                "diagonal": [visualize.zero, visualize.same],
                "control": [visualize.zero, visualize.control],
                "direction": [visualize.zero, visualize.direction],
                "no_ctrl": [visualize.zero, visualize.no_ctrl],
                "all": [visualize.always_true],
            },
            "order": [
                f"evaluating_rewards/HalfCheetahGroundTruth{suffix}-v0"
                for suffix in MUJOCO_STANDARD_ORDER
            ],
        }
        _ = locals()
        del _

    @experiment.named_config
    def hopper():
        """Heatmaps for Hopper-v3."""
        env_name = "evaluating_rewards/Hopper-v3"
        heatmap_kwargs = {}
        heatmap_kwargs["masks"] = {
            "diagonal": [visualize.zero, visualize.same],
            "control": [visualize.zero, visualize.control],
            "direction": [visualize.zero, visualize.direction],
            "no_ctrl": [visualize.zero, visualize.no_ctrl],
            "different_activity": [visualize.zero, _hopper_activity],
            "all": [visualize.always_true],
        }
        activities = ["GroundTruth", "Backflip"]
        heatmap_kwargs["order"] = [
            f"evaluating_rewards/Hopper{prefix}{suffix}-v0"
            for prefix, suffix in itertools.product(activities, MUJOCO_STANDARD_ORDER)
        ]
        heatmap_kwargs["after_plot"] = horizontal_ticks
        heatmap_kwargs["fmt"] = functools.partial(visualize.short_e, precision=0)
        del activities
        _ = locals()
        del _


# pylint:enable=unused-variable
