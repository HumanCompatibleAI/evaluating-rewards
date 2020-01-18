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

"""CLI script to plot heatmap of divergence between pairs of reward models."""

import itertools
import os
from typing import Any, Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import sacred

from evaluating_rewards.experiments import results, visualize
from evaluating_rewards.scripts import script_utils

visualize_divergence_heatmap_ex = sacred.Experiment("visualize_divergence_heatmap")


@visualize_divergence_heatmap_ex.config
def default_config():
    """Default configuration values."""
    # Dataset parameters
    log_root = script_utils.get_output_dir()  # where results are read from/written to
    data_root = os.path.join(log_root, "comparison")  # root of comparison data directory
    data_subdir = "hardcoded"  # optional, if omitted searches all data (slow)
    search = {  # parameters to filter by in datasets
        "env_name": "evaluating_rewards/Hopper-v3",
        "model_wrapper_kwargs": {},
    }

    # Figure parameters
    heatmap_kwargs = {
        "masks": {"all": [results.always_true]},
        "order": None,
    }
    # TODO: style sheet to apply?
    save_kwargs = {
        "fmt": "pdf",
    }

    _ = locals()
    del _


@visualize_divergence_heatmap_ex.config
def logging_config(log_root, search):
    # TODO: timestamp?
    log_dir = os.path.join(  # noqa: F841  pylint:disable=unused-variable
        log_root, "visualize_divergence_heatmap", str(search).replace("/", "_"),
    )


def _norm(args: Iterable[str]) -> bool:
    return any(results.match("evaluating_rewards/PointMassGroundTruth-v0")(args))


def _point_mass_after_plot():
    plt.subplots_adjust(bottom=0.15, top=0.95, left=0.16, right=0.95)


@visualize_divergence_heatmap_ex.named_config
def point_mass():
    """Heatmaps for evaluating_rewards/PointMass* environments."""
    search = {  # noqa: F841  pylint:disable=unused-variable
        "env_name": "evaluating_rewards/PointMassLine-v0",
        "dataset_factory": {
            # can also use evaluating_rewards.experiments.datasets.random_transition_generator
            "escape/py/function": "evaluating_rewards.experiments.datasets.random_policy_generator",
        },
    }
    heatmap_kwargs = {}
    heatmap_kwargs["masks"] = {
        "diagonal": [results.zero, results.same],
        "control": [results.zero, results.control],
        "dense_vs_sparse": [results.zero, results.sparse_or_dense],
        "norm": [results.zero, results.same, _norm],
        "all": [results.always_true],
    }
    order = ["SparseNoCtrl", "Sparse", "DenseNoCtrl", "Dense", "GroundTruth"]
    heatmap_kwargs["order"] = [f"evaluating_rewards/PointMass{label}-v0" for label in order]
    heatmap_kwargs["after_plot"] = _point_mass_after_plot
    del order


@visualize_divergence_heatmap_ex.named_config
def point_maze():
    """Heatmaps for imitation/PointMaze{Left,Right}-v0 environments."""
    search = {
        "env_name": "evaluating_rewards/PointMazeLeft-v0",
    }
    heatmap_kwargs = {
        "masks": {"all": [results.always_true]},  # "all" is still only 3x3
        "order": [
            "imitation/PointMazeGroundTruthWithCtrl-v0",
            "imitation/PointMazeGroundTruthNoCtrl-v0",
            "evaluating_rewards/Zero-v0",
        ],
    }
    _ = locals()
    del _


MUJOCO_STANDARD_ORDER = ["ForwardNoCtrl", "ForwardWithCtrl", "BackwardNoCtrl", "BackwardWithCtrl"]


@visualize_divergence_heatmap_ex.named_config
def half_cheetah():
    """Heatmaps for HalfCheetah-v3."""
    search = {
        "env_name": "evaluating_rewards/HalfCheetah-v3",
    }
    heatmap_kwargs = {
        "masks": {
            "diagonal": [results.zero, results.same],
            "control": [results.zero, results.control],
            "direction": [results.zero, results.direction],
            "no_ctrl": [results.zero, results.no_ctrl],
            "all": [results.always_true],
        },
        "order": [
            f"evaluating_rewards/HalfCheetahGroundTruth{suffix}-v0"
            for suffix in MUJOCO_STANDARD_ORDER
        ],
    }
    _ = locals()
    del _


def hopper_activity(args: Iterable[str]) -> bool:
    pattern = r"evaluating_rewards/(.*)(GroundTruth|Backflip)(.*)"
    repl = results.replace(pattern, r"\1\2")(args)
    return len(set(repl)) > 1 and results.no_ctrl(args)


@visualize_divergence_heatmap_ex.named_config
def hopper():
    """Heatmaps for Hopper-v3."""
    search = {  # noqa: F841  pylint:disable=unused-variable
        "env_name": "evaluating_rewards/Hopper-v3",
    }
    heatmap_kwargs = {}
    heatmap_kwargs["masks"] = {
        "diagonal": [results.zero, results.same],
        "control": [results.zero, results.control],
        "direction": [results.zero, results.direction],
        "no_ctrl": [results.zero, results.no_ctrl],
        "different_activity": [results.zero, hopper_activity],
        "all": [results.always_true],
    }
    activities = ["GroundTruth", "Backflip"]
    heatmap_kwargs["order"] = [
        f"evaluating_rewards/Hopper{prefix}{suffix}-v0"
        for prefix, suffix in itertools.product(activities, MUJOCO_STANDARD_ORDER)
    ]
    heatmap_kwargs["after_plot"] = lambda: plt.yticks(rotation="horizontal")
    del activities


@visualize_divergence_heatmap_ex.main
def visualize_divergence_heatmap(
    data_root: str,
    data_subdir: Optional[str],
    search: Mapping[str, Any],
    heatmap_kwargs: Mapping[str, Any],
    log_dir: str,
    save_kwargs: Mapping[str, Any],
):
    """Entry-point into script to produce divergence heatmaps.

    Args:
        data_root: where to load data from.
        data_kind: subdirectory to load data from.
        search: mapping which Sacred configs must match to be included in results.
        heatmap_kwargs: passed through to `visualize.compact_heatmaps`.
        log_dir: directory to write figures and other logging to.
        save_kwargs: passed through to `visualize.save_figs`.
        """
    data_dir = data_root
    if data_subdir is not None:
        data_dir = os.path.join(data_dir, data_subdir)
    # Workaround tags reserved by Sacred
    search = dict(search)
    for k, v in search.items():
        if isinstance(v, dict):
            search[k] = {inner_k.replace("escape/", ""): inner_v for inner_k, inner_v in v.items()}

    def cfg_filter(cfg):
        return all((cfg.get(k) == v for k, v in search.items()))

    keys = ["source_reward_type", "source_reward_path", "target_reward_type", "seed"]
    stats = results.load_multiple_stats(data_dir, keys, cfg_filter=cfg_filter)
    res = results.pipeline(stats)
    loss = res["loss"]["loss"]
    # TODO: usetex, Apple Color Emoji, make fonts match doc
    heatmap_kwargs = dict(heatmap_kwargs)
    if heatmap_kwargs.get("order") is None:
        heatmap_kwargs["order"] = loss.index.levels[0]

    heatmaps = results.compact_heatmaps(loss=loss, **heatmap_kwargs)
    visualize.save_fig(os.path.join(log_dir, "loss"), res["loss"]["fig"], **save_kwargs)
    visualize.save_fig(os.path.join(log_dir, "affine"), res["affine"]["fig"], **save_kwargs)
    visualize.save_figs(log_dir, heatmaps.items(), **save_kwargs)

    return heatmaps


if __name__ == "__main__":
    script_utils.experiment_main(visualize_divergence_heatmap_ex, "visualize_divergence_heatmap")
