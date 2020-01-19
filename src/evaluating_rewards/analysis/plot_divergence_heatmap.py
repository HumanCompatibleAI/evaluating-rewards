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

# pylint:disable=wrong-import-position,wrong-import-order,ungrouped-imports
import matplotlib  # isort:skip

# Need PGF to use LaTeX with includegraphics
matplotlib.use("pgf")  # noqa: E402

# isort: imports-stdlib
import itertools
import os
from typing import Any, Iterable, Mapping, Optional

from imitation import util
import matplotlib.pyplot as plt
import sacred

from evaluating_rewards import serialize
from evaluating_rewards.analysis import results, stylesheets, visualize
from evaluating_rewards.scripts import script_utils

plot_divergence_heatmap_ex = sacred.Experiment("plot_divergence_heatmap")


def horizontal_ticks() -> None:
    plt.xticks(rotation="horizontal")
    plt.yticks(rotation="horizontal")


@plot_divergence_heatmap_ex.config
def default_config():
    """Default configuration values."""
    # Dataset parameters
    log_root = serialize.get_output_dir()  # where results are read from/written to
    data_root = os.path.join(log_root, "comparison")  # root of comparison data directory
    data_subdir = "hardcoded"  # optional, if omitted searches all data (slow)
    search = {  # parameters to filter by in datasets
        "env_name": "evaluating_rewards/Hopper-v3",
        "model_wrapper_kwargs": {},
    }

    # Figure parameters
    heatmap_kwargs = {
        "masks": {"all": [visualize.always_true]},
        "order": None,
        "after_plot": horizontal_ticks,
    }
    styles = ["paper", "paper-1col", "tex"]
    save_kwargs = {
        "fmt": "pdf",
    }

    _ = locals()
    del _


@plot_divergence_heatmap_ex.config
def logging_config(log_root, search):
    # TODO: timestamp?
    log_dir = os.path.join(  # noqa: F841  pylint:disable=unused-variable
        log_root,
        "plot_divergence_heatmap",
        str(search).replace("/", "_"),
        util.make_unique_timestamp(),
    )


def _norm(args: Iterable[str]) -> bool:
    return any(visualize.match("evaluating_rewards/PointMassGroundTruth-v0")(args))


@plot_divergence_heatmap_ex.named_config
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
        "diagonal": [visualize.zero, visualize.same],
        "control": [visualize.zero, visualize.control],
        "dense_vs_sparse": [visualize.zero, visualize.sparse_or_dense],
        "norm": [visualize.zero, visualize.same, _norm],
        "all": [visualize.always_true],
    }
    order = ["SparseNoCtrl", "Sparse", "DenseNoCtrl", "Dense", "GroundTruth"]
    heatmap_kwargs["order"] = [f"evaluating_rewards/PointMass{label}-v0" for label in order]
    heatmap_kwargs["after_plot"] = horizontal_ticks
    del order


@plot_divergence_heatmap_ex.named_config
def point_maze():
    """Heatmaps for imitation/PointMaze{Left,Right}-v0 environments."""
    search = {
        "env_name": "evaluating_rewards/PointMazeLeft-v0",
    }
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


MUJOCO_STANDARD_ORDER = ["ForwardNoCtrl", "ForwardWithCtrl", "BackwardNoCtrl", "BackwardWithCtrl"]


@plot_divergence_heatmap_ex.named_config
def half_cheetah():
    """Heatmaps for HalfCheetah-v3."""
    search = {
        "env_name": "evaluating_rewards/HalfCheetah-v3",
    }
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


def hopper_activity(args: Iterable[str]) -> bool:
    pattern = r"evaluating_rewards/(.*)(GroundTruth|Backflip)(.*)"
    repl = visualize.replace(pattern, r"\1\2")(args)
    return len(set(repl)) > 1 and visualize.no_ctrl(args)


@plot_divergence_heatmap_ex.named_config
def hopper():
    """Heatmaps for Hopper-v3."""
    search = {  # noqa: F841  pylint:disable=unused-variable
        "env_name": "evaluating_rewards/Hopper-v3",
    }
    heatmap_kwargs = {}
    heatmap_kwargs["masks"] = {
        "diagonal": [visualize.zero, visualize.same],
        "control": [visualize.zero, visualize.control],
        "direction": [visualize.zero, visualize.direction],
        "no_ctrl": [visualize.zero, visualize.no_ctrl],
        "different_activity": [visualize.zero, hopper_activity],
        "all": [visualize.always_true],
    }
    activities = ["GroundTruth", "Backflip"]
    heatmap_kwargs["order"] = [
        f"evaluating_rewards/Hopper{prefix}{suffix}-v0"
        for prefix, suffix in itertools.product(activities, MUJOCO_STANDARD_ORDER)
    ]
    heatmap_kwargs["after_plot"] = horizontal_ticks
    del activities


@plot_divergence_heatmap_ex.main
def plot_divergence_heatmap(
    styles: Iterable[str],
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
        heatmap_kwargs: passed through to `analysis.compact_heatmaps`.
        log_dir: directory to write figures and other logging to.
        save_kwargs: passed through to `analysis.save_figs`.
        """
    if "tex" in styles:
        os.environ["TEXINPUTS"] = stylesheets.LATEX_DIR + ":"
    for style in styles:
        plt.style.use(stylesheets.STYLES[style])

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
    heatmap_kwargs = dict(heatmap_kwargs)
    if heatmap_kwargs.get("order") is None:
        heatmap_kwargs["order"] = loss.index.levels[0]

    figs = {}
    figs["loss"] = visualize.loss_heatmap(loss, res["loss"]["unwrapped_loss"])
    figs["affine"] = visualize.affine_heatmap(res["affine"]["scales"], res["affine"]["constants"])
    heatmaps = visualize.compact_heatmaps(loss=loss, **heatmap_kwargs)
    figs.update(heatmaps)
    visualize.save_figs(log_dir, figs.items(), **save_kwargs)

    return figs


if __name__ == "__main__":
    script_utils.experiment_main(plot_divergence_heatmap_ex, "plot_divergence_heatmap")
