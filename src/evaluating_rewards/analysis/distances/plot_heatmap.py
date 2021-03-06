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

"""CLI script to plot heatmap of previously computed distances between reward models."""

import functools
import logging
import os
import pickle
from typing import Any, Callable, Iterable, Mapping, MutableMapping

from imitation.util import util as imit_util
from matplotlib import pyplot as plt
import pandas as pd
import sacred

from evaluating_rewards import serialize
from evaluating_rewards.analysis import results, stylesheets, visualize
from evaluating_rewards.analysis.distances import aggregated, heatmaps, reward_masks
from evaluating_rewards.distances import common_config
from evaluating_rewards.scripts import script_utils

plot_heatmap_ex = sacred.Experiment("plot_heatmap")
logger = logging.getLogger("evaluating_rewards.analysis.plot_heatmap")


# Default configs (always applied.)


@plot_heatmap_ex.config
def default_config():
    """Default configuration values."""
    data_root = serialize.get_output_dir()  # where values are read from
    log_root = serialize.get_output_dir()  # where results are written to
    vals_path = None  # aggregated data values

    # Reward configurations: models to compare
    x_reward_cfgs = None
    y_reward_cfgs = None
    heatmap_kwargs = {}

    _ = locals()
    del _


@plot_heatmap_ex.config
def logging_config(env_name, log_root):
    """Default logging configuration: hierarchical directory structure based on config."""
    timestamp = imit_util.make_unique_timestamp()
    log_dir = os.path.join(
        log_root,
        "plot_heatmap",
        env_name,
        timestamp,
    )
    _ = locals()
    del _


@plot_heatmap_ex.config
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


# General formatting instructions.
# For example, how large a figure? What precision?


@plot_heatmap_ex.named_config
def large():
    """Large output size, high precision."""
    styles = ["paper", "heatmap", "heatmap-1col", "tex"]
    heatmap_kwargs = {
        "fmt": heatmaps.short_e,
        "cbar_kws": dict(fraction=0.05),
    }
    _ = locals()
    del _


@plot_heatmap_ex.named_config
def paper():
    """Figures suitable for inclusion in paper."""
    # TODO(adam): how to choose whether left, middle, right...?
    # Or perhaps just plot all the algorithms jointly?
    styles = ["paper", "heatmap", "heatmap-3col", "heatmap-3col-left", "tex"]
    heatmap_kwargs = {"cbar": False, "vmin": 0.0, "vmax": 1.0}
    _ = locals()
    del _


@plot_heatmap_ex.named_config
def test():
    """Intended for debugging/unit test."""
    locals().update(**common_config.COMMON_CONFIGS["point_mass"])
    data_root = os.path.join("tests/data")
    vals_path = "aggregated/aggregated.pkl"
    # Do not include "tex" in styles here: this will break on CI.
    styles = ["paper", "heatmap-1col"]
    _ = locals()
    del _


# Environment specific configs.
# (Mostly special-case formatting to make presentation clearer.)


def _norm(args: Iterable[str]) -> bool:
    return any(reward_masks.match("evaluating_rewards/PointMassGroundTruth-v0")(args))


def _hopper_activity(args: Iterable[str]) -> bool:
    pattern = r"evaluating_rewards/(.*)(GroundTruth|Backflip)(.*)"
    repl = reward_masks.replace(pattern, r"\1\2")(args)
    return len(set(repl)) > 1 and reward_masks.no_ctrl(args)


@plot_heatmap_ex.named_config
def point_mass():
    """Heatmaps for evaluating_rewards/PointMass* environments."""
    locals().update(**common_config.COMMON_CONFIGS["point_mass"])
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


@plot_heatmap_ex.named_config
def point_maze():
    """Heatmaps for imitation/PointMaze{Left,Right}-v0 environments."""
    locals().update(**common_config.COMMON_CONFIGS["point_maze"])


@plot_heatmap_ex.named_config
def point_maze_learned():
    """Compare rewards learned in PointMaze to the ground-truth reward."""
    # Analyzes models generated by `runners/transfer_point_maze.sh`.
    locals().update(**common_config.COMMON_CONFIGS["point_maze_learned"])


@plot_heatmap_ex.named_config
def half_cheetah():
    """Heatmaps for seals/HalfCheetah-v0."""
    locals().update(**common_config.COMMON_CONFIGS["half_cheetah"])
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


@plot_heatmap_ex.named_config
def hopper():
    """Heatmaps for seals/Hopper-v0."""
    locals().update(**common_config.COMMON_CONFIGS["hopper"])
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


def _usual_label_fstr(distance_name: str) -> str:
    return "{transform_start}" + distance_name + "({args}){transform_end}"


def pretty_label_fstr(name: str) -> str:
    """Abbreviation to use in legend for colorbar."""
    if name.endswith("lower"):
        return _usual_label_fstr("D_L")
    elif name.endswith("upper"):
        return _usual_label_fstr("D_U")
    elif name.endswith("middle") or name.endswith("mean"):
        return _usual_label_fstr(r"\bar{{D}}")
    elif name.endswith("width"):
        return _usual_label_fstr("D_W")
    elif name.endswith("sd"):
        return r"\mathrm{{SD}}\left[" + _usual_label_fstr("D") + r"\right]"
    else:
        return _usual_label_fstr("D")


def multi_heatmaps(
    dissimilarities: Mapping[str, pd.Series], **kwargs
) -> MutableMapping[str, plt.Figure]:
    """Plot heatmap for each dissimilarity series in `dissimilarities`.

    Args:
        dissimilarities: Mapping from strings to dissimilarity matrix.
        kwargs: Passed through to `heatmaps.compact_heatmaps`.

    Returns:
        A Mapping from strings to figures, with keys "{k}_{mask}" for mask config `mask` from
        `dissimilarities[k]`.
    """
    figs = {}
    for name, val in dissimilarities.items():
        label_fstr = pretty_label_fstr(name)
        extra_kwargs = {}
        if name.endswith("width"):
            extra_kwargs["fmt"] = functools.partial(heatmaps.short_e, precision=0)
        heatmap_figs = heatmaps.compact_heatmaps(
            dissimilarity=val, label_fstr=label_fstr, **kwargs, **extra_kwargs
        )
        figs.update({f"{name}_{k}": v for k, v in heatmap_figs.items()})
    return figs


def _subplot_heatmap(
    data: Iterable[pd.Series], labels: Iterable[pd.Series], kwargs: Iterable[Mapping[str, Any]]
) -> plt.Figure:
    """Heatmap each of the Series in `data` on a separate subplot."""
    data = tuple(data)
    labels = tuple(labels)
    kwargs = tuple(kwargs)
    ncols = len(data)
    assert ncols == len(labels)
    assert ncols == len(kwargs)

    width, height = plt.rcParams.get("figure.figsize")
    fig, axs = plt.subplots(ncols, 1, figsize=(ncols * width, height), squeeze=True)

    for series, lab, kw, ax in zip(data, labels, kwargs, axs):
        heatmaps.comparison_heatmap(series, ax=ax, **kw)
        ax.set_title(lab)

    return fig


def loss_heatmap(loss: pd.Series, unwrapped_loss: pd.Series) -> plt.Figure:
    return _subplot_heatmap([loss, unwrapped_loss], ["Loss", "Unwrapped Loss"], [{}, {}])


def affine_heatmap(scales: pd.Series, constants: pd.Series) -> plt.Figure:
    return _subplot_heatmap(
        [scales, constants], ["Scale", "Constant"], [dict(robust=True), dict(log=False, center=0.0)]
    )


@plot_heatmap_ex.capture
def _plot_heatmap(
    fn: Callable[[], Mapping[str, plt.Figure]],
    vals_path: str,
    styles: Iterable[str],
    styles_for_env: Iterable[str],
    log_dir: str,
    timestamp: str,
    save_kwargs: Mapping[str, Any],
) -> None:
    """Plots a figure for each entry loaded from `vals_path`.

    Args:
        vals_path: path to pickle file containing aggregated values.
            Produced by `evaluating_rewards.scripts.distances.*`.
        data_root: the root with respect to canonicalize reward configurations.
        x_reward_cfgs: tuples of reward_type and reward_path for x-axis.
        y_reward_cfgs: tuples of reward_type and reward_path for y-axis.
        styles: styles to apply from `evaluating_rewards.analysis.stylesheets`.
        styles_for_env: extra styles to apply, concatenated with above.
        log_dir: directory to save data to.
        timestamp: timestamp + unique identifier, usually a component of `log_dir`.
        heatmap_kwargs: passed through to `heatmaps.compact_heatmaps`.
        save_kwargs: passed through to `analysis.save_figs`.
    """
    logging.info("Plotting figures")
    vals_dir = os.path.dirname(vals_path)
    plots_sym_dir = os.path.join(vals_dir, "plots")
    os.makedirs(plots_sym_dir, exist_ok=True)
    plots_sym_path = os.path.join(plots_sym_dir, timestamp)
    os.symlink(log_dir, plots_sym_path)

    styles = list(styles) + list(styles_for_env)
    with stylesheets.setup_styles(styles):
        try:
            figs = fn()
            visualize.save_figs(log_dir, figs.items(), **save_kwargs)
        finally:
            for fig in figs:
                plt.close(fig)


@plot_heatmap_ex.command
def plot_npec_stats(
    vals_path: str,
) -> None:
    """Plot NPEC-specific statistics, such as affine transformation parameters.

    Args:
        vals_path: Path to NPEC statistics pickle file.
    """
    with open(vals_path, "rb") as f:
        npec_stats = pickle.load(f)
    npec_stats = {k + (i,): inner_v for k, v in npec_stats.items() for i, inner_v in enumerate(v)}
    npec_res = results.pipeline(npec_stats)

    def fn():
        figs = {}
        figs["loss"] = loss_heatmap(npec_res["loss"]["loss"], npec_res["loss"]["unwrapped_loss"])
        figs["affine"] = affine_heatmap(
            npec_res["affine"]["scales"], npec_res["affine"]["constants"]
        )
        return figs

    _plot_heatmap(fn)  # pylint:disable=no-value-for-parameter


@plot_heatmap_ex.main
def plot_distances(
    vals_path: str,
    data_root: str,
    x_reward_cfgs: Iterable[common_config.RewardCfg],
    y_reward_cfgs: Iterable[common_config.RewardCfg],
    heatmap_kwargs: Mapping[str, Any],
) -> None:
    """Plots a figure for each entry loaded from `vals_path`.

    Args:
        vals_path: path to pickle file containing aggregated values.
            Produced by `evaluating_rewards.scripts.distances.*`.
        data_root: the root with respect to canonicalize reward configurations.
        x_reward_cfgs: tuples of reward_type and reward_path for x-axis.
        y_reward_cfgs: tuples of reward_type and reward_path for y-axis.
        styles: styles to apply from `evaluating_rewards.analysis.stylesheets`.
        styles_for_env: extra styles to apply, concatenated with above.
        log_dir: directory to save data to.
        timestamp: timestamp + unique identifier, usually a component of `log_dir`.
        heatmap_kwargs: passed through to `heatmaps.compact_heatmaps`.
        save_kwargs: passed through to `analysis.save_figs`.
    """
    # Sacred turns our tuples into lists :(, undo
    x_reward_cfgs = [common_config.canonicalize_reward_cfg(cfg, data_root) for cfg in x_reward_cfgs]
    y_reward_cfgs = [common_config.canonicalize_reward_cfg(cfg, data_root) for cfg in y_reward_cfgs]

    # Take `vals_path` relative to data_root, if `vals_path` not absolute
    vals_path = os.path.join(data_root, vals_path)
    with open(vals_path, "rb") as f:
        raw = pickle.load(f)
    raw = aggregated.select_subset(raw, x_reward_cfgs, y_reward_cfgs)
    vals = {k: aggregated.oned_mapping_to_series(v) for k, v in raw.items()}

    _plot_heatmap(  # pylint:disable=no-value-for-parameter
        lambda: multi_heatmaps(vals, **heatmap_kwargs)
    )


if __name__ == "__main__":
    script_utils.experiment_main(plot_heatmap_ex, "plot_heatmap")
