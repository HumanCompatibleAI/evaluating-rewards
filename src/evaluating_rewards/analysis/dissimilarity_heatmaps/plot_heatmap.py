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
from typing import Any, Iterable, Mapping, Tuple

from imitation.util import util as imit_util
from matplotlib import pyplot as plt
import pandas as pd
import sacred

from evaluating_rewards import serialize
from evaluating_rewards.analysis import stylesheets, visualize
from evaluating_rewards.analysis.dissimilarity_heatmaps import cli_common, heatmaps
from evaluating_rewards.distances import transitions_datasets
from evaluating_rewards.scripts import script_utils
from evaluating_rewards.scripts.distance import common

plot_heatmap_ex = sacred.Experiment("plot_heatmap")
logger = logging.getLogger("evaluating_rewards.analysis.plot_heatmap")


cli_common.make_config(plot_heatmap_ex)
transitions_datasets.make_config(plot_heatmap_ex)


@plot_heatmap_ex.config
def default_config():
    """Default configuration values."""
    data_root = serialize.get_output_dir()  # where values are read from
    log_root = serialize.get_output_dir()  # where results are written to
    vals_path = None

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
    # Do not include "tex" in styles here: this will break on CI.
    styles = ["paper", "heatmap-1col"]
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


def multi_heatmaps(dissimilarities: Mapping[str, pd.Series], **kwargs) -> Mapping[str, plt.Figure]:
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


def oned_mapping_to_series(
    vals: Mapping[Tuple[common.RewardCfg, common.RewardCfg], float]
) -> pd.Series:
    """Converts mapping to a series.

    Args:
        vals: A mapping from pairs of configurations to a float.

    Returns:
        A Series with a multi-index based on the configurations.
    """
    vals = {(xtype, xpath, ytype, ypath): v for ((xtype, xpath), (ytype, ypath)), v in vals.items()}
    vals = pd.Series(vals)
    vals.index.names = [
        "target_reward_type",
        "target_reward_path",
        "source_reward_type",
        "source_reward_path",
    ]
    return vals


def twod_mapping_to_multi_series(
    aggregated: Mapping[str, Mapping[Any, float]]
) -> Mapping[str, pd.Series]:
    """Converts a nested mapping to a mapping of dissimilarity series.

    Args:
        aggregated: A mapping over a mapping from strings to sequences of floats.

    Returns:
        A mapping from strings to MultiIndex series returned by `oned_mapping_to_series`,
        after transposing the inner and outer keys of the mapping.
    """
    return {k: oned_mapping_to_series(v) for k, v in aggregated.items()}


def select_subset(
    vals: common.AggregatedDistanceReturn,
    x_reward_cfgs: Iterable[common.RewardCfg],
    y_reward_cfgs: Iterable[common.RewardCfg],
) -> common.AggregatedDistanceReturn:
    return {
        k: {(x, y): v[(x, y)] for x in x_reward_cfgs for y in y_reward_cfgs}
        for k, v in vals.items()
    }


@plot_heatmap_ex.main
def plot_heatmap(
    vals_path: str,
    data_root: str,
    x_reward_cfgs: Iterable[common.RewardCfg],
    y_reward_cfgs: Iterable[common.RewardCfg],
    styles: Iterable[str],
    styles_for_env: Iterable[str],
    log_dir: str,
    timestamp: str,
    heatmap_kwargs: Mapping[str, Any],
    save_kwargs: Mapping[str, Any],
) -> None:
    """Plots a figure for each entry loaded from `vals_path`."""
    # TODO(adam): document args in docstring
    # Sacred turns our tuples into lists :(, undo
    x_reward_cfgs = [common.canonicalize_reward_cfg(cfg, data_root) for cfg in x_reward_cfgs]
    y_reward_cfgs = [common.canonicalize_reward_cfg(cfg, data_root) for cfg in y_reward_cfgs]
    # TODO(adam): how to specify vals_path?
    with open(vals_path, "rb") as f:
        aggregated_raw = pickle.load(f)
    aggregated_raw = select_subset(aggregated_raw, x_reward_cfgs, y_reward_cfgs)
    aggregated = twod_mapping_to_multi_series(aggregated_raw)

    logging.info("Plotting figures")
    vals_dir = os.path.dirname(vals_path)
    plots_sym_dir = os.path.join(vals_dir, "plots")
    os.makedirs(plots_sym_dir, exist_ok=True)
    plots_sym_path = os.path.join(plots_sym_dir, timestamp)
    os.symlink(log_dir, plots_sym_path)

    styles = list(styles) + list(styles_for_env)
    with stylesheets.setup_styles(styles):
        try:
            figs = multi_heatmaps(aggregated, **heatmap_kwargs)
            visualize.save_figs(log_dir, figs.items(), **save_kwargs)
        finally:
            for fig in figs:
                plt.close(fig)


if __name__ == "__main__":
    script_utils.experiment_main(plot_heatmap_ex, "plot_heatmap")
