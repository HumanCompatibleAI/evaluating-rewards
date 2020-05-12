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

"""CLI script to plot heatmap of EPIC distance between pairs of reward models."""

import logging
import os
from typing import Any, Dict, Iterable, Mapping, Optional

from imitation.util import util
from matplotlib import pyplot as plt
import pandas as pd
import sacred

from evaluating_rewards.analysis import results, stylesheets, visualize
from evaluating_rewards.analysis.dissimilarity_heatmaps import cli_common, heatmaps
from evaluating_rewards.scripts import script_utils

logger = logging.getLogger("evaluating_rewards.analysis.dissimilarity_heatmaps.plot_epic_Heatmap")
plot_epic_heatmap_ex = sacred.Experiment("plot_epic_heatmap")

cli_common.make_config(plot_epic_heatmap_ex)


@plot_epic_heatmap_ex.config
def default_config(log_root):
    """Default configuration values."""
    # Dataset parameters
    data_root = os.path.join(log_root, "comparison")  # root of comparison data directory
    data_subdir = "hardcoded"  # optional, if omitted searches all data (slow)
    search = {}  # parameters to filter by in datasets

    _ = locals()
    del _


@plot_epic_heatmap_ex.config
def search_config(search, env_name):
    search["env_name"] = env_name


@plot_epic_heatmap_ex.config
def logging_config(log_root, search):
    log_dir = os.path.join(  # noqa: F841  pylint:disable=unused-variable
        log_root, "plot_epic_heatmap", str(search).replace("/", "_"), util.make_unique_timestamp(),
    )


@plot_epic_heatmap_ex.named_config
def high_precision():
    """Compute tight confidence intervals for publication quality figures.

    Note the most important thing here is the number of seeds, which is not controllable here --
    need to run more instances of `evaluating_rewards.scripts.model_comparison`.
    """
    n_bootstrap = 10000
    _ = locals()
    del _


@plot_epic_heatmap_ex.named_config
def test():
    """Intended for debugging/unit test."""
    data_root = os.path.join("tests", "data")
    data_subdir = "comparison"
    search = {
        "env_name": "evaluating_rewards/PointMassLine-v0",
    }
    kinds = [
        "evaluating_rewards/PointMassGroundTruth-v0",
        "evaluating_rewards/PointMassSparseWithCtrl-v0",
    ]
    # Do not include "tex" in styles here: this will break on CI.
    styles = ["paper", "heatmap-1col"]
    _ = locals()
    del _


@plot_epic_heatmap_ex.named_config
def normalize():
    heatmap_kwargs = {  # noqa: F841  pylint:disable=unused-variable
        "normalize": True,
    }


def _multi_heatmap(
    data: Iterable[pd.Series], labels: Iterable[pd.Series], kwargs: Iterable[Dict[str, Any]]
) -> plt.Figure:
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
    return _multi_heatmap([loss, unwrapped_loss], ["Loss", "Unwrapped Loss"], [{}, {}])


def affine_heatmap(scales: pd.Series, constants: pd.Series) -> plt.Figure:
    return _multi_heatmap(
        [scales, constants], ["Scale", "Constant"], [dict(robust=True), dict(log=False, center=0.0)]
    )


@plot_epic_heatmap_ex.main
def plot_epic_heatmap(
    x_reward_cfgs: Iterable[cli_common.RewardCfg],
    y_reward_cfgs: Iterable[cli_common.RewardCfg],
    styles: Iterable[str],
    data_root: str,
    data_subdir: Optional[str],
    search: Mapping[str, Any],
    aggregate_fns: Mapping[str, cli_common.AggregateFn],
    heatmap_kwargs: Mapping[str, Any],
    log_dir: str,
    save_kwargs: Mapping[str, Any],
) -> Mapping[str, plt.Figure]:
    """Entry-point into script to produce divergence heatmaps.

    Args:
        x_reward_cfgs: tuples of reward_type and reward_path for x-axis (target).
        y_reward_cfgs: tuples of reward_type and reward_path for y-axis (source).
        styles: styles to apply from `evaluating_rewards.analysis.stylesheets`.
        data_root: where to load data from.
        data_subdir: subdirectory to load data from.
        search: mapping which Sacred configs must match to be included in results.
        aggregate_fns: Mapping from strings to aggregators to be applied on sequences of floats.
        heatmap_kwargs: passed through to `analysis.compact_heatmaps`.
        log_dir: directory to write figures and other logging to.
        save_kwargs: passed through to `analysis.save_figs`.

    Returns:
        A mapping of keywords to figures.
    """
    # Sacred turns our tuples into lists :(, undo
    x_reward_cfgs = [tuple(v) for v in x_reward_cfgs]
    y_reward_cfgs = [tuple(v) for v in y_reward_cfgs]

    data_dir = data_root
    if data_subdir is not None:
        data_dir = os.path.join(data_dir, data_subdir)
    # Workaround tags reserved by Sacred
    search = dict(search)
    for k, v in search.items():
        if isinstance(v, dict):
            search[k] = {inner_k.replace("escape/", ""): inner_v for inner_k, inner_v in v.items()}

    def cfg_filter(cfg):
        matches_search = all((cfg.get(k) == v for k, v in search.items()))
        source_cfg = cfg.get("source_reward_type"), cfg.get("source_reward_path")
        matches_source = source_cfg in y_reward_cfgs
        target_cfg = cfg.get("target_reward_type"), cfg.get("target_reward_path")
        matches_target = target_cfg in x_reward_cfgs
        return matches_search and matches_source and matches_target

    keys = "source_reward_type", "source_reward_path", "target_reward_type", "seed"
    stats = results.load_multiple_stats(data_dir, keys, cfg_filter=cfg_filter)
    res = results.pipeline(stats)
    loss = res["loss"]["loss"]

    vals = {}
    for name, aggregate_fn in aggregate_fns.items():
        logger.info(f"Aggregating {name}")
        aggregated = loss.groupby(list(keys[:-1])).apply(aggregate_fn)
        vals.update(
            {
                f"{name}_{k}": aggregated.loc[(slice(None), slice(None), slice(None), k)]
                for k in aggregated.index.levels[-1]
            }
        )

    with stylesheets.setup_styles(styles):
        figs = {}
        figs["loss"] = loss_heatmap(loss, res["loss"]["unwrapped_loss"])
        figs["affine"] = affine_heatmap(res["affine"]["scales"], res["affine"]["constants"])
        figs.update(cli_common.multi_heatmaps(vals, **heatmap_kwargs))
        visualize.save_figs(log_dir, figs.items(), **save_kwargs)

        return figs


if __name__ == "__main__":
    script_utils.experiment_main(plot_epic_heatmap_ex, "plot_epic_heatmap")
