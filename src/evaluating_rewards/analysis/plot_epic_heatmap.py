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

import os
from typing import Any, Iterable, Mapping, Optional, Tuple

from imitation import util
import sacred

from evaluating_rewards import serialize
from evaluating_rewards.analysis import (
    dissimilarity_heatmap_config,
    results,
    stylesheets,
    visualize,
)
from evaluating_rewards.scripts import script_utils

plot_divergence_heatmap_ex = sacred.Experiment("plot_divergence_heatmap")


dissimilarity_heatmap_config.make_config(plot_divergence_heatmap_ex)


@plot_divergence_heatmap_ex.config
def default_config():
    """Default configuration values."""
    # Dataset parameters
    log_root = serialize.get_output_dir()  # where results are read from/written to
    data_root = os.path.join(log_root, "comparison")  # root of comparison data directory
    data_subdir = "hardcoded"  # optional, if omitted searches all data (slow)
    search = {}  # parameters to filter by in datasets

    _ = locals()
    del _


@plot_divergence_heatmap_ex.config
def search_config(search, env_name):
    search["env_name"] = env_name


@plot_divergence_heatmap_ex.config
def logging_config(log_root, search):
    log_dir = os.path.join(  # noqa: F841  pylint:disable=unused-variable
        log_root, "plot_epic_heatmap", str(search).replace("/", "_"), util.make_unique_timestamp(),
    )


@plot_divergence_heatmap_ex.named_config
def test():
    """Intended for debugging/unit test."""
    data_root = os.path.join("tests", "data")
    data_subdir = "comparison"
    search = {
        "env_name": "evaluating_rewards/PointMassLine-v0",
    }
    # Do not include "tex" in styles here: this will break on CI.
    styles = ["paper", "heatmap-1col"]
    _ = locals()
    del _


@plot_divergence_heatmap_ex.named_config
def normalize():
    heatmap_kwargs = {  # noqa: F841  pylint:disable=unused-variable
        "normalize": True,
    }


@plot_divergence_heatmap_ex.main
def plot_divergence_heatmap(
    x_reward_cfgs: Iterable[Tuple[str, str]],
    y_reward_cfgs: Iterable[Tuple[str, str]],
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
        x_reward_cfgs: tuples of reward_type and reward_path for x-axis (target).
        y_reward_cfgs: tuples of reward_type and reward_path for y-axis (source).
        styles: styles to apply from `evaluating_rewards.analysis.stylesheets`.
        data_root: where to load data from.
        data_subdir: subdirectory to load data from.
        search: mapping which Sacred configs must match to be included in results.
        heatmap_kwargs: passed through to `analysis.compact_heatmaps`.
        log_dir: directory to write figures and other logging to.
        save_kwargs: passed through to `analysis.save_figs`.
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

    keys = ("source_reward_type", "source_reward_path", "target_reward_type", "seed")
    stats = results.load_multiple_stats(data_dir, keys, cfg_filter=cfg_filter)
    res = results.pipeline(stats)
    loss = res["loss"]["loss"]

    with stylesheets.setup_styles(styles):
        figs = {}
        figs["loss"] = visualize.loss_heatmap(loss, res["loss"]["unwrapped_loss"])
        figs["affine"] = visualize.affine_heatmap(
            res["affine"]["scales"], res["affine"]["constants"]
        )
        heatmaps = visualize.compact_heatmaps(dissimilarity=loss, **heatmap_kwargs)
        figs.update(heatmaps)
        visualize.save_figs(log_dir, figs.items(), **save_kwargs)

        return figs


if __name__ == "__main__":
    script_utils.experiment_main(plot_divergence_heatmap_ex, "plot_divergence_heatmap")
