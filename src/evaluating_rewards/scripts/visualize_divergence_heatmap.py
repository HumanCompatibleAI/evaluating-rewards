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

import os
from typing import Any, Dict

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
    data_kind = "hardcoded"  # subdirectory to read from
    search = {  # parameters to filter by in datasets
        "env_name": "evaluating_rewards/Hopper-v3",
    }

    # Figure parameters
    # TODO: style sheet to apply?
    save_kwargs = {
        "fmt": "pdf",
    }

    _ = locals()
    del _


@visualize_divergence_heatmap_ex.config
def logging_config(log_root, search, data_kind):
    # TODO: timestamp?
    log_dir = os.path.join(  # noqa: F841  pylint:disable=unused-variable
        log_root,
        "visualize_divergence_heatmap",
        ":".join(f'{k}={v.replace("/", "_")}' for k, v in search.items()),
        data_kind,
    )


@visualize_divergence_heatmap_ex.main
def visualize_divergence_heatmap(
    data_root: str,
    data_kind: str,
    search: Dict[str, Any],
    log_dir: str,
    save_kwargs: Dict[str, Any],
):
    """Entry-point into script to produce divergence heatmaps.

    Args:
        data_root: where to load data from.
        data_kind: subdirectory to load data from.
        search: dict which Sacred configs must match to be included in results.
        log_dir: directory to write figures and other logging to.
        save_kwargs: passed through to `visualize.save_figs`.
        """
    data_dir = os.path.join(data_root, data_kind)
    # TODO: make keys kind dependent?
    keys = ["source_reward_type", "target_reward_type", "seed"]

    def cfg_filter(cfg):
        return all([cfg[k] == v for k, v in search.items()])

    # TODO: do I need to load everything when I just use loss?
    stats = results.load_multiple_stats(data_dir, keys, cfg_filter=cfg_filter)
    stats = results.pipeline(stats)
    # TODO: handle sequences of heatmaps? functions? or just make dict from env to fn?
    loss = stats["loss"]["loss"]
    # TODO: usetex, Apple Color Emoji, make fonts match doc
    heatmaps = results.compact_heatmaps(
        loss=loss, order=loss.index.levels[0], masks={"all": [results.always_true]},
    )
    visualize.save_figs(log_dir, heatmaps.items(), **save_kwargs)


if __name__ == "__main__":
    script_utils.experiment_main(visualize_divergence_heatmap_ex, "visualize_divergence_heatmap")
