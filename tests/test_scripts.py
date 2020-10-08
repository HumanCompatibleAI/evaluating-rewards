# Copyright 2019 DeepMind Technologies Limited and Adam Gleave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Smoke tests for CLI scripts."""

import tempfile

import pandas as pd
import xarray as xr

from evaluating_rewards.analysis.dissimilarity_heatmaps import (
    plot_epic_heatmap,
    plot_erc_heatmap,
    plot_gridworld_heatmap,
    plot_npec_heatmap,
    table_combined,
)
from evaluating_rewards.analysis.reward_figures import plot_gridworld_reward, plot_pm_reward
from evaluating_rewards.scripts import npec_comparison, train_preferences, train_regress
from tests import common

EXPERIMENTS = {
    # experiment, expected_type, extra_named_configs, config_updates
    "plot_epic_heatmap": (plot_epic_heatmap.plot_epic_heatmap_ex, type(None), [], {}),
    "plot_npec_heatmap": (plot_npec_heatmap.plot_npec_heatmap_ex, type(None), [], {}),
    "plot_erc_heatmap": (plot_erc_heatmap.plot_erc_heatmap_ex, type(None), [], {}),
    "plot_erc_heatmap_spearman": (
        plot_erc_heatmap.plot_erc_heatmap_ex,
        type(None),
        [],
        {"corr_kind": "spearman"},
    ),
    "plot_gridworld_heatmap": (
        plot_gridworld_heatmap.plot_gridworld_heatmap_ex,
        type(None),
        [],
        {},
    ),
    "plot_gridworld_reward": (plot_gridworld_reward.plot_gridworld_reward_ex, type(None), [], {}),
    "plot_pm_reward": (plot_pm_reward.plot_pm_reward_ex, xr.DataArray, [], {}),
    "table_combined": (table_combined.table_combined_ex, type(None), [], {}),
    "comparison": (npec_comparison.npec_comparison_ex, dict, [], {}),
    "comparison_alternating": (
        npec_comparison.npec_comparison_ex,
        dict,
        ["alternating_maximization"],
        {"fit_kwargs": {"epoch_timesteps": 4096}},
    ),
    "preferences": (train_preferences.train_preferences_ex, pd.DataFrame, [], {}),
    "regress": (train_regress.train_regress_ex, dict, [], {}),
}


def add_epic_experiments():
    """Add testcases for `evaluating_rewards.analysis.dissimilarity_heatmaps.plot_epic_heatmap`."""
    for computation_kind in ["sample", "mesh"]:
        for distance_kind in ["direct", "pearson"]:
            EXPERIMENTS[f"plot_epic_heatmap_{computation_kind}_{distance_kind}"] = (
                plot_epic_heatmap.plot_epic_heatmap_ex,
                type(None),
                [],
                {"computation_kind": computation_kind, "distance_kind": distance_kind},
            )
    NAMED_CONFIGS = {
        "random_spaces": ["point_mass", "sample_from_env_spaces"],
        "random_spaces_both": ["point_mass", "sample_from_env_spaces", "dataset_iid"],
        "random_transitions_both": [
            "point_mass",
            "dataset_from_random_transitions",
        ],
    }
    for name, named_configs in NAMED_CONFIGS.items():
        EXPERIMENTS[f"plot_canon_heatmap_{name}"] = (
            plot_epic_heatmap.plot_epic_heatmap_ex,
            type(None),
            named_configs,
            {},
        )


def add_gridworld_experiments():
    """Adds experiments for `plot_gridworld_heatmap`."""
    kinds = [
        "npec",
        "asymmetric",
        "symmetric",
        "symmetric_min",
    ]
    for canonical in plot_gridworld_heatmap.CANONICAL_DESHAPE_FN:
        kinds += [f"{canonical}_direct", f"{canonical}_pearson"]
    for kind in kinds:
        EXPERIMENTS[f"plot_gridworld_heatmap_{kind}"] = (
            plot_gridworld_heatmap.plot_gridworld_heatmap_ex,
            type(None),
            [],
            {"kind": kind},
        )


add_epic_experiments()
add_gridworld_experiments()


@common.mark_parametrize_dict("experiment,expected_type,named_configs,config_updates", EXPERIMENTS)
def test_experiment(experiment, expected_type, named_configs, config_updates):
    named_configs = ["test"] + named_configs
    with tempfile.TemporaryDirectory(prefix="eval-rewards-exp") as tmpdir:
        config_updates["log_root"] = tmpdir
        run = experiment.run(named_configs=named_configs, config_updates=config_updates)
    assert run.status == "COMPLETED"
    assert isinstance(run.result, expected_type)
