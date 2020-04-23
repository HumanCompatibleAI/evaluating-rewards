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

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from evaluating_rewards.analysis.plot_canon_heatmap import plot_canon_heatmap_ex
from evaluating_rewards.analysis.plot_epic_heatmap import plot_epic_heatmap_ex
from evaluating_rewards.analysis.plot_gridworld_divergence import plot_gridworld_divergence_ex
from evaluating_rewards.analysis.plot_gridworld_reward import plot_gridworld_reward_ex
from evaluating_rewards.analysis.plot_pm_reward import plot_pm_reward_ex
from evaluating_rewards.scripts.model_comparison import model_comparison_ex
from evaluating_rewards.scripts.train_preferences import train_preferences_ex
from evaluating_rewards.scripts.train_regress import train_regress_ex
from tests import common

EXPERIMENTS = {
    # experiment, expected_type, extra_named_configs, config_updates
    "plot_canon_heatmap": (plot_canon_heatmap_ex, dict, [], {}),
    "plot_epic_heatmap": (plot_epic_heatmap_ex, dict, [], {}),
    "plot_pm": (plot_pm_reward_ex, xr.DataArray, [], {}),
    "plot_gridworld_divergence": (plot_gridworld_divergence_ex, dict, [], {}),
    "plot_gridworld_reward": (plot_gridworld_reward_ex, plt.Figure, [], {}),
    "comparison": (model_comparison_ex, dict, [], {}),
    "comparison_alternating": (
        model_comparison_ex,
        dict,
        ["alternating_maximization"],
        {"fit_kwargs": {"epoch_timesteps": 4096}},
    ),
    "regress": (train_regress_ex, dict, [], {}),
    "preferences": (train_preferences_ex, pd.DataFrame, [], {}),
}


def add_canon_experiments():
    for computation_kind in ["sample", "mesh"]:
        for distance_kind in ["direct", "pearson"]:
            EXPERIMENTS[f"plot_canon_heatmap_{computation_kind}_{distance_kind}"] = (
                plot_canon_heatmap_ex,
                dict,
                [],
                {"computation_kind": computation_kind, "distance_kind": distance_kind},
            )


add_canon_experiments()


@common.mark_parametrize_dict("experiment,expected_type,named_configs,config_updates", EXPERIMENTS)
def test_experiment(experiment, expected_type, named_configs, config_updates):
    named_configs = ["test"] + named_configs
    with tempfile.TemporaryDirectory(prefix="eval-rewards-exp") as tmpdir:
        config_updates["log_root"] = tmpdir
        run = experiment.run(named_configs=named_configs, config_updates=config_updates)
    assert run.status == "COMPLETED"
    assert isinstance(run.result, expected_type)
