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

import itertools
import tempfile

import pandas as pd
import sacred
import xarray as xr

from evaluating_rewards.analysis.distances import plot_gridworld_heatmap, plot_heatmap
from evaluating_rewards.analysis.reward_figures import plot_gridworld_reward, plot_pm_reward
from evaluating_rewards.scripts.distances import epic, erc, npec, rollout_return
from evaluating_rewards.scripts.pipeline import combined_distances, train_experts
from evaluating_rewards.scripts.rewards import train_preferences, train_regress
from tests import common


def _check_reward_cfg_type(o: object) -> None:
    assert isinstance(o, tuple)
    assert len(o) == 2
    assert isinstance(o[0], str)
    assert isinstance(o[1], str)


def _check_distance_return(run: sacred.experiment.Run) -> None:
    """Sanity checks return type and keys present."""
    assert isinstance(run.result, dict)
    key = next(iter(run.result.keys()))
    assert isinstance(key, str)
    val = next(iter(run.result.values()))
    assert isinstance(val, dict)
    inner_key = next(iter(val.keys()))
    assert isinstance(inner_key, tuple)
    assert len(inner_key) == 2
    _check_reward_cfg_type(inner_key[0])
    _check_reward_cfg_type(inner_key[1])
    inner_val = next(iter(val.values()))
    assert isinstance(inner_val, float)

    x_reward_cfgs = [tuple(x) for x in run.config["x_reward_cfgs"]]
    y_reward_cfgs = [tuple(y) for y in run.config["y_reward_cfgs"]]
    expected_keys = set(itertools.product(x_reward_cfgs, y_reward_cfgs))
    assert set(val.keys()) == expected_keys


EXPERIMENTS = {
    # experiment, expected_type, extra_named_configs, config_updates, extra_check
    "epic_distance": (
        epic.epic_distance_ex,
        dict,
        [],
        {},
        _check_distance_return,
    ),
    "erc_distance": (erc.erc_distance_ex, dict, [], {}, _check_distance_return),
    "npec_distance": (npec.npec_distance_ex, dict, [], {}, _check_distance_return),
    "rollout_distance": (rollout_return.rollout_distance_ex, dict, [], {}, _check_distance_return),
    "npec_distance_alternating": (
        npec.npec_distance_ex,
        dict,
        ["alternating_maximization"],
        {"fit_kwargs": {"epoch_timesteps": 512}},
        _check_distance_return,
    ),
    "erc_distance_spearman": (
        erc.erc_distance_ex,
        dict,
        [],
        {"corr_kind": "spearman"},
        _check_distance_return,
    ),
    "plot_gridworld_heatmap": (
        plot_gridworld_heatmap.plot_gridworld_heatmap_ex,
        type(None),
        [],
        {},
        None,
    ),
    "plot_gridworld_reward": (
        plot_gridworld_reward.plot_gridworld_reward_ex,
        type(None),
        [],
        {},
        None,
    ),
    "plot_heatmap": (
        plot_heatmap.plot_heatmap_ex,
        type(None),
        [],
        {},
        None,
    ),
    "plot_pm_reward": (plot_pm_reward.plot_pm_reward_ex, xr.DataArray, [], {}, None),
    "combined_distances": (combined_distances.combined_distances_ex, type(None), [], {}, None),
    "preferences": (train_preferences.train_preferences_ex, pd.DataFrame, [], {}, None),
    "regress": (train_regress.train_regress_ex, dict, [], {}, None),
    "train_experts": (train_experts.experts_ex, dict, [], {}, None),
}


def add_epic_experiments():
    """Add testcases for `evaluating_rewards.distances.epic`."""
    for computation_kind in ["sample", "mesh"]:
        for distance_kind in ["direct", "pearson"]:
            EXPERIMENTS[f"epic_distance_{computation_kind}_{distance_kind}"] = (
                epic.epic_distance_ex,
                dict,
                [],
                {"computation_kind": computation_kind, "distance_kind": distance_kind},
                _check_distance_return,
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
        EXPERIMENTS[f"epic_distance_{name}"] = (
            epic.epic_distance_ex,
            dict,
            named_configs,
            {},
            _check_distance_return,
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
            None,
        )


add_epic_experiments()
add_gridworld_experiments()


@common.mark_parametrize_dict(
    "experiment,expected_type,named_configs,config_updates,extra_check", EXPERIMENTS
)
def test_experiment(experiment, expected_type, named_configs, config_updates, extra_check):
    """Run Sacred experiment and sanity check run, including type of result."""
    named_configs = named_configs + ["test"]
    with tempfile.TemporaryDirectory(prefix="eval-rewards-exp") as tmpdir:
        config_updates["log_root"] = tmpdir
        run = experiment.run(named_configs=named_configs, config_updates=config_updates)
    assert run.status == "COMPLETED"
    assert isinstance(run.result, expected_type)
    if extra_check is not None:
        extra_check(run)
