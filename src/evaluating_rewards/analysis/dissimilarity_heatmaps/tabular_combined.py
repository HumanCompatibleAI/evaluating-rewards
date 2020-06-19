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

"""CLI script to make table of EPIC, NPEC and ERC distance from a reward model."""

import itertools
import logging
import os
from typing import Any, Mapping, Tuple

from imitation.util import util as imit_util
import pandas as pd
import sacred

from evaluating_rewards import serialize
from evaluating_rewards.analysis.dissimilarity_heatmaps import (
    cli_common,
    plot_epic_heatmap,
    plot_erc_heatmap,
    plot_npec_heatmap,
)
from evaluating_rewards.scripts import script_utils

tabular_combined_ex = sacred.Experiment("tabular_combined")
logger = logging.getLogger("evaluating_rewards.analysis.dissimilarity_heatmaps.tabular_combined")


@tabular_combined_ex.config
def default_config():
    log_root = serialize.get_output_dir()  # where results are read from/written to
    experiment_kinds = ()
    config_updates = {}  # config updates applied to all subcommands
    named_configs = {}
    target_reward_type = None
    target_reward_path = None
    pretty_models = {}
    _ = locals()
    del _


@tabular_combined_ex.config
def logging_config(named_configs, log_root):
    """Default logging configuration: hierarchical directory structure based on config."""
    log_dir = os.path.join(  # noqa: F841  pylint:disable=unused-variable
        log_root, "tabular_combined", ":".join(named_configs), imit_util.make_unique_timestamp(),
    )


@tabular_combined_ex.named_config
def point_maze_learned():
    target_reward_type = "evaluating_rewards/PointMazeGroundTruthWithCtrl-v0"
    target_reward_path = "dummy"
    # TODO(adam): avoid absolute paths for replicability
    named_configs = {
        "global": ("point_maze_learned", "high_precision"),
        "epic": {
            "global": ("sample_from_serialized_policy", "dataset_from_serialized_policy", "test")
        },
        "erc": {"global": ("test",)},
    }
    mixed_policy_cfg = {
        "policy_type": "mixture",
        # TODO(adam): should this be from EFS?
        "policy_path": (
            "0.05:random:dummy:ppo2:/mnt/eval_reward/data/transfer_point_maze"
            "/expert/train/policies/final/"
        ),
    }
    expert_policy_cfg = {
        "policy_type": "ppo2",
        # TODO(adam): should this be from EFS?
        "policy_path": "/mnt/eval_reward/data/transfer_point_maze/expert/train/policies/final/",
    }
    experiment_kinds = ("mixture", "random", "expert")
    config_updates = {
        "global": {"data_root": "/mnt/eval_reward/data/"},
        "epic": {
            "mixture": {
                "visitations_factory_kwargs": mixed_policy_cfg,
                "sample_dist_factory_kwargs": mixed_policy_cfg,
            },
            "random": {},
            "expert": {
                "visitations_factory_kwargs": expert_policy_cfg,
                "sample_dist_factory_kwargs": expert_policy_cfg,
            },
        },
        "npec": {
            kind: {
                # TODO(adam): should this also be from EFS?
                "data_subdir": os.path.join("transfer_point_maze", f"comparison_{kind}"),
            }
            for kind in ("random", "expert", "mixture")
        },
        "erc": {
            "mixture": {"trajectory_factory_kwargs": mixed_policy_cfg},
            "random": {},
            "expert": {"trajectory_factory_kwargs": expert_policy_cfg},
        },
    }
    del mixed_policy_cfg
    del expert_policy_cfg
    pretty_models = {
        r"\regressionmethod{}": (
            "evaluating_rewards/RewardModel-v0",
            "transfer_point_maze/reward/regress/model",
        ),
        r"\preferencesmethod{}": (
            "evaluating_rewards/RewardModel-v0",
            "transfer_point_maze/reward/preferences/model",
        ),
        r"\airlstateonlymethod{}": (
            "imitation/RewardNet_unshaped-v0",
            "transfer_point_maze/reward/irl_state_only/checkpoints/final/discrim/reward_net",
        ),
        r"\airlstateactionmethod{}": (
            "imitation/RewardNet_unshaped-v0",
            "transfer_point_maze/reward/irl_state_action/checkpoints/final/discrim/reward_net",
        ),
    }
    _ = locals()
    del _


@tabular_combined_ex.named_config
def high_precision():
    # TODO(adam): how to merge named configs like this?
    named_configs = {  # noqa: F841  pylint:disable=unused-variable
        "global": ("high_precision",),
    }


PRETTY_MODEL_LOOKUP = {}


@tabular_combined_ex.capture
def make_table(
    vals: Mapping[Tuple[str, str], pd.Series], pretty_models: Mapping[str, cli_common.RewardCfg]
) -> str:
    distance_order = ("epic", "npec", "erc")
    visitation_order = ("random", "expert", "mixture")

    y_reward_cfgs = tuple(vals[(distance_order[0], visitation_order[0])].keys())

    rows = []
    for model in y_reward_cfgs:
        cols = []
        kind, path = model
        label = None

        for search_label, (search_kind, search_path) in pretty_models.items():
            if kind == search_kind and path.endswith(search_path):
                label = search_label
        row = f"{label} &"
        for distance, visitation in itertools.product(distance_order, visitation_order):
            col = vals[(distance, visitation)].loc[model]
            col = f"{col * 1000:.4g}"
            cols.append(col)
        row += r"\resultrow{" + "}{".join(cols) + "}"
        rows.append(row)
    rows.append("")
    return " \\\\\n".join(rows)


@tabular_combined_ex.main
def tabular_combined(
    log_dir: str,
    experiment_kinds: Tuple[str],
    config_updates: Mapping[str, Any],
    named_configs: Mapping[str, Any],
    target_reward_type: str,
    target_reward_path: str,
) -> None:
    experiments = {
        "npec": plot_npec_heatmap.plot_npec_heatmap_ex,
        "epic": plot_epic_heatmap.plot_epic_heatmap_ex,
        "erc": plot_erc_heatmap.plot_erc_heatmap_ex,
    }

    runs = {}
    for ex_key, ex in experiments.items():
        for kind in experiment_kinds:
            local_update = dict(config_updates.get("global", {}))
            local_update.update(config_updates.get(ex_key, {}).get("global", {}))
            local_update.update(config_updates.get(ex_key, {}).get(kind, {}))
            local_update["log_dir"] = log_dir

            local_named = tuple(named_configs.get("global", ()))
            local_named += tuple(named_configs.get(ex_key, {}).get("global", ()))
            local_named += tuple(named_configs.get(ex_key, {}).get(kind, ()))

            runs[(ex_key, kind)] = ex.run(
                "compute_vals", config_updates=local_update, named_configs=local_named
            )
    vals = {k: run.result for k, run in runs.items()}

    # TODO(adam): debugging code, remove? Or load from subdirectory vals instead?
    with open("vals.pkl", "wb") as f:
        import pickle

        pickle.dump(vals, f)

    with open("vals.pkl", "rb") as f:
        import pickle

        vals = pickle.load(f)

    # TODO(adam): how to get generator reward? that might be easiest as side-channel.
    # or separate script, which you could potentially combine here.
    # TODO(adam): there are no actual common keys because erc keys differ. I should probably
    # change that? Or let one specify which type of overlap you look for?
    common_keys = set(next(iter(vals.values())).keys())
    for v in vals.values():
        common_keys = common_keys.intersection(v.keys())

    vals_filtered = {}
    for model_key, outer_val in vals.items():
        for table_key, inner_val in outer_val.items():
            vals_filtered.setdefault(table_key, {})[model_key] = inner_val.xs(
                key=(target_reward_type, target_reward_path),
                level=("target_reward_type", "target_reward_path"),
            )

    for k in common_keys:
        v = vals_filtered[k]
        path = os.path.join(log_dir, f"{k}.csv")
        logger.info(f"Writing table to '{path}'")
        with open(path, "wb") as f:
            table = make_table(v)
            f.write(table.encode())


if __name__ == "__main__":
    script_utils.experiment_main(tabular_combined_ex, "tabular_combined")
