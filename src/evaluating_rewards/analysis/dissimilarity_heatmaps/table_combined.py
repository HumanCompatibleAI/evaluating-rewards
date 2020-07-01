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

import copy
import functools
import itertools
import logging
import os
import pickle
from typing import Any, Iterable, Mapping, Optional, Set, Tuple, TypeVar

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

table_combined_ex = sacred.Experiment("table_combined")
logger = logging.getLogger("evaluating_rewards.analysis.dissimilarity_heatmaps.table_combined")


@table_combined_ex.config
def default_config():
    """Default configuration for table_combined."""
    vals_path = None
    log_root = serialize.get_output_dir()  # where results are read from/written to
    experiment_kinds = ()
    config_updates = {}  # config updates applied to all subcommands
    named_configs = {}
    target_reward_type = None
    target_reward_path = None
    pretty_models = {}
    tag = "default"
    _ = locals()
    del _


@table_combined_ex.config
def logging_config(log_root, tag):
    """Default logging configuration: hierarchical directory structure based on config."""
    log_dir = os.path.join(  # noqa: F841  pylint:disable=unused-variable
        log_root, "table_combined", tag, imit_util.make_unique_timestamp(),
    )


@table_combined_ex.named_config
def point_maze_learned():
    """Compare rewards learned in PointMaze to the ground-truth reward."""
    # Analyzes models generated by `runners/transfer_point_maze.sh`.
    # SOMEDAY(adam): this ignores `log_root` and uses `serialize.get_output_dir()`
    # No way to get `log_root` in a named config due to Sacred config limitations.
    target_reward_type = "evaluating_rewards/PointMazeGroundTruthWithCtrl-v0"
    target_reward_path = "dummy"
    named_configs = {
        "point_maze_learned": {
            "global": ("point_maze_learned"),
            "epic": {"global": ("sample_from_serialized_policy", "dataset_from_serialized_policy")},
        }
    }
    mixed_policy_cfg = {
        "policy_type": "mixture",
        "policy_path": (
            f"0.05:random:dummy:ppo2:{serialize.get_output_dir()}/"
            "transfer_point_maze/expert/train/policies/final/"
        ),
    }
    expert_policy_cfg = {
        "policy_type": "ppo2",
        "policy_path": (
            f"{serialize.get_output_dir()}/transfer_point_maze/expert/train/policies/final/"
        ),
    }
    experiment_kinds = ("random", "expert", "mixture")
    config_updates = {
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
            kind: {"data_subdir": os.path.join("transfer_point_maze", f"comparison_{kind}")}
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
    tag = "point_maze_learned"
    _ = locals()
    del _


@table_combined_ex.named_config
def high_precision():
    named_configs = {  # noqa: F841  pylint:disable=unused-variable
        "precision": {"global": ("high_precision",)}
    }


@table_combined_ex.named_config
def test():
    """Simple, quick config for unit testing."""
    experiment_kinds = ("test",)
    target_reward_type = "evaluating_rewards/PointMassGroundTruth-v0"
    target_reward_path = "dummy"
    named_configs = {
        "test": {"global": ("test",)},
        # duplicate to get some coverage of recursive_dict_merge
        "test2": {"global": ("test",)},
    }
    pretty_models = {
        "GT": ("evaluating_rewards/PointMassGroundTruth-v0", "dummy"),
        "Sparse": ("evaluating_rewards/PointMassSparseWithCtrl-v0", "dummy"),
    }
    _ = locals()
    del _


K = TypeVar("K")


def common_keys(vals: Iterable[Mapping[K, Any]]) -> Set[K]:
    res = set(next(iter(vals)).keys())
    for v in vals:
        res = res.intersection(v.keys())
    return res


def make_table(
    key: str,
    vals: Mapping[Tuple[str, str], pd.Series],
    pretty_models: Mapping[str, cli_common.RewardCfg],
    experiment_kinds: Tuple[str],
) -> str:
    """Generate LaTeX table.

    Args:
        key: Key describing where the data comes from. This is used to change formatting options.
        vals: A Mapping from (distance, visitation) to a Series of values.
        pretty_models: A Mapping from short-form ("pretty") labels to reward configurations.
            A model matching that reward configuration is given the associated short label.
    """
    distance_order = ("epic", "npec", "erc")

    y_reward_cfgs = common_keys(vals.values())

    rows = []
    for model in y_reward_cfgs:
        cols = []
        kind, path = model
        label = None

        for search_label, (search_kind, search_path) in pretty_models.items():
            if kind == search_kind and path.endswith(search_path):
                assert label is None
                label = search_label
        assert label is not None
        row = f"{label} & "
        for distance, visitation in itertools.product(distance_order, experiment_kinds):
            col = vals[(distance, visitation)].loc[model]
            multiplier = 100 if key.endswith("relative") else 1000
            col = f"{col * multiplier:.4g}"
            cols.append(col)
        row += r"\resultrow{" + "}{".join(cols) + "}"
        rows.append(row)
    rows.append("")
    return " \\\\\n".join(rows)


def recursive_dict_merge(dest: dict, update_by: dict, path: Optional[Iterable[str]] = None) -> dict:
    """Merges update_by into dest recursively."""
    if path is None:
        path = []
    for key in update_by:
        if key in dest:
            if isinstance(dest[key], dict) and isinstance(update_by[key], dict):
                recursive_dict_merge(dest[key], update_by[key], path + [str(key)])
            elif isinstance(dest[key], (tuple, list)) and isinstance(update_by[key], (tuple, list)):
                dest[key] = tuple(set(dest[key]).union(update_by[key]))
            elif dest[key] == update_by[key]:
                pass  # same leaf value
            else:
                raise Exception("Conflict at {}".format(".".join(path + [str(key)])))
        else:
            dest[key] = update_by[key]
    return dest


@table_combined_ex.main
def table_combined(
    vals_path: Optional[str],
    log_dir: str,
    experiment_kinds: Tuple[str],
    config_updates: Mapping[str, Any],
    named_configs: Mapping[str, Mapping[str, Any]],
    target_reward_type: str,
    target_reward_path: str,
    pretty_models: Mapping[str, cli_common.RewardCfg],
) -> None:
    """Entry-point into CLI script.

    Args:
        vals_path: path to precomputed values to tabulate. Skips everything but table generation
            if specified. This is useful for regenerating tables in a new style from old data.
        log_dir: directory to write figures and other logging to.
        experiment_kinds: different subsets of data to plot, e.g. visitation distributions.
        config_updates: Config updates to apply. Hierarchically specified by algorithm and
            experiment kind. "global" may be specified at top-level (applies to all algorithms)
            or at first-level (applies to particular algorithm, all experiment kinds).
        named_configs: Named configs to apply. First key is a namespace which has no semantic
            meaning, but should be unique for each Sacred config scope. Second key is the algorithm
            scope and third key the experiment kind, like with config_updates. Values at the leaf
            are tuples of named configs. The dicts across namespaces are recursively merged
            using `recursive_dict_merge`.
        target_reward_type: The target reward type to output distance from in the table;
            others are ignored.
        target_reward_path: The target reward path to output distance from in the table;
            others are ignored.
        pretty_models: A Mapping from short-form ("pretty") labels to reward configurations.
            A model matching that reward configuration has the associated short label.
    """
    if vals_path is not None:
        with open(vals_path, "rb") as f:
            vals = pickle.load(f)
    else:
        experiments = {
            "npec": plot_npec_heatmap.plot_npec_heatmap_ex,
            "epic": plot_epic_heatmap.plot_epic_heatmap_ex,
            "erc": plot_erc_heatmap.plot_erc_heatmap_ex,
        }

        # Merge named_configs. We have a faux top-level layer to workaround Sacred being unable to
        # have named configs build on top of each others definitions in a particular order.
        named_configs = [copy.deepcopy(cfg) for cfg in named_configs.values()]
        named_configs = functools.reduce(recursive_dict_merge, named_configs)

        runs = {}
        for ex_key, ex in experiments.items():
            for kind in experiment_kinds:
                local_update = dict(config_updates.get("global", {}))
                local_update.update(config_updates.get(ex_key, {}).get("global", {}))
                local_update.update(config_updates.get(ex_key, {}).get(kind, {}))
                local_update["log_dir"] = os.path.join(log_dir, kind)

                local_named = tuple(named_configs.get("global", ()))
                local_named += tuple(named_configs.get(ex_key, {}).get("global", ()))
                local_named += tuple(named_configs.get(ex_key, {}).get(kind, ()))

                runs[(ex_key, kind)] = ex.run(
                    "compute_vals", config_updates=local_update, named_configs=local_named
                )
        vals = {k: run.result for k, run in runs.items()}

        with open(os.path.join(log_dir, "vals.pkl"), "wb") as f:
            pickle.dump(vals, f)

    # TODO(adam): how to get generator reward? that might be easiest as side-channel.
    # or separate script, which you could potentially combine here.
    vals_filtered = {}
    for model_key, outer_val in vals.items():
        for table_key, inner_val in outer_val.items():
            vals_filtered.setdefault(table_key, {})[model_key] = inner_val.xs(
                key=(target_reward_type, target_reward_path),
                level=("target_reward_type", "target_reward_path"),
            )

    for k in common_keys(vals.values()):
        v = vals_filtered[k]
        path = os.path.join(log_dir, f"{k}.csv")
        logger.info(f"Writing table to '{path}'")
        with open(path, "wb") as f:
            table = make_table(k, v, pretty_models, experiment_kinds)
            f.write(table.encode())


if __name__ == "__main__":
    script_utils.experiment_main(table_combined_ex, "table_combined")
