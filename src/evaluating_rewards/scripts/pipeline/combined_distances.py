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

"""
CLI script to compute EPIC, NPEC and ERC distances and policy returns for a reward model.

Writes raw distances to a pickle file. Also supports summmarizing to a LaTeX table (default),
or a lineplot (only for timeseries over checkpoints).
"""

# pylint:disable=too-many-lines

import copy
import functools
import glob
import itertools
import logging
import os
import pickle
import re
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, TypeVar

from imitation.util import util as imit_util
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import sacred
import seaborn as sns

from evaluating_rewards import serialize
from evaluating_rewards.analysis import results, stylesheets, visualize
from evaluating_rewards.analysis.distances import aggregated
from evaluating_rewards.distances import common_config
from evaluating_rewards.scripts import script_utils
from evaluating_rewards.scripts.distances import epic, erc, npec, rollout_return

Vals = Mapping[Tuple[str, str], Any]
ValsFiltered = Mapping[str, Mapping[Tuple[str, str], pd.Series]]
DistanceFnMapping = Mapping[str, Callable[..., sacred.run.Run]]

combined_distances_ex = sacred.Experiment("combined_distances")
logger = logging.getLogger("evaluating_rewards.scripts.pipeline.combined_distances")

DISTANCE_EXS = {
    "epic": epic.epic_distance_ex,
    "erc": erc.erc_distance_ex,
    "npec": npec.npec_distance_ex,
    "rl": rollout_return.rollout_distance_ex,
}


@combined_distances_ex.config
def default_config():
    """Default configuration for combined_distances."""
    vals_paths = []
    log_root = serialize.get_output_dir()  # where results are read from/written to
    experiment_kinds = {}
    distance_kinds_order = ("epic", "npec", "erc", "rl")
    config_updates = {}  # config updates applied to all subcommands
    named_configs = {}
    skip = {}
    target_reward_type = None
    target_reward_path = None
    pretty_models = {}
    pretty_algorithms = {}
    # Output formats
    output_fn = latex_table
    styles = ["paper", "tex", "training-curve", "training-curve-1col"]
    tag = "default"
    _ = locals()
    del _


@combined_distances_ex.config
def logging_config(log_root, tag):
    """Default logging configuration: hierarchical directory structure based on config."""
    log_dir = os.path.join(  # noqa: F841  pylint:disable=unused-variable
        log_root,
        "combined_distances",
        tag,
        imit_util.make_unique_timestamp(),
    )


POINT_MAZE_LEARNED_COMMON = {
    "target_reward_type": "evaluating_rewards/PointMazeGroundTruthWithCtrl-v0",
    "target_reward_path": "dummy",
    "named_configs": {
        "point_maze_learned": {
            "global": ("point_maze_learned",),
        }
    },
    "pretty_models": {
        r"\groundtruthmethod{}": ("evaluating_rewards/PointMazeGroundTruthWithCtrl-v0", "dummy"),
        r"\bettergoalmethod{}": ("evaluating_rewards/PointMazeBetterGoalWithCtrl-v0", "dummy"),
        r"\regressionmethod{}": (
            "evaluating_rewards/RewardModel-v0",
            r"(.*/)?transfer_point_maze(_fast)?/reward/regress/checkpoints/(final|[0-9]+)",
        ),
        r"\preferencesmethod{}": (
            "evaluating_rewards/RewardModel-v0",
            r"(.*/)?transfer_point_maze(_fast)?/reward/preferences/checkpoints/(final|[0-9]+)",
        ),
        r"\airlstateonlymethod{}": (
            "imitation/RewardNet_unshaped-v0",
            r"(.*/)?transfer_point_maze(_fast)?/reward/irl_state_only/checkpoints/(final|[0-9]+)"
            "/discrim/reward_net",
        ),
        r"\airlstateactionmethod{}": (
            "imitation/RewardNet_unshaped-v0",
            r"(.*/)?transfer_point_maze(_fast)?/reward/irl_state_action/checkpoints/(final|[0-9]+)"
            "/discrim/reward_net",
        ),
    },
}


def _make_visitations_config_updates(method):
    return {
        "epic": {k: {"visitations_factory_kwargs": v} for k, v in method.items()},
        "erc": {k: {"trajectory_factory_kwargs": v} for k, v in method.items()},
        "npec": {k: {"visitations_factory_kwargs": v} for k, v in method.items()},
    }


_POINT_MAZE_EXPERT = (
    f"{serialize.get_output_dir()}/train_experts/ground_truth/20201203_105631_297835/"
    "imitation_PointMazeLeftVel-v0/evaluating_rewards_PointMazeGroundTruthWithCtrl-v0/"
    "best/policies/final/"
)


@combined_distances_ex.named_config
def point_maze_learned_good():
    """Compare rewards learned in PointMaze to the ground-truth reward.

    Use sensible ("good") visitation distributions.
    """
    # Analyzes models generated by `runners/transfer_point_maze.sh`.
    # SOMEDAY(adam): this ignores `log_root` and uses `serialize.get_output_dir()`
    # No way to get `log_root` in a named config due to Sacred config limitations.
    locals().update(**POINT_MAZE_LEARNED_COMMON)
    experiment_kinds = {k: ("random", "expert", "mixture") for k in ("epic", "npec", "erc")}
    experiment_kinds["rl"] = ("train", "test")
    config_updates = _make_visitations_config_updates(
        {
            "random": {
                "policy_type": "random",
                "policy_path": "dummy",
            },
            "expert": {
                "policy_type": "ppo2",
                "policy_path": _POINT_MAZE_EXPERT,
            },
            "mixture": {
                "policy_type": "mixture",
                "policy_path": f"0.05:random:dummy:ppo2:{_POINT_MAZE_EXPERT}",
            },
            "global": {"env_name": "imitation/PointMazeLeftVel-v0"},
        }
    )
    config_updates["rl"] = {
        "train": {"env_name": "imitation/PointMazeLeftVel-v0"},
        "test": {"env_name": "imitation/PointMazeRightVel-v0"},
    }
    tag = "point_maze_learned"
    _ = locals()
    del _


@combined_distances_ex.named_config
def point_maze_learned_pathological():
    """Compare PointMaze rewards under pathological distributions."""
    locals().update(**POINT_MAZE_LEARNED_COMMON)
    distance_kinds_order = ("epic", "npec", "erc")
    experiment_kinds = {
        k: ("random_policy_permuted", "iid", "small", "wrong") for k in distance_kinds_order
    }
    config_updates = _make_visitations_config_updates(
        {
            "random_policy_permuted": {
                "policy_type": "random",
                "policy_path": "dummy",
            },
            "small": {
                "env_name": "evaluating_rewards/PointMaze0.01Left-v0",
                "policy_type": "random",
                "policy_path": "dummy",
            },
            "wrong": {
                "policy_type": "ppo2",
                "policy_path": (
                    f"{serialize.get_output_dir()}/train_experts/point_maze_wrong_target/"
                    "20201122_053216_fb1b0e/imitation_PointMazeLeftVel-v0/"
                    "evaluating_rewards_PointMazeWrongTarget-v0/0/policies/final"
                ),
            },
            "global": {
                "env_name": "imitation/PointMazeLeftVel-v0",
            },
        }
    )
    del config_updates["erc"]["random_policy_permuted"]
    named_configs = POINT_MAZE_LEARNED_COMMON["named_configs"]
    named_configs["point_maze_learned_pathological"] = {
        k: {
            "iid": (
                "sample_from_env_spaces",
                "dataset_iid",
            ),
            "random_policy_permuted": (
                "visitation_config",
                "dataset_permute",
            ),
        }
        for k in ("epic", "npec")
    }
    skip = {
        # ERC does not support these since there are no trajectories (just transitions).
        "erc": ("iid", "random_policy_permuted")
    }
    tag = "point_maze_learned_pathological"
    _ = locals()
    del _


@combined_distances_ex.named_config
def point_maze_checkpoints():
    """Compare rewards learned in PointMaze to the ground-truth reward over time.

    Use sensible ("good") visitation distributions.
    """
    # Analyzes models generated by `runners/transfer_point_maze.sh`.
    # SOMEDAY(adam): this ignores `log_root` and uses `serialize.get_output_dir()`
    # No way to get `log_root` in a named config due to Sacred config limitations.
    locals().update(**POINT_MAZE_LEARNED_COMMON)
    named_configs = {
        "point_maze_learned": {
            "global": ("point_maze_checkpoints_100",),
        }
    }
    experiment_kinds = {k: ("mixture",) for k in ("epic", "npec", "erc")}
    experiment_kinds["rl"] = ("train", "test")
    config_updates = _make_visitations_config_updates(
        {
            "mixture": {
                "env_name": "imitation/PointMazeLeftVel-v0",
                "policy_type": "mixture",
                "policy_path": f"0.05:random:dummy:ppo2:{_POINT_MAZE_EXPERT}",
            },
        }
    )
    config_updates["epic"]["global"] = {
        "num_cpus": 8,  # more computationally expensive with large # of checkpoints
    }
    config_updates["rl"] = {
        "train": {"env_name": "imitation/PointMazeLeftVel-v0"},
        "test": {"env_name": "imitation/PointMazeRightVel-v0"},
    }
    pretty_algorithms = {
        "EPIC": ("epic", "mixture"),
        "NPEC": ("npec", "mixture"),
        "ERC": ("erc", "mixture"),
        "RL Train": ("rl", "train"),
        "RL Test": ("rl", "test"),
    }
    expert_returns = {
        # `monitor_return_mean` from best policies in:
        # data/train_experts/ground_truth/20201203_105631_297835/
        # Note this is higher than the mean (across seeds) which is what we report in paper for GT.
        # Indeed, this is necessary otherwise the regret would sometimes be negative!
        "RL Train": -4.86,  # PointMazeLeftVel-v0
        "RL Test": -4.38,  # PointMazeRightVel-v0
    }
    tag = "point_maze_checkpoints"
    output_fn = distance_over_time
    _ = locals()
    del _


@combined_distances_ex.named_config
def high_precision():
    named_configs = {  # noqa: F841  pylint:disable=unused-variable
        "precision": {
            "global": ("high_precision",),
        }
    }


@combined_distances_ex.named_config
def test():
    """Simple, quick config for unit testing."""
    experiment_kinds = {k: ("test",) for k in ("epic", "npec", "erc", "rl")}
    target_reward_type = "evaluating_rewards/PointMassGroundTruth-v0"
    target_reward_path = "dummy"
    named_configs = {
        "test": {"global": ("test",)},
        # duplicate to get some coverage of recursive_dict_merge
        "test2": {"global": ("test",)},
        "test3": {
            k: {
                "test": ("test",),
            }
            for k in ("epic", "npec", "erc", "rl")
        },
    }
    pretty_models = {
        "GT": ("evaluating_rewards/PointMassGroundTruth-v0", "dummy"),
        "SparseCtrl": ("evaluating_rewards/PointMassSparseWithCtrl-v0", "dummy"),
        "SparseNoCtrl": ("evaluating_rewards/PointMassSparseNoCtrl-v0", "dummy"),
        "DenseCtrl": ("evaluating_rewards/PointMassDenseWithCtrl-v0", "dummy"),
        "DenseNoCtrl": ("evaluating_rewards/PointMassDenseNoCtrl-v0", "dummy"),
    }
    _ = locals()
    del _


@combined_distances_ex.named_config
def fast():
    named_configs = {  # noqa: F841  pylint:disable=unused-variable
        "precision": {"global": ("fast",)}
    }


def _no_op(vals: ValsFiltered) -> None:
    """Do nothing (no-op) output_fn."""
    del vals


@combined_distances_ex.named_config
def suppress_output():
    """Suppress human-readable output. Still saves the raw values.

    Useful to avoid saving unnecessary output when running script multiple times,
    with the intention of combining the raw values using `vals_paths` configuration.
    """
    output_fn = _no_op  # noqa: F841  pylint:disable=unused-variable


K = TypeVar("K")


def common_keys(vals: Iterable[Mapping[K, Any]]) -> Sequence[K]:
    first = next(iter(vals)).keys()
    res = set(first)
    for v in vals:
        res = res.intersection(v.keys())
    return [k for k in first if k in res]  # preserve order


def _fixed_width_format(x: float, figs: int = 3) -> str:
    """Format x as a number targeting `figs+1` characters.

    This is intended for cramming as much information as possible in a fixed-width
    format. If `x >= 10 ** figs`, then we format it as an integer. Note this will
    use more than `figs` characters if `x >= 10 ** (figs+1)`. Otherwise, we format
    it as a float with as many significant figures as we can fit into the space.
    If `x < 10 ** (-figs + 1)`, then we represent it as "<"+str(10 ** (-figs + 1)),
    unless `x == 0` in which case we format `x` as "0" exactly.

    Args:
        x: The number to format. The code assumes this is non-negative; the return
            value may exceed the character target if it is negative.
        figs: The number of digits to target.

    Returns:
        The number formatted as described above.
    """
    smallest_representable = 10 ** (-figs + 1)
    if 0 < x < 10 ** (-figs + 1):
        return "<" + str(smallest_representable)

    raw_repr = str(x).replace(".", "")
    num_leading_zeros = 0
    for digit in raw_repr:
        if digit == "0":
            num_leading_zeros += 1
        else:
            break
    if x >= 10 ** figs:
        # No decimal point gives us an extra character to use
        figs += 1
    fstr = "{:." + str(max(0, figs - num_leading_zeros)) + "g}"
    res = fstr.format(x)

    delta = (figs + 1) - len(res)
    # g drops trailing zeros, add them back
    if delta > 0 and "." in res:
        res += "0" * delta
    if delta > 1 and "." not in res:
        res += "." + "0" * (delta - 1)

    return res


def _pretty_label(
    cfg: common_config.RewardCfg, pretty_mapping: Mapping[str, common_config.RewardCfg]
) -> str:
    """Map `cfg` to a more readable label in `pretty_models`.

    Raises:
        ValueError if `cfg` does not match any label in `pretty_models`.
    """
    kind, path = cfg
    label = None
    for search_label, (search_kind, search_pattern) in pretty_mapping.items():
        if kind == search_kind and re.match(search_pattern, path):
            if label is not None:
                raise ValueError(f"Duplicate match for '{cfg}' in '{pretty_mapping}'")
            label = search_label
    if label is None:
        raise ValueError(f"Did not find '{cfg}' in '{pretty_mapping}'")
    return label


def make_table(
    key: str,
    vals: Mapping[Tuple[str, str], pd.Series],
    pretty_models: Mapping[str, common_config.RewardCfg],
) -> str:
    """Generate LaTeX table.

    Args:
        key: Key describing where the data comes from. This is used to change formatting options.
        vals: A Mapping from (distance, visitation) to a Series of values.
        pretty_models: A Mapping from short-form ("pretty") labels to reward configurations.
            A model matching that reward configuration is given the associated short label.
    """
    y_reward_cfgs = common_keys(vals.values())
    experiment_kinds = _get_sorted_experiment_kinds()  # pylint:disable=no-value-for-parameter

    first_row = ""
    second_row = ""
    for distance, experiments in experiment_kinds.items():
        first_row += r" & & \multicolumn{" + str(len(experiments)) + "}{c}{" + distance + "}"
        second_row += " & & " + " & ".join(experiments)
    rows = [first_row, second_row]

    for model in y_reward_cfgs:
        cols = []
        label = _pretty_label(model, pretty_models)
        row = f"{label} & & "
        for distance, experiments in experiment_kinds.items():
            for visitation in experiments:
                k = (distance, visitation)
                if k in vals:
                    val = vals[k].loc[model]
                    if distance == "rl":
                        multiplier = 1
                    elif key.endswith("relative"):
                        multiplier = 100
                    else:
                        multiplier = 1000
                    val = val * multiplier

                    col = _fixed_width_format(val)  # fit as many SFs as we can into 4 characters
                    try:
                        float(col)
                        # If we're here, then col is numeric
                        col = "\\num{" + col + "}"
                    except ValueError:
                        pass
                else:
                    col = "---"
                cols.append(col)
            cols.append("")  # spacer between distance metric groups
        row += " & ".join(cols[:-1])
        rows.append(row)
    rows.append("")
    return " \\\\\n".join(rows)


@combined_distances_ex.capture
def _get_sorted_experiment_kinds(
    experiment_kinds: Mapping[str, Tuple[str]],
    distance_kinds_order: Optional[Sequence[str]],
) -> Mapping[str, Tuple[str]]:
    """Sorts experiment_kinds` in order of `distance_kinds_order`, if specified.

    Args:
        experiment_kinds: Different subsets of data to tabulate, e.g. visitation distributions.
        distance_kinds_order: The order in which to run and present the distance algorithms,
            which are keys of `experiment_kinds`.

    Returns:
        `experiment_kinds` sorted according to `distance_kinds_order`.
    """
    if not experiment_kinds:
        raise ValueError("Empty `experiment_kinds`.")

    if distance_kinds_order:
        if len(distance_kinds_order) != len(experiment_kinds):
            raise ValueError(
                f"Order '{distance_kinds_order}' is different length"
                f" to keys '{experiment_kinds.keys()}'."
            )

        if set(distance_kinds_order) != set(experiment_kinds.keys()):
            raise ValueError(
                f"Order '{distance_kinds_order}' is different set"
                f" to keys '{experiment_kinds.keys()}'."
            )

        experiment_kinds = {k: experiment_kinds[k] for k in distance_kinds_order}

    return experiment_kinds


@combined_distances_ex.capture
def _input_validation(
    experiment_kinds: Mapping[str, Tuple[str]],
    config_updates: Mapping[str, Any],
    named_configs: Mapping[str, Mapping[str, Any]],
    skip: Mapping[str, Mapping[str, bool]],
) -> None:
    """Validate input.

    See `combined` for args definition."""

    for dist_key, experiments in experiment_kinds.items():
        if dist_key not in DISTANCE_EXS:
            raise ValueError(f"Unrecognized distance '{dist_key}'.")

        for kind in experiments:
            skipped = kind in skip.get(dist_key, ())
            update_local = config_updates.get(dist_key, {}).get(kind, {})
            named_local = named_configs.get(dist_key, {}).get(kind, ())
            configured = update_local or named_local

            if configured and skipped:
                raise ValueError(f"Skipping ({dist_key}, {kind}) that is configured.")
            if not configured and not skipped:
                raise ValueError(f"({dist_key}, {kind}) unconfigured but not skipped.")


def load_vals(vals_paths: Sequence[str]) -> Vals:
    """Loads and combines values from vals_path, recursively searching in subdirectories."""
    pickle_paths = []
    for path in vals_paths:
        if os.path.isdir(path):
            nested_paths = glob.glob(os.path.join(path, "**", "vals.pkl"), recursive=True)
            if not nested_paths:
                raise ValueError(f"No 'vals.pkl' files found in {path}")
            pickle_paths += nested_paths
        else:
            pickle_paths.append(path)

    vals = {}
    for path in pickle_paths:
        with open(path, "rb") as f:
            val = pickle.load(f)
        script_utils.recursive_dict_merge(vals, val)

    return vals


@combined_distances_ex.capture
def compute_vals(
    config_updates: Mapping[str, Any],
    named_configs: Mapping[str, Mapping[str, Any]],
    skip: Mapping[str, Mapping[str, bool]],
    log_dir: str,
) -> Vals:
    """
    Run experiments to compute distance values.

    Args:
        config_updates: Config updates to apply. Hierarchically specified by algorithm and
            experiment kind. "global" may be specified at top-level (applies to all algorithms)
            or at first-level (applies to particular algorithm, all experiment kinds).
        named_configs: Named configs to apply. First key is a namespace which has no semantic
            meaning, but should be unique for each Sacred config scope. Second key is the algorithm
            scope and third key the experiment kind, like with config_updates. Values at the leaf
            are tuples of named configs. The dicts across namespaces are recursively merged
            using `recursive_dict_merge`.
        skip: If `skip[ex_key][kind]` is True, then skip that experiment (e.g. if a metric
            does not support a particular configuration).
        log_dir: The directory to write tables and other logging to.
    """
    experiment_kinds = _get_sorted_experiment_kinds()  # pylint:disable=no-value-for-parameter
    res = {}
    for dist_key, experiments in experiment_kinds.items():
        dist_ex = DISTANCE_EXS[dist_key]
        for kind in experiments:
            if kind in skip.get(dist_key, ()):
                logger.info(f"Skipping ({dist_key}, {kind})")
                continue

            local_updates = [
                config_updates.get("global", {}),
                config_updates.get(dist_key, {}).get("global", {}),
                config_updates.get(dist_key, {}).get(kind, {}),
            ]
            local_updates = [copy.deepcopy(cfg) for cfg in local_updates]
            local_updates = functools.reduce(
                functools.partial(script_utils.recursive_dict_merge, overwrite=True),
                local_updates,
            )

            if "log_dir" in local_updates:
                raise ValueError("Cannot override `log_dir`.")
            local_updates["log_dir"] = os.path.join(log_dir, dist_key, kind)

            local_named = tuple(named_configs.get("global", ()))
            local_named += tuple(named_configs.get(dist_key, {}).get("global", ()))
            local_named += tuple(named_configs.get(dist_key, {}).get(kind, ()))

            logger.info(f"Running ({dist_key}, {kind}): {local_updates} plus {local_named}")
            run = dist_ex.run(config_updates=local_updates, named_configs=local_named)
            res[(dist_key, kind)] = run.result

    return res


def _canonicalize_cfg(cfg: common_config.RewardCfg) -> common_config.RewardCfg:
    kind, path = cfg
    return kind, results.canonicalize_data_root(path)


@combined_distances_ex.capture
def filter_values(
    vals: Vals,
    target_reward_type: str,
    target_reward_path: str,
) -> ValsFiltered:
    """
    Extract values for the target reward from `vals`.

    Args:
        target_reward_type: The target reward type to output distance from in the table;
            others are ignored.
        target_reward_path: The target reward path to output distance from in the table;
            others are ignored.

    Returns:
        The subset of values in `vals` corresponding to the target, converted to pd.Series.
        Nested dictionary. Outer key corresponds to the table kind (e.g. `bootstrap_lower`,
        `studentt_middle`). Inner key corresponds to the comparison kind (e.g. EPIC with
        a particular visitation distribution).

    """
    vals_filtered = {}
    for model_key, outer_val in vals.items():
        for table_key, inner_val in outer_val.items():
            inner_val = {
                (_canonicalize_cfg(target), _canonicalize_cfg(source)): v
                for (target, source), v in inner_val.items()
            }
            inner_val = aggregated.oned_mapping_to_series(inner_val)
            vals_filtered.setdefault(table_key, {})[model_key] = inner_val.xs(
                key=(target_reward_type, target_reward_path),
                level=("target_reward_type", "target_reward_path"),
            )
    return vals_filtered


@combined_distances_ex.capture
def latex_table(
    vals_filtered: ValsFiltered,
    pretty_models: Mapping[str, common_config.RewardCfg],
    log_dir: str,
) -> None:
    """
    Writes tables of data from `vals_filtered`.

    Args:
        vals_filtered: Filtered values returned by `filter_values`.
        pretty_models: A Mapping from short-form ("pretty") labels to reward configurations.
            A model matching that reward configuration has the associated short label.
        log_dir: Directory to write table to.
    """
    for k, v in vals_filtered.items():
        v = vals_filtered[k]
        path = os.path.join(log_dir, f"{k}.csv")
        logger.info(f"Writing table to '{path}'")
        with open(path, "wb") as f:
            table = make_table(k, v, pretty_models)
            f.write(table.encode())


def _checkpoint_to_progress(df: pd.DataFrame) -> pd.DataFrame:
    ckpts = df["Checkpoint"].astype("int")
    if len(ckpts) == 1:
        progress = ckpts * 0.0
    else:
        progress = ckpts * 100 / ckpts.max()

    return progress


def _add_label_and_progress(
    s: pd.Series, pretty_models: Mapping[str, common_config.RewardCfg]
) -> pd.DataFrame:
    """Add pretty label and checkpoint progress to reward distances."""
    labels = s.index.map(functools.partial(_pretty_label, pretty_mapping=pretty_models))
    df = s.reset_index(name="Distance")

    regex = ".*/checkpoints/(?P<Checkpoint>final|[0-9]+)(?:/.*)?$"
    match = df["source_reward_path"].str.extract(regex)
    match["Reward"] = labels

    grp = match.groupby("Reward")
    progress = grp.apply(_checkpoint_to_progress)
    progress = progress.reset_index("Reward", drop=True)
    df["Progress"] = progress
    df["Reward"] = labels

    return df


def _make_cat_type(s: pd.Series, cat_order: Sequence[str]) -> pd.Series:
    """Convert `s` to categorical data type based on `cat_order`.

    Omits any values not present in `s` from the data type.
    """
    labels = set(s)
    label_cats = [cat for cat in cat_order if cat in labels]
    label_cat_type = pd.CategoricalDtype(label_cats)
    return s.astype(label_cat_type)


def _timeseries_distances(
    vals: Mapping[Tuple[str, str], pd.Series],
    pretty_algorithms: Mapping[str, common_config.RewardCfg],
    pretty_models: Mapping[str, common_config.RewardCfg],
) -> pd.DataFrame:
    """Merge vals into a single DataFrame, adding label and progress."""
    vals = {
        _pretty_label(k, pretty_algorithms): _add_label_and_progress(v, pretty_models)
        for k, v in vals.items()
    }
    df = pd.concat(vals, names=("Algorithm", "Original"))
    df = df.reset_index().drop(columns=["Original"])

    # Assign labels and algorithms a categorical data type. This makes sense
    # semantically -- they take on only a small, finite number of data points.
    # Moreover, it helps Seaborn plot things with consistent colours/markers
    # when only a subset are present in a given plot.
    df["Algorithm"] = _make_cat_type(df["Algorithm"], pretty_algorithms.keys())
    df["Reward"] = _make_cat_type(df["Reward"], pretty_models.keys())

    return df


class CustomCILinePlotter(sns.relational._LinePlotter):  # pylint:disable=protected-access
    """
    LinePlotter supporting custom confidence interval width.

    This is unfortunately entangled with seaborn internals so may break with seaborn upgrades.
    """

    def __init__(self, lower, upper, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.estimator = "dummy"

    def aggregate(self, vals, grouper, units=None):
        y_ci = pd.DataFrame(
            {
                "low": self.lower.loc[vals.index, "Distance"],
                "high": self.upper.loc[vals.index, "Distance"],
            }
        )
        return grouper, vals, y_ci


def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def outside_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    legend_padding: float = 0.04,
    legend_height: float = 0.3,
    **kwargs,
) -> None:
    """Plots a legend immediately above the figure axes.

    Args:
        fig: The figure to plot the legend on.
        ax: The axes to plot the legend above.
        legend_padding: Padding between top of axes and bottom of legend, in inches.
        legend_height: Height of legend, in inches.
        **kwargs: Passed through to `fig.legend`.
    """
    _width, height = fig.get_size_inches()
    pos = ax.get_position()
    legend_left = pos.x0
    legend_right = pos.x0 + pos.width
    legend_width = legend_right - legend_left
    legend_bottom = pos.y0 + pos.height + legend_padding / height
    legend_height = legend_height / height
    bbox = (legend_left, legend_bottom, legend_width, legend_height)
    fig.legend(
        loc="lower left",
        bbox_to_anchor=bbox,
        bbox_transform=fig.transFigure,
        mode="expand",
        **kwargs,
    )


# Use different color palettes for each key to make plots more visually distinct.
_COLOR_PALETTES = {
    "Algorithm": "Set2",
    "Reward": "deep",
}


def custom_ci_line_plot(
    mid: pd.DataFrame,
    lower: pd.DataFrame,
    upper: pd.DataFrame,
    hue_col: str,
    style_col: str,
    ax: plt.Axes,
) -> CustomCILinePlotter:
    """Like `sns.lineplot`, but supporting custom confidence intervals.

    Args:
        mid: Data of mid point.
        lower: Data of lower point.
        upper: Data of upper point.
        hue_col: Column to group with (by hue, i.e. line color).
        style_col: Column to group with (by linestyle).
        ax: Axes to plot on.

    Returns:
        The plotter object.
    """
    variables = sns.relational._LinePlotter.get_semantics(  # pylint:disable=protected-access
        dict(x="Progress", y="Distance", hue=hue_col, style=style_col, size=None, units=None),
    )
    plotter = CustomCILinePlotter(
        variables=variables,
        data=mid,
        lower=lower,
        upper=upper,
        legend=None,
        err_style="band",
    )
    palette = sns.color_palette(_COLOR_PALETTES[hue_col], len(mid[hue_col].dtype.categories))
    plotter.map_hue(palette=palette, order=None, norm=None)  # pylint:disable=no-member
    plotter.map_size(sizes=None, order=None, norm=None)  # pylint:disable=no-member
    plotter.map_style(markers=False, dashes=True, order=None)  # pylint:disable=no-member
    plotter._attach(ax)  # pylint:disable=protected-access
    plotter.plot(ax, {})
    return plotter


def _make_distance_over_time_plot_legend(
    plotter: CustomCILinePlotter,
    fig: plt.Figure,
    ax: plt.Axes,
    hue_col: str,
    style_col: str,
) -> None:
    """Add legend to distance over time plot."""
    plotter.add_legend_data(ax)
    handles, labels = ax.get_legend_handles_labels()
    if hue_col == style_col:
        # Only one key, so legend can fit into one row.
        ncol = len(handles)
    else:
        # Different keys. Legend needs two rows.

        # Make number of columns large enough to fit hue and style each in one row.
        n_hue = len(plotter.lower[hue_col].dtype.categories)
        n_style = len(plotter.lower[style_col].dtype.categories)
        ncol = max(n_hue, n_style)

        # Delete subtitles
        del handles[0], labels[0]  # delete hue subtitle
        del handles[n_hue], labels[n_hue]  # delete style subtitle

        # Pad the smaller row, if they're different length, so its entries are centered
        if n_hue > n_style:
            larger_handles, larger_labels = handles[:n_hue], labels[:n_hue]
            smaller_handles, smaller_labels = handles[n_hue:], labels[n_hue:]
        else:
            larger_handles, larger_labels = handles[n_hue:], labels[n_hue:]
            smaller_handles, smaller_labels = handles[:n_hue], labels[:n_hue]

        delta = len(larger_handles) - len(smaller_handles)
        pad_start = delta // 2 + (delta % 2)
        pad_end = delta // 2
        empty = mpatches.Patch(color="white")
        smaller_handles = [empty] * pad_start + smaller_handles + [empty] * pad_end
        smaller_labels = [""] * pad_start + smaller_labels + [""] * pad_end

        # Reassemble. Note that matplotlib fills column by column, so we zip to "transpose".
        handles = list(itertools.chain(*zip(larger_handles, smaller_handles)))
        labels = list(itertools.chain(*zip(larger_labels, smaller_labels)))

    outside_legend(
        handles=handles,
        labels=labels,
        ncol=ncol,
        fig=fig,
        ax=ax,
    )


@combined_distances_ex.capture
def _return_to_regret(df: pd.DataFrame, expert_returns: Mapping[str, float]) -> pd.DataFrame:
    """Subtracts return from expert return to get an (estimate of) policy regret."""
    df = df.copy()
    for algo in df["Algorithm"].dtype.categories:
        if algo.startswith("RL"):
            best = expert_returns[algo]
            mask = df["Algorithm"] == algo
            df.loc[mask, "Distance"] = best - df.loc[mask, "Distance"]
    return df


def _make_distance_over_time_plot(
    mid: pd.DataFrame,
    lower: pd.DataFrame,
    upper: pd.DataFrame,
    filter_col: str,
    filter_val: str,
    group_col: str,
):
    """Plots timeseries of distance, filtering by `df[filter_col] == filter_val`."""
    vals = [mid, lower, upper]
    vals = [_return_to_regret(df) for df in vals]  # pylint:disable=no-value-for-parameter
    hue_col = group_col
    if filter_val == "RL *":
        vals_distance = [pd.DataFrame() for _ in vals]
        style_col = filter_col
    else:
        vals = [df.loc[df[filter_col] == filter_val] for df in vals]
        vals_distance = [df.loc[~df["Algorithm"].str.startswith("RL")] for df in vals]
        style_col = group_col
    vals_rl = [df.loc[df["Algorithm"].str.startswith("RL")] for df in vals]
    if filter_val == "RL *":
        vals_rl = [df.copy() for df in vals_rl]
        for df in vals_rl:
            df["Algorithm"] = _make_cat_type(df["Algorithm"], ["RL Train", "RL Test"])
    mid_dist, lower_dist, upper_dist = vals_distance
    mid_rl, lower_rl, upper_rl = vals_rl

    fig, ax = plt.subplots(1, 1)
    plotter = None
    if not mid_dist.empty:
        plotter = custom_ci_line_plot(mid_dist, lower_dist, upper_dist, hue_col, style_col, ax)
        ax.set_ylabel("Distance")
        _, y_high = ax.get_ylim()
        ax.set_ylim(0, y_high)
    if not mid_rl.empty:
        if mid_dist.empty:
            rl_ax = ax
        else:
            rl_ax = ax.twinx()
        plotter = custom_ci_line_plot(mid_rl, lower_rl, upper_rl, hue_col, style_col, rl_ax)
        rl_ax.set_ylabel("Regret")

    if filter_col == "Reward":
        # Prepend to clarify it is Reward training progress, not the distance Algorithm
        x_label = filter_val
    else:
        x_label = "Reward Model"
    x_label += " Training Progress (%)"
    ax.set_xlabel(x_label)

    _make_distance_over_time_plot_legend(plotter, fig, ax, hue_col, style_col)

    return fig


@combined_distances_ex.capture
def distance_over_time(
    vals_filtered: ValsFiltered,
    pretty_algorithms: Mapping[str, common_config.RewardCfg],
    pretty_models: Mapping[str, common_config.RewardCfg],
    log_dir: str,
    styles: Iterable[str],
    prefix: str = "bootstrap",
) -> None:
    """
    Plots timeseries of distances.

    Only works with certain configs, like `point_maze_checkpoints`.
    """
    _timeseries_distances_curried = functools.partial(
        _timeseries_distances, pretty_algorithms=pretty_algorithms, pretty_models=pretty_models
    )
    lower = _timeseries_distances_curried(vals_filtered[f"{prefix}_lower"])
    mid = _timeseries_distances_curried(vals_filtered[f"{prefix}_middle"])
    upper = _timeseries_distances_curried(vals_filtered[f"{prefix}_upper"])

    algo_categories = list(mid["Algorithm"].dtype.categories) + ["RL *"]
    for algorithm in algo_categories:
        custom_styles = []
        if algorithm == "RL *":
            custom_styles = [style + "-tall-legend" for style in styles]
            custom_styles = [style for style in custom_styles if style in stylesheets.STYLES]

        with stylesheets.setup_styles(list(styles) + custom_styles):
            fig = _make_distance_over_time_plot(mid, lower, upper, "Algorithm", algorithm, "Reward")
            visualize.save_fig(os.path.join(log_dir, "timeseries", f"algorithm_{algorithm}"), fig)

    for reward in mid["Reward"].dtype.categories:
        with stylesheets.setup_styles(styles):
            fig = _make_distance_over_time_plot(mid, lower, upper, "Reward", reward, "Algorithm")
            visualize.save_fig(os.path.join(log_dir, "timeseries", f"reward_{reward}"), fig)


@combined_distances_ex.main
def combined_distances(
    vals_paths: Sequence[str],
    log_dir: str,
    named_configs: Mapping[str, Mapping[str, Any]],
    output_fn: Callable[[ValsFiltered], None],
) -> None:
    """Entry-point into CLI script.

    Args:
        vals_paths: Paths to precomputed values to tabulate. Skips everything but table generation
            if non-empty. This is useful for regenerating tables in a new style from old data,
            including combining results from multiple previous runs.
        log_dir: The directory to write tables and other logging to.
        named_configs: Named configs to apply. First key is a namespace which has no semantic
            meaning, but should be unique for each Sacred config scope. Second key is the algorithm
            scope and third key the experiment kind, like with config_updates. Values at the leaf
            are tuples of named configs. The dicts across namespaces are recursively merged
            using `recursive_dict_merge`.
        output_fn: Function to call to generate saved output.
    """
    # Merge named_configs. We have a faux top-level layer to workaround Sacred being unable to
    # have named configs build on top of each others definitions in a particular order.
    named_configs = [copy.deepcopy(cfg) for cfg in named_configs.values()]
    named_configs = functools.reduce(script_utils.recursive_dict_merge, named_configs)

    _input_validation(named_configs=named_configs)  # pylint:disable=no-value-for-parameter

    if vals_paths:
        vals = load_vals(vals_paths)
    else:
        vals = compute_vals(named_configs=named_configs)  # pylint:disable=no-value-for-parameter

        with open(os.path.join(log_dir, "vals.pkl"), "wb") as f:
            pickle.dump(vals, f)

    # TODO(adam): how to get generator reward? that might be easiest as side-channel.
    # or separate script, which you could potentially combine here.
    vals_filtered = filter_values(vals)  # pylint:disable=no-value-for-parameter

    output_fn(vals_filtered)


if __name__ == "__main__":
    script_utils.experiment_main(combined_distances_ex, "combined_distances")
