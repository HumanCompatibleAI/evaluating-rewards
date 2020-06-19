# Copyright 2019 DeepMind Technologies Limited
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

"""Helper methods to load and analyse results from experiments."""

import collections
import json
import os
import pickle
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

import gym
from imitation.util import networks
import numpy as np
import pandas as pd
from stable_baselines.common import vec_env

from evaluating_rewards import serialize

Config = Tuple[Any, ...]
Stats = Mapping[str, Any]
ConfigStatsMapping = Mapping[Config, Stats]
DirStatsMapping = Mapping[str, Stats]
DirConfigMapping = Mapping[str, Config]
FilterFn = Callable[[Iterable[str]], bool]
PreprocessFn = Callable[[pd.Series], pd.Series]


def find_sacred_results(root_dir: str) -> DirStatsMapping:
    """Find result directories in root_dir, loading the associated config.

    Finds all directories in `root_dir` that contains a subdirectory "sacred".
    For each such directory, the config in "sacred/config.json" is loaded.

    Args:
        root_dir: A path to recursively search in.

    Returns:
        A dictionary of directory paths to Sacred configs.

    Raises:
        ValueError: if a results directory contains another results directory.
        FileNotFoundError: if no results directories found.
    """
    results = set()
    for root, dirs, _ in os.walk(root_dir):
        if "sacred" in dirs:
            results.add(root)

    if not results:
        raise FileNotFoundError(f"No Sacred results directories in '{root_dir}'.")

    # Sanity check: not expecting nested experiments
    for result in results:
        components = os.path.split(result)
        for i in range(1, len(components)):
            prefix = os.path.join(*components[0:i])
            if prefix in results:
                raise ValueError(f"Parent {prefix} to {result} also a result directory")

    configs = {}
    for result in results:
        config_path = os.path.join(result, "sacred", "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        configs[result] = config

    return configs


def dict_to_tuple(d, keys: Optional[Iterable[str]] = None):
    """Recursively convert dict's to namedtuple's, leaving other values intact."""
    if isinstance(d, dict):
        if keys is None:
            keys = sorted(d.keys())
        key_tuple_cls = collections.namedtuple("KeyTuple", keys)
        return key_tuple_cls(**{k: dict_to_tuple(d[k]) for k in keys})
    else:
        return d


def subset_keys(configs: DirStatsMapping, keys: Iterable[str]) -> DirConfigMapping:
    """Extracts the subset of `keys` from each config in `configs`.

    Args:
        configs: Paths mapping to full Sacred configs, as returned by
                `find_sacred_results`.
        keys: The subset of keys to retain from the config.

    Returns:
        A mapping from paths to tuples of the keys.

    Raises:
        ValueError: If any of the config subsets are duplicates of each other.
    """
    res = {}
    configs_seen = set()

    for path, config in configs.items():
        subset = dict_to_tuple(config, keys)  # type: Tuple[Any, ...]
        if subset in configs_seen:
            raise ValueError(f"Duplicate config '{subset}'")
        configs_seen.add(subset)
        res[path] = subset
    return res


def load_stats(dirname: str) -> Stats:
    """Load training statistics.

    Works on output from train_preferences, train_regress and model_comparison;
    return format may differ between them though.

    Args:
        dirname: The results directory to load the statistics from.

    Returns:
        The statistics.
    """
    path = os.path.join(dirname, "stats.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_multiple_stats(
    root_dir: str,
    keys: Iterable[str],
    cfg_filter: Callable[[Mapping[str, Any]], bool] = lambda cfg: True,
) -> ConfigStatsMapping:
    """Load all training statistics in root_dir."""
    configs = find_sacred_results(root_dir)
    configs = {path: _canonicalize_cfg_path(cfg) for path, cfg in configs.items()}
    subset = {path: cfg for path, cfg in configs.items() if cfg_filter(cfg)}
    subset = subset_keys(subset, keys)

    stats = {}
    for dirname, cfg in subset.items():
        try:
            stats[cfg] = load_stats(dirname)
        except FileNotFoundError:
            print(f"Skipping {cfg}, no stats in '{dirname}'")

    return stats


def to_series(x: pd.Series) -> pd.Series:
    s = pd.Series(x)
    some_key = next(iter(x.keys()))
    s.index.names = some_key._fields
    return s


def average_loss(stats: Stats, n: int = 10) -> float:
    """Compute average loss of last n data points in training."""
    loss = pd.DataFrame(stats["loss"])["singleton"]
    return loss.iloc[-n:].mean()


def average_unwrapped_loss(stats: Stats) -> float:
    """Compute average "unwrapped" loss (original model vs target)."""
    metrics = stats["metrics"]
    unwrapped_loss = [v["singleton"]["unwrapped_loss"] for v in metrics]
    # Does not change during training, so can take mean over entire array
    return np.mean(unwrapped_loss)


def loss_pipeline(
    stats: ConfigStatsMapping, preprocess: Tuple[PreprocessFn] = (),
):
    """Extract losses from stats and visualize in a heatmap."""
    loss = {cfg: average_loss(d) for cfg, d in stats.items()}
    unwrapped_loss = {cfg: average_unwrapped_loss(d) for cfg, d in stats.items()}
    for pre in (to_series,) + preprocess:
        loss = pre(loss)
        unwrapped_loss = pre(unwrapped_loss)

    return {"loss": loss, "unwrapped_loss": unwrapped_loss}


def get_metric(stats: ConfigStatsMapping, key: str, idx: int = -1):
    """Extract affine parameters from training statistics, at epoch idx."""
    return {k: v["metrics"][idx]["singleton"][key] for k, v in stats.items()}


def get_affine_from_models(env_name: str, paths: Iterable[str]):
    """Extract affine parameters from reward model."""
    venv = vec_env.DummyVecEnv([lambda: gym.make(env_name)])
    res = {}
    with networks.make_session():
        for path in paths:
            model = serialize.load_reward(
                "evaluating_rewards/RewardModel-v0", os.path.join(path, "model"), venv,
            )
            return model.models["wrapped"][0].get_weights()
    return res


def affine_pipeline(
    stats: ConfigStatsMapping, preprocess: Tuple[PreprocessFn] = (),
):
    """Extract final affine parameters from stats and visualize in a heatmap."""
    constants = get_metric(stats, "constant")
    scales = get_metric(stats, "scale")
    for pre in (to_series,) + preprocess:
        constants = pre(constants)
        scales = pre(scales)

    return {"constants": constants, "scales": scales}


def pipeline(stats: ConfigStatsMapping, **kwargs):
    """Run loss and affine pipeline on stats."""
    if not stats:
        raise ValueError("'stats' is empty.")

    return {"loss": loss_pipeline(stats, **kwargs), "affine": affine_pipeline(stats, **kwargs)}


# TODO(adam): backwards compatibility -- remove once rerun experiments
DATA_ROOT_PREFIXES = [
    # Older versions of the code stored absolute paths in config.
    # Try and turn these into relative paths for portability.
    "/root/output",
    "/mnt/eval_reward/data",
    "/mnt/eval_reward_efs/data",
]


def _canonicalize_data_root(path: str) -> str:
    if path.endswith("dummy"):
        path = "dummy"
    for root_prefix in DATA_ROOT_PREFIXES:
        if path.startswith(root_prefix):
            path = path.replace(root_prefix, serialize.get_output_dir())
            break
    return path


def _canonicalize_cfg_path(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(cfg)
    for fld in ("source_reward_path", "target_reward_path"):
        if fld in cfg:
            cfg[fld] = _canonicalize_data_root(cfg[fld])
    return cfg


def _find_sacred_parent(
    path: str, seen: Dict[str, str]
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Finds first Sacred directory that is in path or a parent.

    Args:
        path: Path to a directory to start searching from.
        seen: A dictionary from parent paths to children.

    Returns:
        A tuple of the config found and the parent path it is located at.
        As a side-effect, adds path to seen.

    Raises:
        ValueError: if the parent path was already in seen for a different child.
        ValueError: no parent path containing a Sacred directory exists.
    """
    parent = path
    while parent and not os.path.exists(os.path.join(parent, "sacred", "config.json")):
        parent = os.path.dirname(parent)
        if parent == "/":
            parent = ""
    if not parent:
        raise ValueError(f"No parent of '{path}' contains a Sacred directory.")

    if parent in seen and seen[parent] != path:
        raise ValueError(
            f"index contains two paths '{path}' and '{seen[parent]}' "
            f"with common Sacred parent 'f{parent}'."
        )
    seen[parent] = path

    config_path = os.path.join(parent, "sacred", "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    run_path = os.path.join(parent, "sacred", "run.json")
    with open(run_path, "r") as f:
        run = json.load(f)

    return config, run, parent


HARDCODED_TYPES = ["evaluating_rewards/Zero-v0"]


def path_to_config(kinds: Iterable[str], paths: Iterable[str]) -> pd.DataFrame:
    """Extracts relevant config parameters from paths in index.

    Args:
        kinds: An index of reward types.
        paths: An index of paths.

    Returns:
        A MultiIndex consisting of original reward type and seed(s).
    """
    seen = {}
    res = []
    for (kind, path) in zip(kinds, paths):
        if kind in HARDCODED_TYPES or path == "dummy":
            res.append((kind, "hardcoded", 0, 0))
        else:
            path = _canonicalize_data_root(path)
            config, run, path = _find_sacred_parent(path, seen)
            if "target_reward_type" in config:
                # Learning directly from a reward: e.g. train_{regress,preferences}
                pretty_type = {"train_regress": "regress", "train_preferences": "preferences"}
                model_type = pretty_type[run["command"]]
                res.append((config["target_reward_type"], model_type, config["seed"], 0))
            elif "rollout_path" in config:
                # Learning from demos: e.g. train_adversarial
                config["rollout_path"] = _canonicalize_data_root(config["rollout_path"])
                rollout_config, _, _ = _find_sacred_parent(config["rollout_path"], seen)
                reward_type = rollout_config["reward_type"] or "EnvReward"
                reward_args = config["init_trainer_kwargs"]["reward_kwargs"]
                state_only = reward_args.get("state_only", False)
                model_type = "IRL" + ("-SO" if state_only else "-SA")
                res.append((reward_type, model_type, config["seed"], rollout_config["seed"]))
            else:
                raise ValueError(
                    f"Unexpected config at '{path}': does not contain "
                    "'source_reward_type' or 'rollout_path'"
                )

    names = ["source_reward_type", "model_type", "model_seed", "data_seed"]
    return pd.DataFrame(res, columns=names)
