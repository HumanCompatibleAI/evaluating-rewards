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
import functools
import json
import os
import pickle
import re
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

import gym
from imitation import util as imitation_util
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines.common import vec_env

from evaluating_rewards import serialize
from evaluating_rewards.experiments import visualize

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
    stats: ConfigStatsMapping, preprocess: Tuple[PreprocessFn] = (), figsize=(12, 16)
):
    """Extract losses from stats and visualize in a heatmap."""
    loss = {cfg: average_loss(d) for cfg, d in stats.items()}
    unwrapped_loss = {cfg: average_unwrapped_loss(d) for cfg, d in stats.items()}
    for pre in (to_series,) + preprocess:
        loss = pre(loss)
        unwrapped_loss = pre(unwrapped_loss)

    fig, axs = plt.subplots(2, 1, figsize=figsize, squeeze=True)

    visualize.comparison_heatmap(loss, ax=axs[0])
    axs[0].set_title("Matched Loss")

    visualize.comparison_heatmap(unwrapped_loss, ax=axs[1])
    axs[1].set_title("Original Loss")

    return {"fig": fig, "loss": loss, "unwrapped_loss": unwrapped_loss}


def get_metric(stats: ConfigStatsMapping, key: str, idx: int = -1):
    """Extract affine parameters from training statistics, at epoch idx."""
    return {k: v["metrics"][idx]["singleton"][key] for k, v in stats.items()}


def get_affine_from_models(env_name: str, paths: Iterable[str]):
    """Extract affine parameters from reward model."""
    venv = vec_env.DummyVecEnv([lambda: gym.make(env_name)])
    res = {}
    with imitation_util.make_session():
        for path in paths:
            model = serialize.load_reward(
                "evaluating_rewards/RewardModel-v0", os.path.join(path, "model"), venv
            )
            return model.models["wrapped"][0].get_weights()
    return res


def affine_pipeline(
    stats: ConfigStatsMapping, preprocess: Tuple[PreprocessFn] = (), figsize=(12, 16)
):
    """Extract final affine parameters from stats and visualize in a heatmap."""
    constants = get_metric(stats, "constant")
    scales = get_metric(stats, "scale")
    for pre in (to_series,) + preprocess:
        constants = pre(constants)
        scales = pre(scales)

    fig, axs = plt.subplots(2, 1, figsize=figsize, squeeze=True)

    visualize.comparison_heatmap(scales, ax=axs[0], robust=True)
    axs[0].set_title("Scale")

    visualize.comparison_heatmap(constants, log=False, center=0.0)
    axs[1].set_title("Constant")

    return {"fig": fig, "constants": constants, "scales": scales}


def median_seeds(series: pd.Series) -> pd.Series:
    """Take the median over any seeds in a series.."""
    seeds = [name for name in series.index.names if "seed" in name]
    assert seeds, "No seed levels found"
    non_seeds = [name for name in series.index.names if name not in seeds]
    return series.groupby(non_seeds).median()


def compact(series: pd.Series) -> pd.Series:
    """Make series smaller, suitable for e.g. small figures."""
    series = median_seeds(series)
    if "target_reward_type" in series.index.names:
        targets = series.index.get_level_values("target_reward_type")
        series = series.loc[targets != "evaluating_rewards/Zero-v0"]
    return series


def same(args: Iterable[Any]) -> bool:
    """All values in args are the same."""
    some_arg = next(iter(args))
    return all(arg == some_arg for arg in args)


def replace(pattern: str, replacement: str) -> Callable[[Iterable[str]], Iterable[str]]:
    """Iterable version version of re.sub."""
    pattern = re.compile(pattern)

    def helper(args: Iterable[str]) -> Iterable[str]:
        return tuple(re.sub(pattern, replacement, x) for x in args)

    return helper


def match(pattern: str) -> Callable[[Iterable[str]], Iterable[bool]]:
    """Iterable version of re.match."""
    pattern = re.compile(pattern)

    def helper(args: Iterable[str]) -> Iterable[bool]:
        return tuple(bool(pattern.match(x)) for x in args)

    return helper


def zero(args: Iterable[str]) -> bool:
    """Are any of the arguments the Zero reward function."""
    return any(arg == "evaluating_rewards/Zero-v0" for arg in args)


def control(args: Iterable[str]) -> bool:
    """Do args differ only in terms of with/without control cost?"""
    args = replace(r"(.*?)(NoCtrl|WithCtrl|)-(.*)", r"\1-\3")(args)
    return same(args)


def sparse_or_dense(args: Iterable[str]) -> bool:
    """Args are both sparse or both dense?"""
    args = replace(r"(.*)(Sparse|Dense)(.*)", r"\1\2")(args)
    return same(args)


def direction(args: Iterable[str]) -> bool:
    """Do args differ only in terms of forward/backward and control cost?"""
    pattern = r"evaluating_rewards/(.*?)(Backward|Forward)(WithCtrl|NoCtrl)(.*)"
    args = replace(pattern, r"\1\4")(args)
    return same(args)


def no_ctrl(args: Iterable[str]) -> bool:
    """Are args all without control cost?"""
    return all("NoCtrl" in arg for arg in args)


def always_true(args: Iterable[str]) -> bool:
    """Constant true."""
    del args
    return True


def mask(
    series: pd.Series,
    matchings: Iterable[FilterFn],
    levels: Iterable[str] = ("source_reward_type", "target_reward_type"),
) -> pd.Series:
    """Masks values in `series` not matching any of `matchings`.

    Args:
        series: Input data.
        matchings: A sequence of callables.
        levels: A sequence of levels to match.

    Returns:
        A boolean Series, with an index equal to `series`.
        The value is true iff one of `matchings` returns true.
    """
    xs = []
    for level in levels:
        xs.append(set(series.index.get_level_values(level)))

    index = series.index
    for level in index.names:
        if level not in levels:
            index = index.droplevel(level=level)

    res = pd.Series(False, index=index)
    for matching in matchings:
        res |= index.map(matching)
    res = ~res
    res.index = series.index

    return res


def pipeline(stats: ConfigStatsMapping, **kwargs):
    """Run loss and affine pipeline on stats."""
    if not stats:
        raise ValueError("'stats' is empty.")

    return {"loss": loss_pipeline(stats, **kwargs), "affine": affine_pipeline(stats, **kwargs)}


short_fmt = functools.partial(visualize.short_e, precision=0)


def compact_heatmaps(
    loss: pd.Series,
    order: Iterable[str],
    masks: Mapping[str, Iterable[FilterFn]],
    fmt: Callable[[float], str] = short_fmt,
    after_plot: Callable[[], None] = lambda: None,
) -> Mapping[str, plt.Figure]:
    """Plots a series of compact heatmaps, suitable for presentations.

    Args:
        loss: The loss between source and target.
                The index should consist of target_reward_type, one of
                source_reward_{type,path}, and any number of seed indices.
                source_reward_path, if present, is rewritten into source_reward_type
                and a seed index.
        order: The order to plot the source and reward types.
        masks: A mapping from strings to collections of filter functions. Any
                (source, reward) pair not matching one of these filters is masked
                from the figure.
        fmt: A Callable mapping losses to strings to annotate cells in heatmap.
        after_plot: Called after plotting, for environment-specific tweaks.

    Returns:
        A mapping from strings to figures.
    """
    loss = loss.copy()
    loss = visualize.rewrite_index(loss)
    loss = compact(loss)

    source_order = list(order)
    zero_type = "evaluating_rewards/Zero-v0"
    if zero_type in loss.index.get_level_values("source_reward_type"):
        if zero_type not in source_order:
            source_order.append(zero_type)
    loss = loss.reindex(index=source_order, level="source_reward_type")
    loss = loss.reindex(index=order, level="target_reward_type")

    figs = {}
    for name, matching in masks.items():
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), squeeze=True)
        match_mask = mask(loss, matching)
        visualize.comparison_heatmap(loss, fmt=fmt, preserve_order=True, mask=match_mask, ax=ax)
        # make room for multi-line xlabels
        after_plot()
        figs[name] = fig

    return figs
