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

"""Methods to generate plots and visualize data."""

import functools
import logging
import math
import os
import re
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evaluating_rewards.analysis import results

TRANSFORMATIONS = {
    r"^evaluating_rewards[_/](.*)-v0": r"\1",
    r"^imitation[_/](.*)-v0": r"\1",
    "^Zero": r"\\zeroreward{}",
    "^PointMassDense": r"\\dense{}",
    "^PointMassGroundTruth": r"\\magnitude{}\\controlpenalty{}",
    "^PointMassSparse": r"\\sparse{}",
    "^PointMazeGroundTruth": "GT",
    r"(.*)(Hopper|HalfCheetah)GroundTruth(.*)": r"\1\2\\running{}\3",
    r"(.*)(Hopper|HalfCheetah)Backflip(.*)": r"\1\2\\backflipping{}\3",
    r"^Hopper(.*)": r"\1",
    r"^HalfCheetah(.*)": r"\1",
    r"^(.*)Backward(.*)": r"\\backward{\1\2}",
    r"^(.*)Forward(.*)": r"\\forward{\1\2}",
    r"^(.*)WithCtrl(.*)": r"\1\\controlpenalty{}\2",
    r"^(.*)NoCtrl(.*)": r"\1\\nocontrolpenalty{}\2",
    "sparse_goal": r"\\sparsegoal{}",
    "transformed_goal": r"\\densegoal{}",
    "center_goal": r"\\centergoal{}",
    "sparse_penalty": r"\\sparsepenalty{}",
    "all_zero": r"\\zeroreward{}",
    "dirt_path": r"\\dirtpath{}",
    "cliff_walk": r"\\cliffwalk{}",
}


LEVEL_NAMES = {
    # Won't normally use both type and path in one plot
    "source_reward_type": "Source",
    "source_reward_path": "Source",
    "target_reward_type": "Target",
    "target_reward_path": "Target",
}

WHITELISTED_LEVELS = ["source_reward_type", "target_reward_type"]  # never remove these levels


# Saving figures


def save_fig(path: str, fig: plt.Figure, fmt: str = "pdf", dpi: int = 300, **kwargs):
    path = f"{path}.{fmt}"
    root_dir = os.path.dirname(path)
    os.makedirs(root_dir, exist_ok=True)
    logging.info(f"Saving figure to {path}")
    kwargs.setdefault("transparent", True)
    with open(path, "wb") as f:
        fig.savefig(f, format=fmt, dpi=dpi, **kwargs)


def save_figs(root_dir: str, generator: Iterable[Tuple[str, plt.Figure]], **kwargs) -> None:
    for name, fig in generator:
        name = name.replace("/", "_")
        path = os.path.join(root_dir, name)
        save_fig(path, fig, **kwargs)


# Rewrite: manipulating indices for readability


def pretty_rewrite(x):
    if not isinstance(x, str):
        return x

    for pattern, repl in TRANSFORMATIONS.items():
        x = re.sub(pattern, repl, x)
    return x


def rewrite_index(series: pd.Series) -> pd.Series:
    """Replace `source_reward_path` with info extracted from config at that path."""
    if "source_reward_path" in series.index.names:
        new_index = series.index.to_frame(index=False)
        source_reward = results.path_to_config(
            new_index["source_reward_type"], new_index["source_reward_path"]
        )
        new_index = new_index.drop(columns=["source_reward_type", "source_reward_path"])
        new_index = pd.concat([source_reward, new_index], axis=1)
        new_index = pd.MultiIndex.from_frame(new_index)
        series.index = new_index
    return series


def short_e(x: float, precision: int = 2) -> str:
    """Formats 1.2345 as 1.2e-1, rather than Python 1.2e-01."""
    if not math.isfinite(x):
        return str(x)
    fmt = "{:." + str(precision) + "e}"
    formatted = fmt.format(x)
    base, exponent = formatted.split("e")
    exponent = int(exponent)
    return f"{base}e{exponent}"


def remove_constant_levels(index: pd.MultiIndex) -> pd.MultiIndex:
    index = index.copy()
    levels = index.names
    for level in levels:
        if len(index.get_level_values(level).unique()) == 1 and level not in WHITELISTED_LEVELS:
            index = index.droplevel(level=level)
    return index


# Plotting: generate figures


def plot_shaping_comparison(
    df: pd.DataFrame, cols: Optional[Iterable[str]] = None, **kwargs
) -> pd.DataFrame:
    """Plots return value of experiments.compare_synthetic."""
    if cols is None:
        cols = ["Intrinsic", "Shaping"]
    df = df.loc[:, cols]
    longform = df.reset_index()
    longform = pd.melt(
        longform,
        id_vars=["Reward Noise", "Potential Noise"],
        var_name="Metric",
        value_name="Distance",
    )
    sns.lineplot(
        x="Reward Noise",
        y="Distance",
        hue="Potential Noise",
        style="Metric",
        data=longform,
        **kwargs,
    )
    return longform


def _heatmap_reformat(series, preserve_order):
    """Helper to reformat labels for ease of interpretability."""
    series = series.copy()
    series = rewrite_index(series)
    series.index = remove_constant_levels(series.index)
    series.index.names = [LEVEL_NAMES.get(name, name) for name in series.index.names]
    series = series.rename(index=pretty_rewrite)

    # Preserve order of inputs
    df = series.unstack("Target")
    if preserve_order:
        df = df.reindex(columns=series.index.get_level_values("Target").unique())
        for level in series.index.names:
            kwargs = {}
            if isinstance(df.index, pd.MultiIndex):
                kwargs = dict(level=level)
            if level != "Target":
                df = df.reindex(index=series.index.get_level_values(level).unique(), **kwargs)
    else:
        df = df.sort_index()
    return df


def comparison_heatmap(
    vals: pd.Series,
    ax: plt.Axes,
    log: bool = True,
    fmt: Callable[[float], str] = short_e,
    cbar_kws: Optional[Dict[str, Any]] = None,
    cmap: str = "GnBu",
    robust: bool = False,
    preserve_order: bool = False,
    mask: Optional[pd.Series] = None,
    **kwargs,
) -> None:
    """Plot a heatmap, with target_reward_type as x-axis and remainder as y-axis.

    This is intended for plotting the output of `model_comparison.py` runs,
    comparing models on the y-axis to those on the x-axis. Values visualized may
    include loss, other distance measures and scale transformations.

    Args:
        vals: The values to visualize.
        log: log-10 scale for the values if True.
        fmt: format string for annotated values.
        cmap: color map.
        robust: If true, set vmin and vmax to 25th and 75th quantiles.
                This makes the color scale robust to outliers, but will compress it
                more than is desirable if the data does not contain outliers.
        preserve_order: If true, retains the same order as the input index
                after rewriting the index values for readability. If false,
                sorts the rewritten values alphabetically.
        mask: If provided, only display cells where mask is True.
        **kwargs: passed through to sns.heatmap.
    """
    vals = _heatmap_reformat(vals, preserve_order)
    if mask is not None:
        mask = _heatmap_reformat(mask, preserve_order)

    data = np.log10(vals) if log else vals
    cbar_kws = dict(cbar_kws or {})
    if log:
        cbar_kws.setdefault("label", r"$-\log_{10}(q)$")

    annot = vals.applymap(fmt)

    if robust:
        flat = data.values.flatten()
        kwargs["vmin"], kwargs["vmax"] = np.quantile(flat, [0.25, 0.75])
    sns.heatmap(
        data, annot=annot, fmt="s", cmap=cmap, cbar_kws=cbar_kws, mask=mask, ax=ax, **kwargs
    )

    ax.set_xlabel(r"Target $R_T$")
    ax.set_ylabel(r"Source $R_S$")


def median_seeds(series: pd.Series) -> pd.Series:
    """Take the median over any seeds in a series."""
    seeds = [name for name in series.index.names if "seed" in name]
    if seeds:
        non_seeds = [name for name in series.index.names if name not in seeds]
        series = series.groupby(non_seeds).median()
    return series


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


def compute_mask(
    series: pd.Series,
    matchings: Iterable[results.FilterFn],
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


short_fmt = functools.partial(short_e, precision=0)


def compact_heatmaps(
    loss: pd.Series,
    masks: Mapping[str, Iterable[results.FilterFn]],
    order: Optional[Iterable[str]] = None,
    fmt: Callable[[float], str] = short_fmt,
    after_plot: Callable[[], None] = lambda: None,
    **kwargs: Dict[str, Any],
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
        kwargs: passed through to `comparison_heatmap`.

    Returns:
        A mapping from strings to figures.
    """
    if order is None:
        order = loss.index.levels[0]
    loss = loss.copy()
    loss = rewrite_index(loss)
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
        fig, ax = plt.subplots(1, 1, squeeze=True)
        match_mask = compute_mask(loss, matching)
        comparison_heatmap(loss, ax=ax, fmt=fmt, preserve_order=True, mask=match_mask, **kwargs)
        # make room for multi-line xlabels
        after_plot()
        figs[name] = fig

    return figs


def _multi_heatmap(
    data: Iterable[pd.Series], labels: Iterable[pd.Series], kwargs: Iterable[Dict[str, Any]]
) -> plt.Figure:
    data = tuple(data)
    labels = tuple(labels)
    kwargs = tuple(kwargs)
    ncols = len(data)
    assert ncols == len(labels)
    assert ncols == len(kwargs)

    width, height = plt.rcParams.get("figure.figsize")
    fig, axs = plt.subplots(ncols, 1, figsize=(ncols * width, height), squeeze=True)

    for series, lab, kw, ax in zip(data, labels, kwargs, axs):
        comparison_heatmap(series, ax=ax, **kw)
        ax.set_title(lab)

    return fig


def loss_heatmap(loss: pd.Series, unwrapped_loss: pd.Series) -> plt.Figure:
    return _multi_heatmap([loss, unwrapped_loss], ["Loss", "Unwrapped Loss"], [{}, {}])


def affine_heatmap(scales: pd.Series, constants: pd.Series) -> plt.Figure:
    return _multi_heatmap(
        [scales, constants], ["Scale", "Constant"], [dict(robust=True), dict(log=False, center=0.0)]
    )
