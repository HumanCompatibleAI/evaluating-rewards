# Copyright 2019, 2020 DeepMind Technologies Limited, Adam Gleave
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

"""Helper methods for plotting dissimilarity heatmaps."""

import functools
import math
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evaluating_rewards import serialize
from evaluating_rewards.analysis import results
from evaluating_rewards.analysis.dissimilarity_heatmaps import reward_masks, transformations


def short_e(x: float, precision: int = 2) -> str:
    """Formats 1.2345 as 1.2e-1, rather than Python 1.2e-01."""
    if not math.isfinite(x):
        return str(x)
    fmt = "{:." + str(precision) + "e}"
    formatted = fmt.format(x)
    base, exponent = formatted.split("e")
    exponent = int(exponent)
    return f"{base}e{exponent}"


def horizontal_ticks() -> None:
    plt.xticks(rotation="horizontal")
    plt.yticks(rotation="horizontal")


def normalize_dissimilarity(s: pd.Series) -> pd.Series:
    """Divides by distance from Zero reward, an upper bound on the distance."""
    df = s.unstack(level=["source_reward_type", "source_reward_path"])
    zero_col_name = (serialize.ZERO_REWARD, "dummy")
    zero_dissimilarity = df.pop(zero_col_name)
    df = df.apply(lambda x: x / zero_dissimilarity)
    return df.unstack(level=df.index.names)


def comparison_heatmap(
    vals: pd.Series,
    ax: plt.Axes,
    log: bool = False,
    fmt: Callable[[float], str] = lambda x: f"{x:.2f}",
    cbar_kws: Optional[Dict[str, Any]] = None,
    cmap: str = "GnBu",
    robust: bool = False,
    preserve_order: bool = False,
    label_fstr: Optional[str] = None,
    mask: Optional[pd.Series] = None,
    yaxis: bool = True,
    **kwargs,
) -> None:
    """Plot a heatmap, with target_reward_type as x-axis and remainder as y-axis.

    This is intended for plotting reward function distances, comparing reward functions
    on the y-axis to those on the x-axis.

    Args:
        vals: The values to visualize.
        log: log-10 scale for the values if True.
        fmt: format string for annotated values.
        cmap: color map.
        robust: If true, set vmin and vmax to 25th and 75th quantiles.
            This makes the color scale robust to outliers, but will compress it
            more than is desirable if the data does not contain outliers.
        preserve_order: If True, retains the same order as the input index
            after rewriting the index values for readability. If false,
            sorts the rewritten values alphabetically.
        label_fstr: Format string to use for the label for the colorbar legend.` {args}` is
            replaced with arguments to distance and `{transform_start}` and `{transform_end}`
            is replaced with any transformations of the distance (e.g. log).
        mask: If provided, only display cells where mask is True.
        yaxis: Whether to plot labels for y-axis.
        **kwargs: passed through to sns.heatmap.
    """
    vals = transformations.index_reformat(vals, preserve_order)
    if mask is not None:
        mask = transformations.index_reformat(mask, preserve_order)

    data = np.log10(vals) if log else vals
    annot = vals.applymap(fmt)
    cbar_kws = dict(cbar_kws or {})

    if label_fstr is None:
        label_fstr = "{transform_start}D({args}){transform_end}"
    transform_start = r"\log_{10}\left(" if log else ""
    transform_end = r"\right)" if log else ""
    label = label_fstr.format(
        transform_start=transform_start, args="R_A,R_B", transform_end=transform_end
    )
    cbar_kws.setdefault("label", f"${label}$")

    if robust:
        flat = data.values.flatten()
        kwargs["vmin"], kwargs["vmax"] = np.quantile(flat, [0.25, 0.75])
    yticklabels = "auto" if yaxis else False
    sns.heatmap(
        data,
        annot=annot,
        fmt="s",
        cmap=cmap,
        cbar_kws=cbar_kws,
        mask=mask,
        ax=ax,
        yticklabels=yticklabels,
        **kwargs,
    )
    ax.set_xlabel(r"$R_B$")
    ax.set_ylabel(r"$R_A$" if yaxis else "")


def median_seeds(series: pd.Series) -> pd.Series:
    """Take the median over any seeds in a series."""
    seeds = [name for name in series.index.names if "seed" in name]
    if seeds:
        non_seeds = [name for name in series.index.names if name not in seeds]
        series = series.groupby(non_seeds, sort=False).median()
    return series


def compact(series: pd.Series) -> pd.Series:
    """Make series smaller, suitable for e.g. small figures."""
    series = median_seeds(series)
    if "target_reward_type" in series.index.names:
        targets = series.index.get_level_values("target_reward_type")
        series = series.loc[targets != serialize.ZERO_REWARD]
    return series


short_fmt = functools.partial(short_e, precision=1)


def compact_heatmaps(
    dissimilarity: pd.Series,
    masks: Mapping[str, Iterable[results.FilterFn]],
    after_plot: Callable[[], None] = lambda: None,
    **kwargs: Dict[str, Any],
) -> Mapping[str, plt.Figure]:
    """Plots a series of compact heatmaps, suitable for presentations.

    Args:
        dissimilarity: The loss between source and target.
                The index should consist of target_reward_type, one of
                source_reward_{type,path}, and any number of seed indices.
                source_reward_path, if present, is rewritten into source_reward_type
                and a seed index.
        masks: A mapping from strings to collections of filter functions. Any
                (source, reward) pair not matching one of these filters is masked
                from the figure.
        fmt: A Callable mapping losses to strings to annotate cells in heatmap.
        after_plot: Called after plotting, for environment-specific tweaks.
        kwargs: passed through to `comparison_heatmap`.

    Returns:
        A mapping from strings to figures.
    """
    dissimilarity = dissimilarity.copy()
    dissimilarity = transformations.rewrite_index(dissimilarity)
    dissimilarity = compact(dissimilarity)

    figs = {}
    for name, matchings in masks.items():
        fig, ax = plt.subplots(1, 1, squeeze=True)
        match_mask = reward_masks.compute_mask(dissimilarity, matchings)
        comparison_heatmap(dissimilarity, ax=ax, preserve_order=True, mask=match_mask, **kwargs)
        after_plot()
        figs[name] = fig

    return figs
