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

import json
import logging
import os
import re
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Internal dependencies

SYMBOLS = {
    "running": "\U0001f3c3",  # runner
    "backflip": "\U0001f938",  # cartwheel: closest Unicode has
    "withctrl": "\U0001f40c",  # snail
    "noctrl": "\U0001f406",  # cheetah
}


TRANSFORMATIONS = {
    r"^evaluating_rewards[_/](.*)-v0": r"\1",
    r"^imitation[_/](.*)-v0": r"\1",
    "^Zero-v0": "Zero",
    "^PointMassDense": "Dense",
    "^PointMassDenseNoCtrl": "Dense\nNo Ctrl",
    "^PointMassGroundTruth": "Norm",
    "^PointMassSparse": "Sparse",
    "^PointMassSparseNoCtrl": "Sparse\nNo Ctrl",
    "^PointMazeGroundTruth": "GT",
    r"(.*)(Hopper|HalfCheetah)GroundTruth(.*)": f"\\1\\2{SYMBOLS['running']}\\3",
    r"(.*)(Hopper|HalfCheetah)Backflip(.*)": f"\\1\\2{SYMBOLS['backflip']}\\3",
    r"^Hopper(.*)": r"\1",
    r"^HalfCheetah(.*)": r"\1",
    r"^(.*)Backward(.*)": r"\1←\2",
    r"^(.*)Forward(.*)": r"\1→\2",
    r"^(.*)WithCtrl(.*)": f"\\1{SYMBOLS['withctrl']}\\2",
    r"^(.*)NoCtrl(.*)": f"\\1{SYMBOLS['noctrl']}\\2",
}


LEVEL_NAMES = {
    # Won't normally use both type and path in one plot
    "source_reward_type": "Source",
    "source_reward_path": "Source",
    "target_reward_type": "Target",
    "target_reward_path": "Target",
}


def pretty_rewrite(x):
    if not isinstance(x, str):
        return x

    for pattern, repl in TRANSFORMATIONS.items():
        x = re.sub(pattern, repl, x)
    return x


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


def save_fig(path: str, fig: plt.Figure, fmt: str = "pdf", dpi: int = 300, **kwargs):
    path = f"{path}.{fmt}"
    root_dir = os.path.dirname(path)
    os.makedirs(root_dir, exist_ok=True)
    logging.info(f"Saving figure to {path}")
    with open(path, "wb") as f:
        fig.savefig(f, format=fmt, dpi=dpi, transparent=True, **kwargs)


def save_figs(root_dir: str, generator: Iterable[Tuple[str, plt.Figure]], **kwargs) -> None:
    for name, fig in generator:
        name = name.replace("/", "_")
        path = os.path.join(root_dir, name)
        save_fig(path, fig, **kwargs)


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


def path_to_config(types: Iterable[str], paths: Iterable[str]) -> pd.DataFrame:
    """Extracts relevant config parameters from paths in index.

    Args:
        types: An index of typses.
        paths: An index of paths.

    Returns:
        A MultiIndex consisting of original reward type and seed(s).
    """
    seen = {}
    res = []
    assert len(types) == len(paths)
    for (type, path) in zip(types, paths):

        if type in HARDCODED_TYPES:
            res.append((type, "hardcoded", 0, 0))
            continue
        else:
            config, run, path = _find_sacred_parent(path, seen)
            if "target_reward_type" in config:
                # Learning directly from a reward: e.g. train_{regress,preferences}
                pretty_type = {"train_regress": "regress", "train_preferences": "preferences"}
                model_type = pretty_type[run["command"]]
                res.append((config["target_reward_type"], model_type, config["seed"], 0))
            elif "rollout_path" in config:
                # Learning from demos: e.g. train_adversarial
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


def rewrite_index(series: pd.Series) -> pd.Series:
    if "source_reward_path" in series.index.names:
        new_index = series.index.to_frame(index=False)
        source_reward = path_to_config(
            new_index["source_reward_type"], new_index["source_reward_path"]
        )
        new_index = new_index.drop(columns="source_reward_path")
        new_index = pd.concat([source_reward, new_index], axis=1)
        new_index = pd.MultiIndex.from_frame(new_index)
        series.index = new_index
    return series


def short_e(x: float, precision: int = 2) -> str:
    """Formats 1.2345 as 1.2e-1, rather than Python 1.2e-01."""
    fmt = "{:." + str(precision) + "e}"
    formatted = fmt.format(x)
    base, exponent = formatted.split("e")
    exponent = int(exponent)
    return f"{base}e{exponent}"


def _is_str_ascii(x: str) -> bool:
    # TODO(): in 3.7+, can just use .isascii method
    try:
        x.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _is_ascii(idx: pd.Index) -> bool:
    return all([_is_str_ascii(str(x)) for x in idx])


def comparison_heatmap(
    vals: pd.Series,
    log: bool = True,
    fmt: Callable[[float], str] = short_e,
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

    def to_df(series):
        """Helper to reformat labels for ease of interpretability."""
        series = series.copy()
        series = rewrite_index(series)
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

    vals = to_df(vals)
    if mask is not None:
        mask = to_df(mask)

    data = np.log10(vals) if log else vals
    cbar_kws = dict(label=r"$-\log_{10}(q)$") if log else dict()

    annot = vals.applymap(fmt)

    if robust:
        flat = data.values.flatten()
        kwargs["vmin"], kwargs["vmax"] = np.quantile(flat, [0.25, 0.75])
    ax = sns.heatmap(data, annot=annot, fmt="s", cmap=cmap, cbar_kws=cbar_kws, mask=mask, **kwargs)

    labels_symbols = not (_is_ascii(vals.index) and _is_ascii(vals.columns))
    if labels_symbols:
        # TODO(): ideally we'd set symbola as a fall-back font, but this
        # is not supported by matplotlib currently, see e.g.
        # https://stackoverflow.com/questions/53581589/matplotlib-can-i-use-a-secondary-font-for-missing-glyphs
        ax.set_xticklabels(ax.get_xticklabels(), fontfamily="symbola")
        ax.set_yticklabels(ax.get_yticklabels(), fontfamily="symbola")


def print_symbols(
    out_dir: str, symbols: Optional[Mapping[str, str]] = None, fontfamily: str = "symbola"
) -> Mapping[str, plt.Figure]:
    """Save PNG rendering of `symbols` to `out_dir`."""
    if symbols is None:
        symbols = SYMBOLS

    figs = {}
    for symbol_name, symbol in symbols.items():
        fig = plt.figure(figsize=(0.15, 0.15))
        plt.text(0, 0, symbol, fontfamily=fontfamily)
        plt.axis("off")
        plt.tight_layout()
        figs[symbol_name] = fig

    save_figs(out_dir, figs.items(), bbox_inches=0, fmt="png", dpi=1200)

    return figs
