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

"""Configurations for dissimilarity_heatmaps heatmaps.

Shared between `evaluating_rewards.analysis.{plot_epic_heatmap,plot_canon_heatmap}`.
"""

import functools
import itertools
import logging
import os
import pickle
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sacred
import scipy.stats
from stable_baselines.common import vec_env

from evaluating_rewards import rewards, serialize, util
from evaluating_rewards.analysis import stylesheets, visualize
from evaluating_rewards.analysis.dissimilarity_heatmaps import heatmaps, reward_masks

AggregateFn = Callable[[Sequence[float]], Mapping[str, float]]
RewardCfg = Tuple[str, str]  # (type, path)

logger = logging.getLogger("evaluating_rewards.analysis.dissimilarity_heatmaps.cli_common")


def canonicalize_reward_cfg(reward_cfg: RewardCfg, data_root: str) -> RewardCfg:
    """Canonicalize path in reward configuration.

    Specifically, join paths with the `data_root`, unless they are the special "dummy" path.
    Also ensure the return value is actually of type RewardCfg: it is forgiving and will accept
    any iterable pair as input `reward_cfg`. This is important since Sacred has the bad habit of
    converting tuples to lists in configurations.

    Args:
        reward_cfg: Iterable of configurations to canonicailze.
        data_root: The root to join paths to.

    Returns:
        Canonicalized RewardCfg.
    """
    kind, path = reward_cfg
    if path != "dummy":
        path = os.path.join(data_root, path)
    return (kind, path)


def load_models(
    env_name: str, reward_cfgs: Iterable[RewardCfg], discount: float,
) -> Mapping[RewardCfg, rewards.RewardModel]:
    """Load models specified by the `reward_cfgs`.

    Args:
        - env_name: The environment name in the Gym registry of the rewards to compare.
        - reward_cfgs: Iterable of reward configurations.
        - discount: Discount to use for reward models (mostly for shaping).

    Returns:
         A mapping from reward configurations to the loaded reward model.
    """
    venv = vec_env.DummyVecEnv([lambda: gym.make(env_name)])
    return {
        (kind, path): serialize.load_reward(kind, path, venv, discount)
        for kind, path in reward_cfgs
    }


def bootstrap_ci(vals: Iterable[float], n_bootstrap: int, alpha: float) -> Mapping[str, float]:
    """Compute `alpha` %ile confidence interval of mean of `vals` from `n_bootstrap` samples."""
    bootstrapped = util.bootstrap(np.array(vals), stat_fn=np.mean, n_samples=n_bootstrap)
    lower, middle, upper = util.empirical_ci(bootstrapped, alpha)
    return {"lower": lower, "middle": middle, "upper": upper, "width": upper - lower}


def studentt_ci(vals: Sequence[float], alpha: float) -> Mapping[str, float]:
    """Compute `alpha` %ile confidence interval of mean of `vals` using t-distribution."""
    assert len(vals) > 1
    df = len(vals) - 1
    mu = np.mean(vals)
    stderr = scipy.stats.sem(vals)
    lower, upper = scipy.stats.t.interval(alpha / 100, df, loc=mu, scale=stderr)
    return {"lower": lower, "middle": mu, "upper": upper, "width": upper - lower}


def sample_mean_sd(vals: Sequence[float]) -> Mapping[str, float]:
    """Returns sample mean and (unbiased) standard deviation."""
    assert len(vals) > 1
    return {"mean": np.mean(vals), "sd": np.std(vals, ddof=1)}


def oned_mapping_to_series(dissimilarity: Mapping[Tuple[RewardCfg, RewardCfg], float]) -> pd.Series:
    """Converts mapping to a series.

    Args:
        dissimilarity: A mapping from pairs of configurations to a float.

    Returns:
        A Series with a multi-index based on the configurations.
    """
    dissimilarity = {
        (xtype, xpath, ytype, ypath): v
        for ((xtype, xpath), (ytype, ypath)), v in dissimilarity.items()
    }
    dissimilarity = pd.Series(dissimilarity)
    dissimilarity.index.names = [
        "target_reward_type",
        "target_reward_path",
        "source_reward_type",
        "source_reward_path",
    ]
    return dissimilarity


def twod_mapping_to_multi_series(
    aggregated: Mapping[Any, Mapping[str, float]]
) -> Mapping[str, pd.Series]:
    """Converts a nested mapping to a mapping of dissimilarity series.

    Args:
        aggregated: A mapping over a mapping from strings to sequences of floats.

    Returns:
        A mapping from strings to MultiIndex series returned by `oned_mapping_to_series`,
        after transposing the inner and outer keys of the mapping.
    """
    keys = list(set((tuple(v.keys()) for v in aggregated.values())))
    assert len(keys) == 1
    vals = {outer_key: {k: v[outer_key] for k, v in aggregated.items()} for outer_key in keys[0]}
    return {k: oned_mapping_to_series(v) for k, v in vals.items()}


def apply_multi_aggregate_fns(
    dissimilarities: Mapping[Any, Sequence[float]], aggregate_fns: Mapping[str, AggregateFn],
) -> Mapping[str, pd.Series]:
    """Aggregate dissimilarities: e.g. confidence intervals.

    Args:
        dissimilarities: Mapping over sequences of floats.
        aggregate_fns: Mapping from strings to aggregators to be applied on sequences of floats.

    Returns:
         A mapping from string keys of the form "{aggregator}_{inner}" to Series, where
         `aggregator` is the name of the aggregation function and `inner` is a key returned
         by the aggregator.
    """
    res = {}
    for name, aggregate_fn in aggregate_fns.items():
        logger.info(f"Aggregating {name}")
        aggregated = {k: aggregate_fn(v) for k, v in dissimilarities.items()}
        aggregated = twod_mapping_to_multi_series(aggregated)
        res.update({f"{name}_{k}": v for k, v in aggregated.items()})
    return res


def _usual_label_fstr(distance_name: str) -> str:
    return "{transform_start}" + distance_name + "({args}){transform_end}"


def pretty_label_fstr(name: str) -> str:
    """Abbreviation to use in legend for colorbar."""
    if name.endswith("lower"):
        return _usual_label_fstr("D_L")
    elif name.endswith("upper"):
        return _usual_label_fstr("D_U")
    elif name.endswith("middle") or name.endswith("mean"):
        return _usual_label_fstr(r"\bar{{D}}")
    elif name.endswith("width"):
        return _usual_label_fstr("D_W")
    elif name.endswith("sd"):
        return r"\mathrm{{SD}}\left[" + _usual_label_fstr("D") + r"\right]"
    else:
        return _usual_label_fstr("D")


def multi_heatmaps(dissimilarities: Mapping[str, pd.Series], **kwargs) -> Mapping[str, plt.Figure]:
    """Plot heatmap for each dissimilarity series in `dissimilarities`.

    Args:
        dissimilarities: Mapping from strings to dissimilarity matrix.
        kwargs: Passed through to `heatmaps.compact_heatmaps`.

    Returns:
        A Mapping from strings to figures, with keys "{k}_{mask}" for mask config `mask` from
        `dissimilarities[k]`.
    """
    figs = {}
    for name, val in dissimilarities.items():
        label_fstr = pretty_label_fstr(name)
        extra_kwargs = {}
        if name.endswith("width"):
            extra_kwargs["fmt"] = functools.partial(heatmaps.short_e, precision=0)
        heatmap_figs = heatmaps.compact_heatmaps(
            dissimilarity=val, label_fstr=label_fstr, **kwargs, **extra_kwargs
        )
        figs.update({f"{name}_{k}": v for k, v in heatmap_figs.items()})
    return figs


def save_artifacts(
    vals: Mapping[str, pd.Series], styles: Iterable[str], log_dir: str, heatmap_kwargs, save_kwargs
) -> None:
    """Plot a figure for each entry in `vals`, and save figures as well as pickled raw values."""
    os.makedirs(log_dir, exist_ok=True)

    logging.info("Saving raw values")
    with open(os.path.join(log_dir, "vals.pkl"), "wb") as f:
        pickle.dump(vals, f)

    logging.info("Plotting figures")
    with stylesheets.setup_styles(styles):
        try:
            figs = multi_heatmaps(vals, **heatmap_kwargs)
            visualize.save_figs(log_dir, figs.items(), **save_kwargs)
        finally:
            for fig in figs:
                plt.close(fig)


MUJOCO_STANDARD_ORDER = [
    "ForwardNoCtrl",
    "ForwardWithCtrl",
    "BackwardNoCtrl",
    "BackwardWithCtrl",
]


POINT_MASS_KINDS = [
    f"evaluating_rewards/PointMass{label}-v0"
    for label in ["SparseNoCtrl", "SparseWithCtrl", "DenseNoCtrl", "DenseWithCtrl", "GroundTruth"]
]


def _norm(args: Iterable[str]) -> bool:
    return any(reward_masks.match("evaluating_rewards/PointMassGroundTruth-v0")(args))


def _hopper_activity(args: Iterable[str]) -> bool:
    pattern = r"evaluating_rewards/(.*)(GroundTruth|Backflip)(.*)"
    repl = reward_masks.replace(pattern, r"\1\2")(args)
    return len(set(repl)) > 1 and reward_masks.no_ctrl(args)


def _hardcoded_model_cfg(kinds: Iterable[str]) -> Iterable[RewardCfg]:
    return [(kind, "dummy") for kind in kinds]


def make_config(
    experiment: sacred.Experiment,
):  # pylint: disable=unused-variable,too-many-statements
    """Adds configs and named configs to `experiment`.

    The standard config parameters it defines are:
        - vals_path (Optional[str]): path to precomputed values to plot.
        - env_name (str): The environment name in the Gym registry of the rewards to compare.
        - x_reward_cfgs (Iterable[RewardCfg]): tuples of reward_type and reward_path for x-axis.
        - y_reward_cfgs (Iterable[RewardCfg]): tuples of reward_type and reward_path for y-axis.
        - log_root (str): the root directory to log; subdirectory path automatically constructed.
        - n_bootstrap (int): the number of bootstrap samples to take.
        - alpha (float): percentile confidence interval
        - aggregate_kinds (Iterable[str]): the type of aggregations to perform across seeds.
            Not used in `plot_return_heatmap` which only supports its own kind of bootstrapping.
        - heatmap_kwargs (dict): passed through to `analysis.compact_heatmaps`.
        - styles (Iterable[str]): styles to apply from `evaluating_rewards.analysis.stylesheets`.
        - save_kwargs (dict): passed through to `analysis.save_figs`.
    """

    @experiment.config
    def default_config():
        """Default configuration values."""
        env_name = "evaluating_rewards/PointMassLine-v0"
        kinds = POINT_MASS_KINDS
        log_root = serialize.get_output_dir()  # where results are read from/written to
        n_bootstrap = 1000  # number of bootstrap samples
        alpha = 95  # percentile confidence interval
        aggregate_kinds = ("bootstrap", "studentt", "sample")
        vals_path = None

        # Reward configurations: models to compare
        x_reward_cfgs = None
        y_reward_cfgs = None

        _ = locals()
        del _

    @experiment.config
    def reward_config(kinds, x_reward_cfgs, y_reward_cfgs):
        """Default reward configuration: hardcoded reward model types from kinds."""
        if kinds is not None:
            if x_reward_cfgs is None:
                x_reward_cfgs = _hardcoded_model_cfg(kinds)
            if y_reward_cfgs is None:
                y_reward_cfgs = _hardcoded_model_cfg(kinds)
        _ = locals()
        del _

    @experiment.config
    def figure_config(kinds):
        """Defaults for figure parameters."""
        heatmap_kwargs = {
            "masks": {"all": [reward_masks.always_true]},
            "after_plot": heatmaps.horizontal_ticks,
        }
        if kinds and "order" not in heatmap_kwargs:
            heatmap_kwargs["order"] = kinds
        styles = ["paper", "heatmap", "heatmap-3col", "tex"]
        styles_for_env = []
        save_kwargs = {
            "fmt": "pdf",
        }
        _ = locals()
        del _

    @experiment.config
    def aggregate_fns(aggregate_kinds, n_bootstrap, alpha):
        """Make a mapping of aggregate functions of kinds `subset` with specified parameters.

        Used in plot_{canon,epic}_heatmap; currently ignored by plot_return_heatmap since
        it does not use multiple seeds and instead bootstraps over samples.
        """
        aggregate_fns = {}
        if "bootstrap" in aggregate_kinds:
            aggregate_fns["bootstrap"] = functools.partial(
                bootstrap_ci, n_bootstrap=n_bootstrap, alpha=alpha
            )
        if "studentt" in aggregate_kinds:
            aggregate_fns["studentt"] = functools.partial(studentt_ci, alpha=alpha)
        if "sample" in aggregate_kinds:
            aggregate_fns["sample"] = sample_mean_sd

    @experiment.named_config
    def large():
        """Large output size, high precision."""
        styles = ["paper", "heatmap", "heatmap-1col", "tex"]
        heatmap_kwargs = {
            "fmt": heatmaps.short_e,
            "cbar_kws": dict(fraction=0.05),
        }
        _ = locals()
        del _

    @experiment.named_config
    def point_mass():
        """Heatmaps for evaluating_rewards/PointMass* environments."""
        env_name = "evaluating_rewards/PointMassLine-v0"
        kinds = POINT_MASS_KINDS
        heatmap_kwargs = {}
        heatmap_kwargs["masks"] = {
            "diagonal": [reward_masks.zero, reward_masks.same],
            "control": [reward_masks.zero, reward_masks.control],
            "dense_vs_sparse": [reward_masks.zero, reward_masks.sparse_or_dense],
            "norm": [reward_masks.zero, reward_masks.same, _norm],
            "all": [reward_masks.always_true],
        }
        _ = locals()
        del _

    @experiment.named_config
    def point_maze():
        """Heatmaps for imitation/PointMaze{Left,Right}-v0 environments."""
        env_name = "imitation/PointMazeLeft-v0"
        kinds = [
            "imitation/PointMazeGroundTruthWithCtrl-v0",
            "imitation/PointMazeGroundTruthNoCtrl-v0",
        ]
        heatmap_kwargs = {
            "masks": {"all": [reward_masks.always_true]},  # "all" is still only 2x2
        }
        _ = locals()
        del _

    @experiment.named_config
    def point_maze_learned():
        """Compare rewards learned in PointMaze to the ground-truth reward."""
        env_name = "imitation/PointMazeLeftVel-v0"
        x_reward_cfgs = [
            ("evaluating_rewards/PointMazeGroundTruthWithCtrl-v0", "dummy"),
        ]
        y_reward_cfgs = [
            ("evaluating_rewards/RewardModel-v0", "transfer_point_maze/reward/regress/model"),
            ("evaluating_rewards/RewardModel-v0", "transfer_point_maze/reward/preferences/model"),
            (
                "imitation/RewardNet_unshaped-v0",
                "transfer_point_maze/reward/irl_state_only/checkpoints/final/discrim/reward_net",
            ),
            (
                "imitation/RewardNet_unshaped-v0",
                "transfer_point_maze/reward/irl_state_action/checkpoints/final/discrim/reward_net",
            ),
        ]
        kinds = None
        _ = locals()
        del _

    @experiment.named_config
    def half_cheetah():
        """Heatmaps for HalfCheetah-v3."""
        env_name = "seals/HalfCheetah-v0"
        kinds = [
            f"evaluating_rewards/HalfCheetahGroundTruth{suffix}-v0"
            for suffix in MUJOCO_STANDARD_ORDER
        ]
        heatmap_kwargs = {
            "masks": {
                "diagonal": [reward_masks.zero, reward_masks.same],
                "control": [reward_masks.zero, reward_masks.control],
                "direction": [reward_masks.zero, reward_masks.direction],
                "no_ctrl": [reward_masks.zero, reward_masks.no_ctrl],
                "all": [reward_masks.always_true],
            },
        }
        styles_for_env = ["small-labels"]  # downscale emoji labels slightly
        _ = locals()
        del _

    @experiment.named_config
    def hopper():
        """Heatmaps for Hopper-v3."""
        env_name = "seals/Hopper-v0"
        activities = ["GroundTruth", "Backflip"]
        kinds = [
            f"evaluating_rewards/Hopper{prefix}{suffix}-v0"
            for prefix, suffix in itertools.product(activities, MUJOCO_STANDARD_ORDER)
        ]
        del activities
        heatmap_kwargs = {}
        heatmap_kwargs["masks"] = {
            "diagonal": [reward_masks.zero, reward_masks.same],
            "control": [reward_masks.zero, reward_masks.control],
            "direction": [reward_masks.zero, reward_masks.direction],
            "no_ctrl": [reward_masks.zero, reward_masks.no_ctrl],
            "different_activity": [reward_masks.zero, _hopper_activity],
            "all": [reward_masks.always_true],
        }
        styles_for_env = ["tiny-font"]
        _ = locals()
        del _


def make_main(
    experiment, compute_vals: Callable[[], Mapping[str, pd.Series]]
):  # pylint: disable=unused-variable
    """Insert entry-point into script to produce dissimilarity heatmaps.

    Args:
        experiment: The Sacred experiment to modify.
        compute_vals: An thunk (no-argument callable) which, when called, computes a mapping from
            keywords to Series containing dissimilarity values to plot. It is usually a Sacred
            capture function which obtains its other arguments from the Sacred config implicitly.
    """

    @experiment.main
    def main(
        vals_path: Optional[str],
        styles: Iterable[str],
        styles_for_env: Iterable[str],
        log_dir: str,
        heatmap_kwargs: Mapping[str, Any],
        save_kwargs: Mapping[str, Any],
    ) -> None:
        """
        Entry-point into script to produce dissimilarity heatmaps.

        Args:
            vals_path: path to precomputed values to plot. Skips everything but plotting logic
                if specified. This is useful for regenerating figures in a new style from old data.
            styles: styles to apply from `evaluating_rewards.analysis.stylesheets`.
            styles_for_env: additional styles to apply, set by environment-specific configs.
            log_dir: directory to write figures and other logging to.
            heatmap_kwargs: passed through to `analysis.compact_heatmaps`.
            save_kwargs: passed through to `analysis.save_figs`.
        """
        if vals_path is not None:
            with open(vals_path, "rb") as f:
                vals = pickle.load(f)
        else:
            vals = compute_vals()

        styles = list(styles) + list(styles_for_env)
        save_artifacts(vals, styles, log_dir, heatmap_kwargs, save_kwargs)
