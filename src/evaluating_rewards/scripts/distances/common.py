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

"""Helper methods and common configurations for scripts to evaluate distances between rewards."""

import functools
import logging
import os
import pickle
from typing import Callable, Iterable, Mapping, Sequence

import gym
import numpy as np
import sacred
import scipy.stats
from stable_baselines.common import vec_env
import tensorflow as tf

from evaluating_rewards import serialize
from evaluating_rewards.analysis import util
from evaluating_rewards.distances import common_config
from evaluating_rewards.rewards import base

AggregateFn = Callable[[Sequence[float]], Mapping[str, float]]

logger = logging.getLogger("evaluating_rewards.scripts.distances.common")


def load_models(
    env_name: str,
    reward_cfgs: Iterable[common_config.RewardCfg],
    discount: float,
) -> Mapping[common_config.RewardCfg, base.RewardModel]:
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


def relative_error(
    lower: float,
    mean: float,
    upper: float,
) -> float:
    """Compute relative upper bound rel, which the mean is within +/- rel % of.

    Since this may be asymmetric between lower and upper, returns the maximum of
    upper / mean - 1 (relative upper bound) and 1 - lower / mean (relative lower bound).
    """
    return max(upper / mean - 1, 1 - lower / mean)


def bootstrap_ci(vals: Iterable[float], n_bootstrap: int, alpha: float) -> Mapping[str, float]:
    """Compute `alpha` %ile confidence interval of mean of `vals` from `n_bootstrap` samples."""
    bootstrapped = util.bootstrap(np.array(vals), stat_fn=np.mean, n_samples=n_bootstrap)
    lower, middle, upper = util.empirical_ci(bootstrapped, alpha)
    return {
        "lower": lower,
        "middle": middle,
        "upper": upper,
        "width": upper - lower,
        "relative": relative_error(lower, middle, upper),
    }


def studentt_ci(vals: Sequence[float], alpha: float) -> Mapping[str, float]:
    """Compute `alpha` %ile confidence interval of mean of `vals` using t-distribution."""
    assert len(vals) > 1
    df = len(vals) - 1
    mu = np.mean(vals)
    stderr = scipy.stats.sem(vals)
    lower, upper = scipy.stats.t.interval(alpha / 100, df, loc=mu, scale=stderr)
    return {
        "lower": lower,
        "middle": mu,
        "upper": upper,
        "width": upper - lower,
        "relative": relative_error(lower, mu, upper),
    }


def sample_mean_sd(vals: Sequence[float]) -> Mapping[str, float]:
    """Returns sample mean and (unbiased) standard deviation."""
    assert len(vals) > 1
    return {"mean": np.mean(vals), "sd": np.std(vals, ddof=1)}


def make_config(
    experiment: sacred.Experiment,
):  # pylint: disable=unused-variable,too-many-statements
    """Adds configs and named configs to `experiment`.

    The standard config parameters it defines are:
        - vals_path (Optional[str]): path to precomputed values to plot.
        - env_name (str): The environment name in the Gym registry of the rewards to compare.
        - x_reward_cfgs (Iterable[common_config.RewardCfg]): tuples of reward_type and reward_path
            for x-axis.
        - y_reward_cfgs (Iterable[common_config.RewardCfg]): tuples of reward_type and reward_path
            for y-axis.
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
        data_root = serialize.get_output_dir()  # where models are read from
        log_root = serialize.get_output_dir()  # where results are written to
        n_bootstrap = 1000  # number of bootstrap samples
        alpha = 95  # percentile confidence interval
        aggregate_kinds = ("bootstrap", "studentt", "sample")
        vals_path = None

        _ = locals()
        del _

    @experiment.config
    def point_mass_as_default():
        """Default to PointMass as environment so scripts work out-of-the-box."""
        locals().update(**common_config.COMMON_CONFIGS["point_mass"])

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

    for name, cfg in common_config.COMMON_CONFIGS.items():
        experiment.add_named_config(name, cfg)


def make_main(
    experiment: sacred.Experiment,
    compute_vals: Callable[..., common_config.AggregatedDistanceReturn],
):  # pylint: disable=unused-variable
    """Helper to make main function for distance scripts.

    Specifically, register a main function with `experiment` that:
      - Creates log directory.
      - Canonicalizes reward configurations.
      - Loads reward models.
      - Calls `compute_vals`.
      - Saves the return value of `compute_vals` in the log directory.

    Args:
        experiment: the Sacred experiment to register a main function with.
        compute_vals: a function to call to compute the distances.
            It is always passed a graph `g`, session `sess`, loaded models `models`, and
            canonicalized reward configurations `x_reward_cfgs` and `y_reward_cfgs`.
            It is typically defined with `@experiment.capture` in which case Sacred will
            fill in other parameters defined in the configuration.
      -"""

    @experiment.main
    def main(
        env_name: str,
        discount: float,
        x_reward_cfgs: Iterable[common_config.RewardCfg],
        y_reward_cfgs: Iterable[common_config.RewardCfg],
        data_root: str,
        log_dir: str,
    ) -> common_config.AggregatedDistanceReturn:
        """Wrapper around `compute_vals` performing common setup and saving logic.

        Args:
            env_name: the name of the environment to compare rewards for.
            x_reward_cfgs: tuples of reward_type and reward_path for x-axis.
            y_reward_cfgs: tuples of reward_type and reward_path for y-axis.
            data_root: directory to load learned reward models from.
            log_dir: directory to save data to.

        Returns:
            The values returned by `compute_vals`.
        """
        os.makedirs(log_dir, exist_ok=True)  # fail fast if log directory cannot be created

        # Sacred turns our tuples into lists :(, undo
        x_reward_cfgs = [
            common_config.canonicalize_reward_cfg(cfg, data_root) for cfg in x_reward_cfgs
        ]
        y_reward_cfgs = [
            common_config.canonicalize_reward_cfg(cfg, data_root) for cfg in y_reward_cfgs
        ]

        logger.info("Loading models")
        g = tf.Graph()
        with g.as_default():
            sess = tf.Session()
            with sess.as_default():
                reward_cfgs = x_reward_cfgs + y_reward_cfgs
                models = load_models(env_name, reward_cfgs, discount)

        # If `compute_vals` is a capture function, then Sacred will fill in other parameters
        aggregated = compute_vals(
            g=g, sess=sess, models=models, x_reward_cfgs=x_reward_cfgs, y_reward_cfgs=y_reward_cfgs
        )

        aggregated_path = os.path.join(log_dir, "aggregated.pkl")
        logger.info(f"Saving aggregated values to {aggregated_path}")
        with open(aggregated_path, "wb") as f:
            pickle.dump(aggregated, f)

        return aggregated
