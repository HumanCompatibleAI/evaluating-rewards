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
from typing import Callable, Iterable, Mapping, Sequence, Tuple
import warnings

import gym
import numpy as np
import sacred
import scipy.stats
from stable_baselines.common import vec_env
import tensorflow as tf

from evaluating_rewards import datasets, serialize
from evaluating_rewards.analysis import util
from evaluating_rewards.distances import common_config
from evaluating_rewards.rewards import base

AggregateFn = Callable[[Sequence[float]], Mapping[str, float]]
Dissimilarities = Mapping[Tuple[common_config.RewardCfg, common_config.RewardCfg], Sequence[float]]

logger = logging.getLogger("evaluating_rewards.scripts.distances.common")


def load_models(
    env_name: str,
    discount: float,
    reward_cfgs: Iterable[common_config.RewardCfg],
) -> Mapping[common_config.RewardCfg, base.RewardModel]:
    """Load models specified by the `reward_cfgs`.

    Args:
        - env_name: The environment name in the Gym registry of the rewards to compare.
        - discount: Discount to use for reward models (mostly for shaping).
        - reward_cfgs: Iterable of reward configurations.

    Returns:
         A mapping from reward configurations to the loaded reward model.
    """
    venv = vec_env.DummyVecEnv([lambda: gym.make(env_name)])
    return {
        (kind, path): serialize.load_reward(kind, path, venv, discount)
        for kind, path in reward_cfgs
    }


def load_models_create_sess(
    env_name: str,
    discount: float,
    reward_cfgs: Iterable[common_config.RewardCfg],
) -> Tuple[Mapping[Tuple[str, str], base.RewardModel], tf.Graph, tf.Session]:
    """Load models specified by `reward_cfgs`, in a fresh session."""
    logger.info("Loading models")
    g = tf.Graph()
    with g.as_default():
        sess = tf.Session()
        with sess.as_default():
            models = load_models(env_name, discount, reward_cfgs)
    return models, g, sess


def relative_error(
    lower: float,
    mean: float,
    upper: float,
) -> float:
    """Compute relative upper bound rel, which the mean is within +/- rel % of.

    Since this may be asymmetric between lower and upper, returns the maximum of
    upper / mean - 1 (relative upper bound) and 1 - lower / mean (relative lower bound).

    To handle negative numbers, returns the absolute value of this.
    """
    if lower * upper < 0:
        warnings.warn(f"lower = '{lower}' different sign to upper = '{upper}'", UserWarning)
        return np.nan
    else:
        return abs(max(upper / mean - 1, 1 - lower / mean))


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


def make_config(experiment: sacred.Experiment) -> None:
    """Adds configs and named configs to `experiment`.

    The standard config parameters it defines are:
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
    # pylint: disable=unused-variable,too-many-statements

    @experiment.config
    def default_config():
        """Default configuration values."""
        data_root = serialize.get_output_dir()  # where models are read from
        log_root = serialize.get_output_dir()  # where results are written to
        n_bootstrap = 1000  # number of bootstrap samples
        alpha = 95  # percentile confidence interval
        aggregate_kinds = ("bootstrap", "studentt", "sample")

        _ = locals()
        del _

    @experiment.config
    def aggregate_fns(aggregate_kinds, n_bootstrap, alpha):
        """Make a mapping of aggregate functions of kinds `subset` with specified parameters.

        Used in scripts.distances.{epic,npec}; currently ignored by erc since
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

    @experiment.config
    def point_mass_as_default():
        """Default to PointMass as environment so scripts work out-of-the-box."""
        locals().update(**common_config.COMMON_CONFIGS["point_mass"])

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

        # If `compute_vals` is a capture function, then Sacred will fill in other parameters
        aggregated = compute_vals(x_reward_cfgs=x_reward_cfgs, y_reward_cfgs=y_reward_cfgs)

        aggregated_path = os.path.join(log_dir, "aggregated.pkl")
        logger.info(f"Saving aggregated values to {aggregated_path}")
        with open(aggregated_path, "wb") as f:
            pickle.dump(aggregated, f)

        return aggregated


def _visitation_config(env_name, visitations_factory_kwargs):
    """Default visitation distribution config: rollouts from random policy."""
    # visitations_factory only has an effect when computation_kind == "sample"
    visitations_factory = datasets.transitions_factory_from_serialized_policy
    if visitations_factory_kwargs is None:
        visitations_factory_kwargs = {
            "env_name": env_name,
            "policy_type": "random",
            "policy_path": "dummy",
        }
    dataset_tag = "random_policy"
    return locals()


def aggregate_seeds(
    aggregate_fns: Mapping[str, AggregateFn],
    dissimilarities: Dissimilarities,
) -> common_config.AggregatedDistanceReturn:
    """Use `aggregate_fns` to aggregate sequences of data in `dissimilarities`.

    Args:
        aggregate_fns: Mapping from string names to aggregate functions.
        dissimilarities: Mapping from pairs of reward configurations to sequences
            of floats -- numerical dissimilarities from different seeds.

    Returns:
        The different seeds aggregated using `aggregate_fns`. The mapping has keys
        comprised of `{name}_{k}` where `name` is from a key in `aggregate_fns`
        and `k` is a key from the return value of the aggregation function.
    """
    vals = {}
    for name, aggregate_fn in aggregate_fns.items():
        logger.info(f"Aggregating {name}")
        for k, v in dissimilarities.items():
            for k2, v2 in aggregate_fn(v).items():
                outer_key = f"{name}_{k2}"
                vals.setdefault(outer_key, {})[k] = v2

    return vals


def _ignore_extraneous_dataset_iid(real_kwargs, **_):
    """Thin-wrapper to workaround Sacred inability to delete keys from config dict."""
    return datasets.transitions_factory_iid_from_sample_dist_factory(**real_kwargs)


def _ignore_extraneous_random_model(real_kwargs, **_):
    """Thin-wrapper to workaround Sacred inability to delete keys from config dict."""
    return datasets.transitions_factory_from_random_model(**real_kwargs)


def make_transitions_configs(
    experiment: sacred.Experiment,
):  # pylint: disable=unused-variable
    """Add configs to experiment `ex` related to visitations transition factory."""

    @experiment.config
    def _visitation_dummy_config():
        visitations_factory_kwargs = None  # noqa: F841

    @experiment.config
    def _visitation_default_config(env_name, visitations_factory_kwargs):
        locals().update(**_visitation_config(env_name, visitations_factory_kwargs))

    @experiment.named_config
    def visitation_config(env_name):
        """Named config that sets default visitation factory.

        This is needed for other named configs that manipulate visitation factories, but can
        otherwise be omitted since `_visitation_default_config` will do the same update."""
        locals().update(**_visitation_config(env_name, None))

    @experiment.named_config
    def sample_from_env_spaces(env_name):
        """Randomly sample from Gym spaces."""
        obs_sample_dist_factory = functools.partial(datasets.sample_dist_from_env_name, obs=True)
        act_sample_dist_factory = functools.partial(datasets.sample_dist_from_env_name, obs=False)
        sample_dist_factory_kwargs = {"env_name": env_name}
        obs_sample_dist_factory_kwargs = {}
        act_sample_dist_factory_kwargs = {}
        sample_dist_tag = "random_space"  # only used for logging
        _ = locals()
        del _

    @experiment.named_config
    def dataset_iid(
        env_name,
        obs_sample_dist_factory,
        act_sample_dist_factory,
        obs_sample_dist_factory_kwargs,
        act_sample_dist_factory_kwargs,
        sample_dist_factory_kwargs,
        sample_dist_tag,
    ):
        """Visitation distribution is i.i.d. samples from sample distributions.

        Set this to make `computation_kind` "sample" consistent with "mesh".

        WARNING: you *must* override the `sample_dist` *before* calling this,
        e.g. by using `sample_from_env_spaces`, since by default it is marginalized from
        `visitations_factory`, leading to an infinite recursion.
        """
        visitations_factory = _ignore_extraneous_dataset_iid
        visitations_factory_kwargs = {
            "real_kwargs": {
                "obs_dist_factory": obs_sample_dist_factory,
                "act_dist_factory": act_sample_dist_factory,
                "obs_kwargs": obs_sample_dist_factory_kwargs,
                "act_kwargs": act_sample_dist_factory_kwargs,
                "env_name": env_name,
            }
        }
        visitations_factory_kwargs["real_kwargs"].update(**sample_dist_factory_kwargs)
        dataset_tag = "iid_" + sample_dist_tag
        _ = locals()
        del _

    @experiment.named_config
    def dataset_from_random_transitions(env_name):
        visitations_factory = _ignore_extraneous_random_model
        visitations_factory_kwargs = {"real_kwargs": {"env_name": env_name}}
        dataset_tag = "random_transitions"
        _ = locals()
        del _

    @experiment.named_config
    def dataset_permute(visitations_factory, visitations_factory_kwargs, dataset_tag):
        """Permute transitions of factory specified in *previous* named configs on the CLI."""
        visitations_factory_kwargs["factory"] = visitations_factory
        visitations_factory = datasets.transitions_factory_permute_wrapper
        dataset_tag = "permuted_" + dataset_tag
        _ = locals()
        del _

    @experiment.named_config
    def dataset_noise_rollouts(env_name):
        """Add noise to rollouts of serialized policy."""
        visitations_factory_kwargs = {
            "trajectory_factory": datasets.trajectory_factory_noise_wrapper,
            "factory": datasets.trajectory_factory_from_serialized_policy,
            "policy_type": "random",
            "policy_path": "dummy",
            "noise_env_name": env_name,
            "env_name": env_name,
        }
        visitations_factory = datasets.transitions_factory_from_trajectory_factory
        dataset_tag = "noised_random_policy"
        _ = locals()
        del _
