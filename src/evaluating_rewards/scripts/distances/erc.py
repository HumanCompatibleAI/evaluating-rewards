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

"""CLI script to plot heatmaps based on episode return of reward models."""

import collections
import itertools
import logging
import os
import pickle
from typing import Any, Dict, Iterable, Mapping

from imitation.data import rollout
from imitation.util import util as imit_util
import numpy as np
import sacred

from evaluating_rewards import datasets
from evaluating_rewards.analysis import util
from evaluating_rewards.distances import common_config, tabular
from evaluating_rewards.rewards import base
from evaluating_rewards.scripts import script_utils
from evaluating_rewards.scripts.distances import common

erc_distance_ex = sacred.Experiment("erc_distance")
logger = logging.getLogger("evaluating_rewards.scripts.distances.erc")


common.make_config(erc_distance_ex)


@erc_distance_ex.config
def default_config(env_name):
    """Default configuration values."""
    computation_kind = "sample"  # either "sample" or "mesh"
    corr_kind = "pearson"  # either "pearson" or "spearman"
    discount = 0.99  # discount rate for shaping
    n_episodes = 1024  # number of episodes to compute correlation w.r.t.

    # n_samples and n_mean_samples only applicable for sample approach
    trajectory_factory = datasets.trajectory_factory_from_serialized_policy
    trajectory_factory_kwargs = {
        "env_name": env_name,
        "policy_type": "random",
        "policy_path": "dummy",
        "n_envs": 16,
        "parallel": True,
    }
    dataset_tag = "random"

    _ = locals()
    del _


@erc_distance_ex.config
def logging_config(log_root, env_name, dataset_tag, corr_kind, discount):
    """Default logging configuration: hierarchical directory structure based on config."""
    log_dir = os.path.join(  # noqa: F841  pylint:disable=unused-variable
        log_root,
        "erc",
        env_name,
        dataset_tag,
        corr_kind,
        f"discount{discount}",
        imit_util.make_unique_timestamp(),
    )


@erc_distance_ex.named_config
def dataset_noise_rollouts(env_name):
    """Add noise to rollouts of serialized policy."""
    trajectory_factory = datasets.trajectory_factory_noise_wrapper
    trajectory_factory_kwargs = {
        "factory": datasets.trajectory_factory_from_serialized_policy,
        "policy_type": "random",
        "policy_path": "dummy",
        "noise_env_name": env_name,
        "env_name": env_name,
    }
    _ = locals()
    del _


@erc_distance_ex.named_config
def high_precision():
    """Compute tight confidence intervals for publication quality figures.

    Slow and not that much more informative so not worth it for exploratory data analysis.
    """
    n_episodes = 131072
    n_bootstrap = 10000
    _ = locals()
    del _


FAST_CONFIG = dict(
    n_episodes=64,
    trajectory_factory_kwargs={
        "n_envs": 4,
        "parallel": False,
    },
)

# Duplicate to have consistent interface with EPIC & NPEC
erc_distance_ex.add_named_config("test", FAST_CONFIG)
erc_distance_ex.add_named_config("fast", FAST_CONFIG)


def batch_compute_returns(
    trajectory_callable: datasets.TrajectoryCallable,
    models: Mapping[common_config.RewardCfg, base.RewardModel],
    discount: float,
    n_episodes: int,
    batch_episodes: int = 256,
) -> Mapping[common_config.RewardCfg, np.ndarray]:
    """Compute returns under `models` of trajectories sampled from `trajectory_callable`.

    Batches the trajectory sampling and computation to efficiently compute returns with a small
    memory footprint.

    Args:
        trajectory_callable: a callable which generates trajectories.
        models: a mapping from configurations to reward models.
        discount: the discount rate for computing returns.
        n_episodes: the total number of episodes to sample from `trajectory_callable`.
        batch_episodes: the maximum number of episodes to sample at a time. This should be chosen
            to be large enough to take advantage of parallel evaluation of the reward models and to
            amortize the fixed costs inherent in trajectory sampling. However, it should be small
            enough that a batch of trajectories does not take up too much memory. The default of
            256 episodes works well in most environments, but might need to be decreased for
            environments with very large trajectories (e.g. image observations, long episodes).
    """
    logger.info("Computing returns")
    remainder = n_episodes
    returns = collections.defaultdict(list)
    while remainder > 0:
        batch_size = min(batch_episodes, remainder)

        logger.info(f"Computing returns for {batch_size} episodes: {remainder}/{n_episodes} left")
        trajectories = trajectory_callable(rollout.min_episodes(batch_size))
        rets = base.compute_return_of_models(models, trajectories, discount)
        for k, v in rets.items():
            returns[k].append(v)
        remainder -= batch_size
    returns = {k: np.concatenate(v) for k, v in returns.items()}
    return returns


@erc_distance_ex.capture
def correlation_distance(
    returns: Mapping[common_config.RewardCfg, np.ndarray],
    x_reward_cfgs: Iterable[common_config.RewardCfg],
    y_reward_cfgs: Iterable[common_config.RewardCfg],
    corr_kind: str,
    n_bootstrap: int,
    alpha: float = 0.95,
) -> common_config.AggregatedDistanceReturn:
    """
    Computes correlation of episode returns.

    Args:
        returns: returns associated with each reward model for all of
            `x_reward_cfgs` and `y_reward_cfgs`.
        x_reward_cfgs: tuples of reward_type and reward_path for x-axis.
        y_reward_cfgs: tuples of reward_type and reward_path for y-axis.
        corr_kind: method to compute results, either "pearson" or "spearman".
        n_bootstrap: The number of bootstrap samples to take.
        alpha: The proportion to give a confidence interval over.

    Returns:
        Dissimilarity matrix.
    """
    x_rets = {cfg: returns[cfg] for cfg in x_reward_cfgs}
    y_rets = {cfg: returns[cfg] for cfg in y_reward_cfgs}

    if corr_kind == "pearson":
        distance_fn = tabular.pearson_distance
    elif corr_kind == "spearman":
        distance_fn = tabular.spearman_distance
    else:
        raise ValueError(f"Unrecognized correlation '{corr_kind}'")

    def ci_fn(rewa: np.ndarray, rewb: np.ndarray) -> Mapping[str, float]:
        distances = util.bootstrap(rewa, rewb, stat_fn=distance_fn, n_samples=n_bootstrap)
        lower, middle, upper = util.empirical_ci(distances, alpha)
        return {
            "bootstrap_lower": lower,
            "bootstrap_middle": middle,
            "bootstrap_upper": upper,
            "bootstrap_width": upper - lower,
            "bootstrap_relative": common.relative_error(lower, middle, upper),
        }

    logger.info("Computing distance")
    distance = util.cross_distance(x_rets, y_rets, ci_fn, parallelism=1)

    vals = {}
    for k1, v1 in distance.items():
        for k2, v2 in v1.items():
            vals.setdefault(k2, {})[k1] = v2
    return vals


@erc_distance_ex.capture
def compute_vals(
    env_name: str,
    discount: float,
    x_reward_cfgs: Iterable[common_config.RewardCfg],
    y_reward_cfgs: Iterable[common_config.RewardCfg],
    trajectory_factory: datasets.TrajectoryFactory,
    trajectory_factory_kwargs: Dict[str, Any],
    n_episodes: int,
    log_dir: str,
) -> common_config.AggregatedDistanceReturn:
    """Entry-point into script to produce divergence heatmaps.

    Args:
        env_name: the name of the environment to compare rewards for.
        discount: discount to use for reward models (mostly for shaping).
        x_reward_cfgs: tuples of reward_type and reward_path for x-axis.
        y_reward_cfgs: tuples of reward_type and reward_path for y-axis.
        discount: the discount rate for shaping.
        trajectory_factory: factory to generate trajectories.
        trajectory_factory_kwargs: arguments to pass to the factory.
        n_episodes: the number of episodes to compute correlation over.
        log_dir: directory to save data to.

    Returns:
        Nested dictionary of aggregated distance values.
    """
    models, _, sess = common.load_models_create_sess(
        env_name, discount, itertools.chain(x_reward_cfgs, y_reward_cfgs)
    )

    logger.info("Sampling trajectories")
    with trajectory_factory(**trajectory_factory_kwargs) as trajectory_callable:
        with sess.as_default():
            returns = batch_compute_returns(trajectory_callable, models, discount, n_episodes)

    logger.info("Saving episode returns")
    with open(os.path.join(log_dir, "returns.pkl"), "wb") as f:
        pickle.dump(returns, f)

    aggregated = correlation_distance(  # pylint:disable=no-value-for-parameter
        returns, x_reward_cfgs, y_reward_cfgs
    )
    return aggregated


common.make_main(erc_distance_ex, compute_vals)

if __name__ == "__main__":
    script_utils.experiment_main(erc_distance_ex, "erc_distance")
