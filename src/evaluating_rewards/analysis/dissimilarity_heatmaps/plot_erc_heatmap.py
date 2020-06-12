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
import logging
import os
from typing import Any, Dict, Iterable, Mapping, Tuple

from imitation.util import util as imit_util
import numpy as np
import pandas as pd
import sacred
import tensorflow as tf

from evaluating_rewards import datasets, rewards, tabular, util
from evaluating_rewards.analysis.dissimilarity_heatmaps import cli_common
from evaluating_rewards.scripts import script_utils

plot_erc_heatmap_ex = sacred.Experiment("plot_erc_heatmap")
logger = logging.getLogger("evaluating_rewards.analysis.plot_erc_heatmap")


cli_common.make_config(plot_erc_heatmap_ex)


@plot_erc_heatmap_ex.config
def default_config(env_name, log_root):
    """Default configuration values."""
    data_root = log_root  # root of data directory for learned reward models
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

    # Figure parameters
    heatmap_kwargs = {}
    _ = locals()
    del _


@plot_erc_heatmap_ex.config
def logging_config(log_root, env_name, dataset_tag, corr_kind, discount):
    """Default logging configuration: hierarchical directory structure based on config."""
    log_dir = os.path.join(  # noqa: F841  pylint:disable=unused-variable
        log_root,
        "plot_return_heatmap",
        env_name,
        dataset_tag,
        corr_kind,
        f"discount{discount}",
        imit_util.make_unique_timestamp(),
    )


@plot_erc_heatmap_ex.named_config
def paper():
    """Figures suitable for inclusion in paper.

    By convention we present them to the right, so turn off y-axis labels.
    """
    styles = ["paper", "heatmap", "heatmap-3col", "heatmap-3col-right", "tex"]
    heatmap_kwargs = {"yaxis": False, "vmin": 0.0, "vmax": 1.0}
    _ = locals()
    del _


@plot_erc_heatmap_ex.named_config
def high_precision():
    """Compute tight confidence intervals for publication quality figures.

    Slow and not that much more informative so not worth it for exploratory data analysis.
    """
    n_episodes = 131072
    n_bootstrap = 10000
    _ = locals()
    del _


@plot_erc_heatmap_ex.named_config
def test():
    """Intended for debugging/unit test."""
    n_episodes = 64
    # Do not include "tex" in styles here: this will break on CI.
    styles = ["paper", "heatmap-1col"]
    _ = locals()
    del _


@plot_erc_heatmap_ex.capture
def correlation_distance(
    returns: Mapping[cli_common.RewardCfg, np.ndarray],
    x_reward_cfgs: Iterable[cli_common.RewardCfg],
    y_reward_cfgs: Iterable[cli_common.RewardCfg],
    corr_kind: str,
    n_bootstrap: int,
    alpha: float = 0.95,
) -> Mapping[Tuple[cli_common.RewardCfg, cli_common.RewardCfg], Mapping[str, float]]:
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
        return {"lower": lower, "middle": middle, "upper": upper, "width": upper - lower}

    logger.info("Computing distance")
    return util.cross_distance(x_rets, y_rets, ci_fn, parallelism=1)


def batch_compute_returns(
    trajectory_callable: datasets.TrajectoryCallable,
    models: Mapping[cli_common.RewardCfg, rewards.RewardModel],
    discount: float,
    n_episodes: int,
    batch_episodes: int = 256,
) -> Mapping[cli_common.RewardCfg, np.ndarray]:
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
        trajectories = trajectory_callable(batch_size)
        rets = rewards.compute_return_of_models(models, trajectories, discount)
        for k, v in rets.items():
            returns[k].append(v)
        remainder -= batch_size
    returns = {k: np.concatenate(v) for k, v in returns.items()}
    return returns


@plot_erc_heatmap_ex.capture
def compute_vals(
    env_name: str,
    discount: float,
    x_reward_cfgs: Iterable[cli_common.RewardCfg],
    y_reward_cfgs: Iterable[cli_common.RewardCfg],
    trajectory_factory: datasets.TrajectoryFactory,
    trajectory_factory_kwargs: Dict[str, Any],
    n_episodes: int,
    data_root: str,
) -> Mapping[str, pd.Series]:
    """Entry-point into script to produce divergence heatmaps.

    Args:
        env_name: the name of the environment to plot rewards for.
        discount: the discount rate for shaping.
        x_reward_cfgs: tuples of reward_type and reward_path for x-axis.
        y_reward_cfgs: tuples of reward_type and reward_path for y-axis.
        trajectory_factory: factory to generate trajectories.
        trajectory_factory_kwargs: arguments to pass to the factory.
        n_episodes: the number of episodes to compute correlation over.
        data_root: directory to load learned reward models from.

    Returns:
        A mapping of keywords to Series.
    """
    # Sacred turns our tuples into lists :(, undo
    x_reward_cfgs = cli_common.canonicalize_reward_cfg(x_reward_cfgs, data_root)
    y_reward_cfgs = cli_common.canonicalize_reward_cfg(y_reward_cfgs, data_root)

    logger.info("Loading models")
    g = tf.Graph()
    with g.as_default():
        sess = tf.Session()
        with sess.as_default():
            reward_cfgs = list(x_reward_cfgs) + list(y_reward_cfgs)
            models = cli_common.load_models(env_name, reward_cfgs, discount)

    logger.info("Sampling trajectories")
    with trajectory_factory(**trajectory_factory_kwargs) as trajectory_callable:
        with sess.as_default():
            returns = batch_compute_returns(trajectory_callable, models, discount, n_episodes)

    aggregated = correlation_distance(  # pylint:disable=no-value-for-parameter
        returns, x_reward_cfgs, y_reward_cfgs
    )
    return cli_common.twod_mapping_to_multi_series(aggregated)


cli_common.make_main(plot_erc_heatmap_ex, compute_vals)


if __name__ == "__main__":
    script_utils.experiment_main(plot_erc_heatmap_ex, "plot_erc_heatmap")
