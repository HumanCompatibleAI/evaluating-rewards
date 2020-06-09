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

"""CLI script to plot heatmap of EPIC distance between pairs of reward models."""

import functools
import logging
import os
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from imitation.util import util as imit_util
import numpy as np
import sacred
import tensorflow as tf

from evaluating_rewards import datasets, epic_sample, rewards, tabular, util
from evaluating_rewards.analysis.dissimilarity_heatmaps import cli_common
from evaluating_rewards.scripts import script_utils

plot_canon_heatmap_ex = sacred.Experiment("plot_canon_heatmap")
logger = logging.getLogger("evaluating_rewards.analysis.plot_canon_heatmap")


cli_common.make_config(plot_canon_heatmap_ex)


@plot_canon_heatmap_ex.config
def default_config(env_name, log_root):
    """Default configuration values."""
    data_root = log_root  # root of data directory for learned reward models
    computation_kind = "sample"  # either "sample" or "mesh"
    distance_kind = "pearson"  # either "direct" or "pearson"
    direct_p = 1  # the power to use for direct distance
    discount = 0.99  # discount rate for shaping
    n_seeds = 3

    # n_samples and n_mean_samples only applicable for sample approach
    n_samples = 4096  # number of samples in dataset
    n_mean_samples = 4096  # number of samples to estimate mean
    visitations_factory = None  # defaults to datasets.iid_transition_generator
    visitations_factory_kwargs = {"env_name": env_name}
    dataset_tag = "iid"
    # n_obs and n_act only applicable for mesh approach
    n_obs = 256
    n_act = 256

    # Figure parameters
    heatmap_kwargs = {}
    _ = locals()
    del _


@plot_canon_heatmap_ex.config
def sample_dist_config(env_name):
    """Default sample distribution config: randomly sample from Gym spaces."""
    obs_sample_dist_factory = functools.partial(datasets.sample_dist_from_env_name, obs=True)
    act_sample_dist_factory = functools.partial(datasets.sample_dist_from_env_name, obs=False)
    sample_dist_factory_kwargs = {"env_name": env_name}
    sample_dist_tag = "random_space"  # only used for logging
    _ = locals()
    del _


@plot_canon_heatmap_ex.config
def logging_config(
    env_name, sample_dist_tag, dataset_tag, computation_kind, distance_kind, discount, log_root
):
    """Default logging configuration: hierarchical directory structure based on config."""
    log_dir = os.path.join(  # noqa: F841  pylint:disable=unused-variable
        log_root,
        "plot_canon_heatmap",
        env_name,
        sample_dist_tag,
        dataset_tag,
        computation_kind,
        distance_kind,
        f"discount{discount}",
        imit_util.make_unique_timestamp(),
    )


@plot_canon_heatmap_ex.named_config
def paper():
    """Figures suitable for inclusion in paper.

    By convention we present them to the left, so turn off colorbar.
    """
    styles = ["paper", "heatmap", "heatmap-3col", "heatmap-3col-left", "tex"]
    heatmap_kwargs = {"cbar": False, "vmin": 0.0, "vmax": 1.0}
    _ = locals()
    del _


@plot_canon_heatmap_ex.named_config
def high_precision():
    """Compute tight confidence intervals for publication quality figures.

    Slow and not that much more informative so not worth it for exploratory data analysis.
    """
    n_seeds = 30
    n_samples = 32768
    n_mean_samples = 32768
    n_bootstrap = 10000
    _ = locals()
    del _


SAMPLE_FROM_DATASET_FACTORY = dict(
    obs_sample_dist_factory=functools.partial(
        datasets.transitions_factory_to_sample_dist_factory, obs=True
    ),
    act_sample_dist_factory=functools.partial(
        datasets.transitions_factory_to_sample_dist_factory, obs=False
    ),
)


@plot_canon_heatmap_ex.named_config
def sample_from_serialized_policy():
    """Configure script to sample observations and actions from rollouts of a serialized policy."""
    locals().update(**SAMPLE_FROM_DATASET_FACTORY)
    sample_dist_factory_kwargs = {
        "transitions_factory": datasets.transitions_factory_from_serialized_policy,
        "policy_type": "random",
        "policy_path": "dummy",
    }
    sample_dist_tag = "random_policy"
    _ = locals()
    del _


@plot_canon_heatmap_ex.named_config
def dataset_from_serialized_policy():
    """Configure script to sample batches from rollouts of a serialized policy.

    Only has effect when `computation_kind` equals `"sample"`.
    """
    visitations_factory = datasets.transitions_factory_from_serialized_policy
    visitations_factory_kwargs = {
        "policy_type": "random",
        "policy_path": "dummy",
    }
    dataset_tag = "random_policy"
    _ = locals()
    del _


@plot_canon_heatmap_ex.named_config
def sample_from_random_transitions():
    locals().update(**SAMPLE_FROM_DATASET_FACTORY)
    sample_dist_factory_kwargs = {
        "transitions_factory": datasets.transitions_factory_from_random_model
    }
    sample_dist_tag = "random_transitions"
    _ = locals()
    del _


@plot_canon_heatmap_ex.named_config
def dataset_from_random_transitions():
    visitations_factory = datasets.transitions_factory_from_random_model
    dataset_tag = "random_transitions"
    _ = locals()
    del _


@plot_canon_heatmap_ex.named_config
def test():
    """Intended for debugging/unit test."""
    n_samples = 64
    n_mean_samples = 64
    n_obs = 16
    n_act = 16
    # Do not include "tex" in styles here: this will break on CI.
    styles = ["paper", "heatmap-1col"]
    _ = locals()
    del _


@plot_canon_heatmap_ex.capture
def mesh_canon(
    g: tf.Graph,
    sess: tf.Session,
    obs_dist: datasets.SampleDist,
    act_dist: datasets.SampleDist,
    models: Mapping[cli_common.RewardCfg, rewards.RewardModel],
    x_reward_cfgs: Iterable[cli_common.RewardCfg],
    y_reward_cfgs: Iterable[cli_common.RewardCfg],
    distance_kind: str,
    discount: float,
    n_obs: int,
    n_act: int,
    direct_p: int,
) -> Mapping[Tuple[cli_common.RewardCfg, cli_common.RewardCfg], float]:
    """
    Computes approximation of canon distance by discretizing and then using a tabular method.

    Specifically, we first call `sample_canon_shaping.discrete_iid_evaluate_models` to evaluate
    on a mesh, and then use `tabular.fully_connected_random_canonical_reward` to remove the shaping.

    Args:
        g: the TensorFlow graph.
        sess: the TensorFlow session.
        obs_dist: the distribution over observations.
        act_dist: the distribution over actions.
        models: loaded reward models for all of `x_reward_cfgs` and `y_reward_cfgs`.
        x_reward_cfgs: tuples of reward_type and reward_path for x-axis.
        y_reward_cfgs: tuples of reward_type and reward_path for y-axis.
        distance_kind: the distance to use after deshaping: direct or Pearson.
        discount: the discount rate for shaping.
        n_obs: The number of observations and next observations to use in the mesh.
        n_act: The number of actions to use in the mesh.
        direct_p: When `distance_kind` is "direct", the power used for comparison in the L^p norm.

    Returns:
        Dissimilarity matrix.
    """
    with g.as_default():
        with sess.as_default():
            mesh_rews, _, _ = epic_sample.discrete_iid_evaluate_models(
                models, obs_dist, act_dist, n_obs, n_act
            )
    x_rews = {cfg: mesh_rews[cfg] for cfg in x_reward_cfgs}
    y_rews = {cfg: mesh_rews[cfg] for cfg in y_reward_cfgs}

    if distance_kind == "direct":
        distance_fn = functools.partial(tabular.canonical_reward_distance, p=direct_p)
    elif distance_kind == "pearson":
        distance_fn = tabular.deshape_pearson_distance
    else:
        raise ValueError(f"Unrecognized distance '{distance_kind}'")
    distance_fn = functools.partial(
        distance_fn, discount=discount, deshape_fn=tabular.fully_connected_random_canonical_reward
    )
    logger.info("Computing distance")
    return util.cross_distance(x_rews, y_rews, distance_fn=distance_fn)


def _direct_distance(rewa: np.ndarray, rewb: np.ndarray, p: int) -> float:
    return 0.5 * tabular.direct_distance(rewa, rewb, p=p)


@plot_canon_heatmap_ex.capture
def sample_canon(
    g: tf.Graph,
    sess: tf.Session,
    obs_dist: datasets.SampleDist,
    act_dist: datasets.SampleDist,
    models: Mapping[cli_common.RewardCfg, rewards.RewardModel],
    x_reward_cfgs: Iterable[cli_common.RewardCfg],
    y_reward_cfgs: Iterable[cli_common.RewardCfg],
    distance_kind: str,
    discount: float,
    visitations_factory: Optional[datasets.TransitionsFactory],
    visitations_factory_kwargs: Optional[Dict[str, Any]],
    n_samples: int,
    n_mean_samples: int,
    direct_p: int,
) -> Mapping[Tuple[cli_common.RewardCfg, cli_common.RewardCfg], float]:
    """
    Computes approximation of canon distance using `canonical_sample.sample_canon_shaping`.

    Args:
        g: the TensorFlow graph.
        sess: the TensorFlow session.
        obs_dist: the distribution over observations.
        act_dist: the distribution over actions.
        models: loaded reward models for all of `x_reward_cfgs` and `y_reward_cfgs`.
        x_reward_cfgs: tuples of reward_type and reward_path for x-axis.
        y_reward_cfgs: tuples of reward_type and reward_path for y-axis.
        distance_kind: the distance to use after deshaping: direct or Pearson.
        discount: the discount rate for shaping.
        n_samples: the number of samples to estimate the distance with.
        n_mean_samples: the number of samples to estimate the mean reward for canonicalization.
        direct_p: When `distance_kind` is "direct", the power used for comparison in the L^p norm.

    Returns:
        Dissimilarity matrix.
    """
    del g
    logger.info("Sampling dataset")
    if visitations_factory is None:
        visitations_factory = datasets.transitions_factory_iid_from_sample_dist
        visitations_factory_kwargs = dict(obs_dist=obs_dist, act_dist=act_dist)
    with visitations_factory(**visitations_factory_kwargs) as batch_callable:
        batch = batch_callable(n_samples)

    with sess.as_default():
        logger.info("Removing shaping")
        deshaped_rew = epic_sample.sample_canon_shaping(
            models, batch, act_dist, obs_dist, n_mean_samples, discount, direct_p,
        )
        x_deshaped_rew = {cfg: deshaped_rew[cfg] for cfg in x_reward_cfgs}
        y_deshaped_rew = {cfg: deshaped_rew[cfg] for cfg in y_reward_cfgs}

    if distance_kind == "direct":
        distance_fn = functools.partial(_direct_distance, p=direct_p)
    elif distance_kind == "pearson":
        distance_fn = tabular.pearson_distance
    else:
        raise ValueError(f"Unrecognized distance '{distance_kind}'")

    logger.info("Computing distance")
    return util.cross_distance(x_deshaped_rew, y_deshaped_rew, distance_fn, parallelism=1,)


@plot_canon_heatmap_ex.main
def plot_canon_heatmap(
    env_name: str,
    discount: float,
    x_reward_cfgs: Iterable[cli_common.RewardCfg],
    y_reward_cfgs: Iterable[cli_common.RewardCfg],
    obs_sample_dist_factory: datasets.SampleDistFactory,
    act_sample_dist_factory: datasets.SampleDistFactory,
    sample_dist_factory_kwargs: Dict[str, Any],
    n_seeds: int,
    aggregate_fns: Mapping[str, cli_common.AggregateFn],
    computation_kind: str,
    styles: Iterable[str],
    heatmap_kwargs: Mapping[str, Any],
    log_dir: str,
    data_root: str,
    save_kwargs: Mapping[str, Any],
) -> None:
    """Entry-point into script to produce divergence heatmaps.

    Args:
        env_name: the name of the environment to plot rewards for.
        x_reward_cfgs: tuples of reward_type and reward_path for x-axis.
        y_reward_cfgs: tuples of reward_type and reward_path for y-axis.
        obs_sample_dist_factory: factory to generate sample distribution for observations.
        act_sample_dist_factory: factory to generate sample distribution for actions.
        sample_dist_factory_kwargs: keyword arguments for sample distribution factories.
        n_seeds: the number of independent seeds to take.
        aggregate_fns: Mapping from strings to aggregators to be applied on sequences of floats.
        n_bootstrap: the number of bootstrap samples to take when computing the mean of seeds.
        alpha: percentile confidence interval.
        computation_kind: method to compute results, either "sample" or "mesh" (generally slower).
        styles: styles to apply from `evaluating_rewards.analysis.stylesheets`.
        heatmap_kwargs: passed through to `analysis.compact_heatmaps`.
        log_dir: directory to write figures and other logging to.
        data_root: directory to load learned reward models from.
        save_kwargs: passed through to `analysis.save_figs`.

    Returns:
        A mapping of keywords to figures.
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

    if computation_kind == "sample":
        computation_fn = sample_canon
    elif computation_kind == "mesh":
        computation_fn = mesh_canon
    else:
        raise ValueError(f"Unrecognized computation kind '{computation_kind}'")

    dissimilarities = {}
    for i in range(n_seeds):
        logger.info(f"Seed {i}")
        with obs_sample_dist_factory(**sample_dist_factory_kwargs) as obs_dist:
            with act_sample_dist_factory(**sample_dist_factory_kwargs) as act_dist:
                dissimilarity = computation_fn(
                    g, sess, obs_dist, act_dist, models, x_reward_cfgs, y_reward_cfgs
                )
                for k, v in dissimilarity.items():
                    dissimilarities.setdefault(k, []).append(v)

    vals = {}
    for name, aggregate_fn in aggregate_fns.items():
        logger.info(f"Aggregating {name}")
        aggregated = {k: aggregate_fn(v) for k, v in dissimilarities.items()}
        aggregated = cli_common.twod_mapping_to_multi_series(aggregated)
        vals.update({f"{name}_{k}": v for k, v in aggregated.items()})

    cli_common.save_artifacts(vals, styles, log_dir, heatmap_kwargs, save_kwargs)


if __name__ == "__main__":
    script_utils.experiment_main(plot_canon_heatmap_ex, "plot_canon_heatmap")
