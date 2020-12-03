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

"""CLI script to compute EPIC distance between pairs of reward models."""

import functools
import itertools
import logging
import os
import pickle
from typing import Any, Callable, Dict, Iterable, Mapping, Tuple

from imitation.util import util as imit_util
import numpy as np
import ray
import sacred
import tensorflow as tf

from evaluating_rewards import datasets
from evaluating_rewards.analysis import util
from evaluating_rewards.distances import common_config, epic_sample, tabular
from evaluating_rewards.rewards import base
from evaluating_rewards.scripts import script_utils
from evaluating_rewards.scripts.distances import common

Dissimilarity = Mapping[Tuple[common_config.RewardCfg, common_config.RewardCfg], float]

epic_distance_ex = sacred.Experiment("epic_distance")
logger = logging.getLogger("evaluating_rewards.scripts.distances.epic")


common.make_config(epic_distance_ex)
common.make_transitions_configs(epic_distance_ex)


@epic_distance_ex.config
def default_config():
    """Default configuration values."""
    ray_kwargs = {}  # arguments to Ray, used for parallelization
    num_cpus = 4  # how many CPUs needed per seed
    computation_kind = "sample"  # either "sample" or "mesh"
    distance_kind = "pearson"  # either "direct" or "pearson"
    direct_p = 1  # the power to use for direct distance
    discount = 0.99  # discount rate for shaping
    n_seeds = 3

    # n_samples and n_mean_samples only applicable for sample approach
    n_samples = 4096  # number of samples in dataset
    n_mean_samples = 4096  # number of samples to estimate mean
    sample_dist_factory_kwargs = {}
    # n_obs and n_act only applicable for mesh approach
    n_obs = 256
    n_act = 256

    _ = locals()
    del _


@epic_distance_ex.config
def sample_dist_config(sample_dist_factory_kwargs, visitations_factory, visitations_factory_kwargs):
    """Default sample distribution config: marginalize from visitation factory."""
    obs_sample_dist_factory = functools.partial(
        datasets.transitions_factory_to_sample_dist_factory, obs=True
    )
    act_sample_dist_factory = functools.partial(
        datasets.transitions_factory_to_sample_dist_factory, obs=False
    )
    if not sample_dist_factory_kwargs:
        sample_dist_factory_kwargs = dict(visitations_factory_kwargs)
        sample_dist_factory_kwargs["transitions_factory"] = visitations_factory
    obs_sample_dist_factory_kwargs = {}
    act_sample_dist_factory_kwargs = {}
    sample_dist_tag = "marginalized"  # only used for logging
    _ = locals()
    del _


@epic_distance_ex.config
def logging_config(
    env_name, sample_dist_tag, dataset_tag, computation_kind, distance_kind, discount, log_root
):
    """Default logging configuration: hierarchical directory structure based on config."""
    log_dir = os.path.join(  # noqa: F841  pylint:disable=unused-variable
        log_root,
        "epic",
        env_name,
        sample_dist_tag,
        dataset_tag,
        computation_kind,
        distance_kind,
        f"discount{discount}",
        imit_util.make_unique_timestamp(),
    )


@epic_distance_ex.named_config
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


FAST_CONFIG = dict(
    ray_kwargs={
        # CI build only has 1 core per test
        "num_cpus": 1,
    },
    num_cpus=1,
    n_samples=64,
    n_mean_samples=64,
    n_obs=16,
    n_act=16,
    n_seeds=2,
)


@epic_distance_ex.named_config
def fast():
    """Intended for debugging."""
    locals().update(**FAST_CONFIG)


@epic_distance_ex.named_config
def test():
    """Unit test config."""
    locals().update(**FAST_CONFIG)
    visitations_factory_kwargs = {  # noqa: F841  pylint:disable=unused-variable
        "env_name": "evaluating_rewards/PointMassLine-v0",
        "parallel": False,
        "policy_type": "random",
        "policy_path": "dummy",
    }


def mesh_canon(
    g: tf.Graph,
    sess: tf.Session,
    seed: int,
    obs_dist: datasets.SampleDist,
    act_dist: datasets.SampleDist,
    models: Mapping[common_config.RewardCfg, base.RewardModel],
    x_reward_cfgs: Iterable[common_config.RewardCfg],
    y_reward_cfgs: Iterable[common_config.RewardCfg],
    distance_kind: str,
    discount: float,
    n_obs: int,
    n_act: int,
    direct_p: int,
) -> Dissimilarity:
    """
    Computes approximation of canon distance by discretizing and then using a tabular method.

    Specifically, we first call `sample_canon_shaping.discrete_iid_evaluate_models` to evaluate
    on a mesh, and then use `tabular.fully_connected_random_canonical_reward` to remove the shaping.

    In expectation this method should be equivalent to `sample_canon` when the visitation
    distribution is IID samples from `obs_dist` and `act_dist`.

    Args:
        g: the TensorFlow graph.
        sess: the TensorFlow session.
        seed: (unused) the seed for any stochastic computation.
        obs_dist: the distribution over observations.
        act_dist: the distribution over actions.
        models: loaded reward models for all of `x_reward_cfgs` and `y_reward_cfgs`.
        x_reward_cfgs: tuples of reward_type and reward_path for x-axis.
        y_reward_cfgs: tuples of reward_type and reward_path for y-axis.
        distance_kind: the distance to use after deshaping: "direct" or "pearson".
        discount: the discount rate for shaping.
        n_obs: The number of observations and next observations to use in the mesh.
        n_act: The number of actions to use in the mesh.
        direct_p: When `distance_kind` is "direct", the power used for comparison in the L^p norm.

    Returns:
        Dissimilarity matrix.
    """
    del seed

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


def sample_canon(
    g: tf.Graph,
    sess: tf.Session,
    seed: int,
    obs_dist: datasets.SampleDist,
    act_dist: datasets.SampleDist,
    models: Mapping[common_config.RewardCfg, base.RewardModel],
    x_reward_cfgs: Iterable[common_config.RewardCfg],
    y_reward_cfgs: Iterable[common_config.RewardCfg],
    distance_kind: str,
    discount: float,
    # TODO(adam): add back type annotations for factories once GH cloudpickle#399 is closed
    visitations_factory,
    visitations_factory_kwargs: Dict[str, Any],
    n_samples: int,
    n_mean_samples: int,
    direct_p: int,
) -> Dissimilarity:
    """
    Computes approximation of canon distance using `canonical_sample.sample_canon_shaping`.

    Args:
        g: the TensorFlow graph.
        sess: the TensorFlow session.
        seed: the seed for `visitations_factory`.
        obs_dist: the distribution over observations.
        act_dist: the distribution over actions.
        models: loaded reward models for all of `x_reward_cfgs` and `y_reward_cfgs`.
        x_reward_cfgs: tuples of reward_type and reward_path for x-axis.
        y_reward_cfgs: tuples of reward_type and reward_path for y-axis.
        distance_kind: the distance to use after deshaping: "direct" or "pearson".
        discount: the discount rate for shaping.
        n_samples: the number of samples to estimate the distance with.
        n_mean_samples: the number of samples to estimate the mean reward for canonicalization.
        direct_p: When `distance_kind` is "direct", the power used for comparison in the L^p norm.
        visitations_factory: sample transitions from this factory for the outermost expectation.
        visitations_factory_kwargs: keyword arguments to pass through to factory.

    Returns:
        Dissimilarity matrix.
    """
    del g
    logger.info("Sampling dataset")
    with visitations_factory(seed=seed, **visitations_factory_kwargs) as batch_callable:
        batch = batch_callable(n_samples)

    next_obs_samples = obs_dist(n_mean_samples)
    act_samples = act_dist(n_mean_samples)

    with sess.as_default():
        logger.info("Removing shaping")
        deshaped_rew = epic_sample.sample_canon_shaping(
            models,
            batch,
            act_samples,
            next_obs_samples,
            discount,
            direct_p,
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
    return util.cross_distance(x_deshaped_rew, y_deshaped_rew, distance_fn, parallelism=1)


@ray.remote
def epic_worker(
    computation_fn: Callable[..., Dissimilarity],
    seed: int,
    env_name: str,
    x_reward_cfgs: Iterable[common_config.RewardCfg],
    y_reward_cfgs: Iterable[common_config.RewardCfg],
    # TODO(adam): add back type annotations for factories once GH cloudpickle#399 is closed
    obs_sample_dist_factory,
    act_sample_dist_factory,
    sample_dist_factory_kwargs: Dict[str, Any],
    discount: float,
    distance_kind: str,
    direct_p: int,
) -> Dissimilarity:
    """Computes EPIC distance for a single seed `seed`.

    Args:
        computation_fn: Function to compute, one of `mesh_canon` or `sample_canon`.
        seed: seed for sampling.
        env_name: the name of the environment to compare rewards for.
        x_reward_cfgs: tuples of reward_type and reward_path for x-axis.
        y_reward_cfgs: tuples of reward_type and reward_path for y-axis.
        obs_sample_dist_factory: factory to generate sample distribution for observations.
        act_sample_dist_factory: factory to generate sample distribution for actions.
        sample_dist_factory_kwargs: keyword arguments for sample distribution factories.
        discount: discount to use for reward models (mostly for shaping).
        distance_kind: the distance to use after deshaping: "direct" or "pearson".
        direct_p: When `distance_kind` is "direct", the power used for comparison in the L^p norm.
    """
    # Configure logging, since Ray children do not by default inherit logging configs.
    script_utils.configure_logging()

    models, g, sess = common.load_models_create_sess(
        env_name, discount, itertools.chain(x_reward_cfgs, y_reward_cfgs)
    )

    logger.info(f"Seed {seed}")
    with obs_sample_dist_factory(seed=seed, **sample_dist_factory_kwargs) as obs_dist:
        with act_sample_dist_factory(seed=seed, **sample_dist_factory_kwargs) as act_dist:
            return computation_fn(  # pylint:disable=no-value-for-parameter
                g=g,
                sess=sess,
                seed=seed,
                obs_dist=obs_dist,
                act_dist=act_dist,
                models=models,
                x_reward_cfgs=x_reward_cfgs,
                y_reward_cfgs=y_reward_cfgs,
                distance_kind=distance_kind,
                direct_p=direct_p,
                discount=discount,
            )


@epic_distance_ex.capture
def compute_vals(
    # Internal arguments (not passed to computation_fn or epic_worker)
    ray_kwargs: Mapping[str, Any],
    num_cpus: int,
    computation_kind: str,
    aggregate_fns: Mapping[str, common.AggregateFn],
    log_dir: str,
    n_seeds: int,
    # Common arguments (passed to epic_worker)
    env_name: str,
    x_reward_cfgs: Iterable[common_config.RewardCfg],
    y_reward_cfgs: Iterable[common_config.RewardCfg],
    obs_sample_dist_factory: datasets.SampleDistFactory,
    act_sample_dist_factory: datasets.SampleDistFactory,
    sample_dist_factory_kwargs: Dict[str, Any],
    discount: float,
    distance_kind: str,
    direct_p: int,
    # Specific to `sample_canon`
    visitations_factory: datasets.TransitionsFactory,
    visitations_factory_kwargs: Dict[str, Any],
    n_samples: int,
    n_mean_samples: int,
    # Specific to `mesh_canon`
    n_obs: int,
    n_act: int,
) -> common_config.AggregatedDistanceReturn:
    """Computes values for dissimilarity heatmaps.

    Args:
        ray_kwargs: Passed through to `ray.init`.
        num_cpus: number of CPUs needed to execute each seed.
        computation_kind: method to compute results, either "sample" or "mesh" (generally slower).
        aggregate_fns: Mapping from strings to aggregators to be applied on sequences of floats.
        log_dir: directory to save data to.
        n_seeds: the number of independent seeds to take.
        env_name: the name of the environment to compare rewards for.
        x_reward_cfgs: tuples of reward_type and reward_path for x-axis.
        y_reward_cfgs: tuples of reward_type and reward_path for y-axis.
        obs_sample_dist_factory: factory to generate sample distribution for observations.
        act_sample_dist_factory: factory to generate sample distribution for actions.
        sample_dist_factory_kwargs: keyword arguments for sample distribution factories.
        discount: discount to use for reward models (mostly for shaping).
        distance_kind: the distance to use after deshaping: "direct" or "pearson".
        direct_p: When `distance_kind` is "direct", the power used for comparison in the L^p norm.
        visitations_factory: (sample only) sample transitions from this factory for the
            outermost expectation.
        visitations_factory_kwargs: (sample only) keyword arguments to pass through to factory.
        n_samples: (sample only) the number of samples to estimate the distance with.
        n_mean_samples: (sample only) the number of samples to estimate the mean reward
            for canonicalization.
        n_obs: (mesh only) The number of observations and next observations to use in the mesh.
        n_act: (mesh only) The number of actions to use in the mesh.

    Returns:
        Nested dictionary of aggregated distance values.
    """
    if computation_kind == "sample":
        computation_fn = functools.partial(
            sample_canon,
            visitations_factory=visitations_factory,
            visitations_factory_kwargs=visitations_factory_kwargs,
            n_samples=n_samples,
            n_mean_samples=n_mean_samples,
        )
    elif computation_kind == "mesh":
        computation_fn = functools.partial(mesh_canon, n_obs=n_obs, n_act=n_act)
    else:
        raise ValueError(f"Unrecognized computation kind '{computation_kind}'")

    ray.init(**ray_kwargs)

    refs = []
    try:
        epic_worker_tagged = epic_worker.options(num_cpus=num_cpus)
        for seed in range(n_seeds):
            ref = epic_worker_tagged.remote(
                computation_fn=computation_fn,
                seed=seed,
                env_name=env_name,
                discount=discount,
                x_reward_cfgs=x_reward_cfgs,
                y_reward_cfgs=y_reward_cfgs,
                obs_sample_dist_factory=obs_sample_dist_factory,
                act_sample_dist_factory=act_sample_dist_factory,
                sample_dist_factory_kwargs=sample_dist_factory_kwargs,
                distance_kind=distance_kind,
                direct_p=direct_p,
            )
            refs.append(ref)
        results = ray.get(refs)
    finally:
        ray.shutdown()

    dissimilarities = {}
    for dissimilarity in results:
        for k, v in dissimilarity.items():
            dissimilarities.setdefault(k, []).append(v)

    logger.info("Saving raw dissimilarities")
    with open(os.path.join(log_dir, "dissimilarities.pkl"), "wb") as f:
        pickle.dump(dissimilarities, f)

    return common.aggregate_seeds(aggregate_fns, dissimilarities)


common.make_main(epic_distance_ex, compute_vals)

if __name__ == "__main__":
    script_utils.experiment_main(epic_distance_ex, "epic_distance")
