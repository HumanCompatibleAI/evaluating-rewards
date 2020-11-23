# Copyright 2019, 2020 DeepMind Technologies Limited and Adam Gleave
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

"""CLI script to compare one source model onto a target model."""

import copy
import functools
import logging
import os
import pickle
from typing import Any, Dict, Iterable, Mapping, Type

import numpy as np
import ray
import sacred

from evaluating_rewards import datasets, serialize
from evaluating_rewards.distances import common_config, npec
from evaluating_rewards.rewards import base, comparisons
from evaluating_rewards.scripts import regress_utils, script_utils
from evaluating_rewards.scripts.distances import common

npec_distance_ex = sacred.Experiment("npec_distance")
logger = logging.getLogger("evaluating_rewards.scripts.distances.npec")

common.make_config(npec_distance_ex)
common.make_transitions_configs(npec_distance_ex)

ZERO_CFG = (serialize.ZERO_REWARD, "dummy")


@npec_distance_ex.config
def default_config():
    """Default configuration values."""
    # Parallelization
    ray_kwargs = {}
    num_cpus = 2

    # Aggregation
    n_seeds = 3
    normalize = True  # divide by distance from Zero reward, an upper bound on the distance

    # Model to train and hyperparameters
    model_reward_type = base.MLPRewardModel
    discount = 0.99
    comparison_class = npec.RegressWrappedModel
    comparison_kwargs = {
        "learning_rate": 1e-2,
    }
    total_timesteps = int(1e6)
    batch_size = 4096
    fit_kwargs = {
        "affine_size": 16384,  # number of timesteps to use in pretraining; set to None to disable
    }

    _ = locals()  # quieten flake8 unused variable warning
    del _


@npec_distance_ex.config
def default_kwargs(comparison_class, comparison_kwargs):
    """Sets comparison_kwargs to defaults when not overridden."""
    # TODO(): remove this function when Sacred issue #238 is fixed
    if comparison_class == npec.RegressWrappedModel and "model_wrapper" not in comparison_kwargs:
        comparison_kwargs["model_wrapper"] = npec.equivalence_model_wrapper
    _ = locals()  # quieten flake8 unused variable warning
    del _


@npec_distance_ex.named_config
def alternating_maximization():
    """Use less flexible (but sometimes more accurate) RegressEquivalentLeastSq.

    Uses least-squares loss and affine + potential shaping wrapping.
    """
    comparison_class = npec.RegressEquivalentLeastSqModel
    _ = locals()  # quieten flake8 unused variable warning
    del _


@npec_distance_ex.named_config
def affine_only():
    """Equivalence class consists of just affine transformations."""
    comparison_kwargs = {  # noqa: F841  pylint:disable=unused-variable
        "model_wrapper": functools.partial(npec.equivalence_model_wrapper, potential=False),
    }


@npec_distance_ex.named_config
def no_rescale():
    """Equivalence class are shifts plus potential shaping (no scaling)."""
    comparison_kwargs = {  # noqa: F841  pylint:disable=unused-variable
        "model_wrapper": functools.partial(
            npec.equivalence_model_wrapper,
            affine_kwargs=dict(scale=False),
        ),
    }


@npec_distance_ex.named_config
def shaping_only():
    """Equivalence class consists of just potential shaping."""
    comparison_kwargs = {
        "model_wrapper": functools.partial(npec.equivalence_model_wrapper, affine=False),
    }
    fit_kwargs = {"affine_size": None}
    _ = locals()  # quieten flake8 unused variable warning
    del _


@npec_distance_ex.named_config
def ellp_loss():
    """Use mean (x-y)^p loss, default to p=0.5 (sparsity inducing)"""
    p = 0.5
    # Note if p specified at CLI, it will take priority over p above here
    # (Sacred configuration scope magic).
    comparison_kwargs = {
        "loss_fn": functools.partial(comparisons.ellp_norm_loss, p=p),
    }
    _ = locals()  # quieten flake8 unused variable warning
    del _


# TODO(): add a sparsify named config combining ellp_loss, no_rescale
# and Zero target. (Sacred does not currently support combining named configs
# but they're intending to add it.)


@npec_distance_ex.named_config
def test():
    """Small number of epochs, finish quickly, intended for tests / debugging."""
    n_seeds = 1
    # Disable studentt_ci and sample_mean_sd since they need >1 seed (this test is already slow)
    aggregate_kinds = ("bootstrap",)
    fit_kwargs = {"affine_size": 512}
    ray_kwargs = {
        # CI build only has 1 core per test
        "num_cpus": 1,
    }
    num_cpus = 1
    visitations_factory_kwargs = {
        "env_name": "evaluating_rewards/PointMassLine-v0",
        "parallel": False,
        "policy_type": "random",
        "policy_path": "dummy",
    }
    batch_size = 512
    total_timesteps = 2048
    _ = locals()  # quieten flake8 unused variable warning
    del _


@npec_distance_ex.named_config
def high_precision():
    """Increase number of timesteps to increase change of convergence."""
    total_timesteps = int(10e6)  # noqa: F841  pylint:disable=unused-variable


script_utils.add_logging_config(npec_distance_ex, "npec")


@ray.remote
def npec_worker(
    seed: int,
    # Dataset
    env_name: str,
    discount: float,
    visitations_factory,
    visitations_factory_kwargs: Dict[str, Any],
    # Models to compare
    source_reward_cfg: common_config.RewardCfg,
    target_reward_cfg: common_config.RewardCfg,
    # Model parameters
    comparison_class: Type[comparisons.RegressModel],
    comparison_kwargs: Dict[str, Any],
    total_timesteps: int,
    batch_size: int,
    fit_kwargs: Dict[str, Any],
    log_dir: str,
) -> comparisons.FitStats:
    """Performs a single NPEC comparison by fitting a model.

    Args:
        seed: Seed used for visitation factory and model initialization.
        env_name: the name of the environment to compare rewards for.
        discount: discount to use for reward models (mostly for shaping).
        visitations_factory: factory to sample transitions from during training.
        visitations_factory_kwargs: keyword arguments for the visitations factory.
        source_reward_cfg: specifies the serialized source reward.
        target_reward_cfg: specifies the serialized target reward to fit the source onto.
        comparison_class: how to fit the source onto target.
        comparison_kwargs: keyword arguments customizing `comparison_class`.
        total_timesteps: the total number of timesteps to train for.
        batch_size: the number of timesteps in each training batch.
        fit_kwargs: extra arguments to pass to the `fit` method of `comparison_class`.
        log_dir: directory to save data to.

    Returns:
        Statistics for training, including the final loss aka estimated NPEC distance.
    """
    with visitations_factory(seed=seed, **visitations_factory_kwargs) as dataset_generator:

        def make_source(venv):
            kind, path = source_reward_cfg
            return serialize.load_reward(kind, path, venv, discount)

        def make_trainer(model, model_scope, target):
            del model_scope
            return comparison_class(model, target, **comparison_kwargs)

        def do_training(target, trainer):
            del target
            return trainer.fit(
                dataset_generator,
                total_timesteps=total_timesteps,
                batch_size=batch_size,
                **fit_kwargs,
            )

        target_reward_type, target_reward_path = target_reward_cfg
        return regress_utils.regress(
            seed=seed,
            env_name=env_name,
            discount=discount,
            make_source=make_source,
            source_init=False,
            make_trainer=make_trainer,
            do_training=do_training,
            target_reward_type=target_reward_type,
            target_reward_path=target_reward_path,
            log_dir=log_dir,
        )


@npec_distance_ex.capture
def compute_npec(  # pylint:disable=unused-argument
    num_cpus: int,
    seed: int,
    # Dataset
    env_name: str,
    discount: float,
    visitations_factory: datasets.TransitionsFactory,
    visitations_factory_kwargs: Dict[str, Any],
    # Models to compare
    source_reward_cfg: common_config.RewardCfg,
    target_reward_cfg: common_config.RewardCfg,
    # Model parameters
    comparison_class: Type[comparisons.RegressModel],
    comparison_kwargs: Dict[str, Any],
    total_timesteps: int,
    batch_size: int,
    fit_kwargs: Dict[str, Any],
    log_dir: str,
) -> ray.ObjectRef:
    """Compute NPEC comparison between reward models in the background.

    Thin wrapper around `npec_worker`; arguments are the same as `npec_worker`.
    This sets up hierarchical log directory structure (sanitizing paths using
    `sanitize_path`) and copies arguments to make them pickleable. It also is
    declared as a Sacred capture function, which `npec_worker` cannot be since
    Sacred experiments are not picklable.
    """
    # deepcopy to workaround sacred issue GH#499
    visitations_factory_kwargs = copy.deepcopy(visitations_factory_kwargs)
    comparison_kwargs = copy.deepcopy(comparison_kwargs)
    fit_kwargs = copy.deepcopy(fit_kwargs)
    log_dir = os.path.join(
        log_dir,
        script_utils.sanitize_path(source_reward_cfg),
        script_utils.sanitize_path(target_reward_cfg),
        str(seed),
    )

    params = locals()
    del params["num_cpus"]
    return npec_worker.options(num_cpus=num_cpus).remote(**params)


@npec_distance_ex.capture
def compute_vals(
    ray_kwargs: Mapping[str, Any],
    n_seeds: int,
    aggregate_fns: Mapping[str, common.AggregateFn],
    log_dir: str,
    x_reward_cfgs: Iterable[common_config.RewardCfg],
    y_reward_cfgs: Iterable[common_config.RewardCfg],
    normalize: bool,
) -> common_config.AggregatedDistanceReturn:
    """Entry-point into script to regress source onto target reward model."""
    if normalize:
        x_reward_cfgs = list(x_reward_cfgs) + [ZERO_CFG]

    ray.init(**ray_kwargs)

    try:
        keys = []
        refs = []
        for target in x_reward_cfgs:
            for source in y_reward_cfgs:
                for seed in range(n_seeds):
                    obj_ref = compute_npec(  # pylint:disable=no-value-for-parameter
                        seed=seed, source_reward_cfg=source, target_reward_cfg=target
                    )
                    keys.append((target, source))
                    refs.append(obj_ref)
        values = ray.get(refs)
    finally:
        ray.shutdown()

    stats = {}
    for k, v in zip(keys, values):
        stats.setdefault(k, []).append(v)

    logger.info("Saving raw statistics")
    with open(os.path.join(log_dir, "stats.pkl"), "wb") as f:
        pickle.dump(stats, f)

    dissimilarities = {k: [v["loss"][-1]["singleton"] for v in s] for k, s in stats.items()}
    if normalize:
        mean = {k: np.mean(v) for k, v in dissimilarities.items()}
        normalized = {}
        for k, v in dissimilarities.items():
            target, source = k
            if target == ZERO_CFG:
                continue
            zero_mean = mean[(ZERO_CFG, source)]
            normalized[k] = [x / zero_mean for x in v]
        dissimilarities = normalized

    return common.aggregate_seeds(aggregate_fns, dissimilarities)


common.make_main(npec_distance_ex, compute_vals)


if __name__ == "__main__":
    script_utils.experiment_main(npec_distance_ex, "npec_distance")
