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

"""CLI script to compare one source model onto a target model."""

import functools
import os
from typing import Any, Dict, Mapping

import sacred
import tensorflow as tf

from evaluating_rewards import comparisons, datasets, serialize
from evaluating_rewards.scripts import regress_utils, script_utils

model_comparison_ex = sacred.Experiment("model_comparison")


@model_comparison_ex.config
def default_config():
    """Default configuration values."""
    locals().update(**regress_utils.DEFAULT_CONFIG)
    dataset_factory = datasets.rollout_serialized_policy_generator
    dataset_factory_kwargs = dict()

    # Model to fit to target
    source_reward_type = "evaluating_rewards/PointMassSparseWithCtrl-v0"
    source_reward_path = "dummy"

    # Model to train and hyperparameters
    model_wrapper_fn = comparisons.equivalence_model_wrapper  # equivalence class
    model_wrapper_kwargs = dict()
    loss_fn = tf.losses.mean_squared_error
    pretrain = True  # set initial scale and constant to match target
    pretrain_size = 16386  # number of timesteps to use in pretraining
    total_timesteps = 1e6
    batch_size = 4096
    learning_rate = 1e-2

    # Logging
    log_root = os.path.join("output", "train_regress")  # output directory
    _ = locals()  # quieten flake8 unused variable warning
    del _


@model_comparison_ex.config
def default_kwargs(dataset_factory, dataset_factory_kwargs):
    """Sets dataset_factory_kwargs to defaults when dataset_factory not overridden."""
    # TODO(): remove this function when Sacred issue #238 is fixed
    if (  # pylint:disable=comparison-with-callable
        dataset_factory == datasets.rollout_serialized_policy_generator
        and not dataset_factory_kwargs
    ):
        dataset_factory_kwargs = dict(policy_type="random", policy_path="dummy")
    _ = locals()  # quieten flake8 unused variable warning
    del _


@model_comparison_ex.named_config
def affine_only():
    """Equivalence class consists of just affine transformations."""
    model_wrapper_fn = comparisons.equivalence_model_wrapper
    model_wrapper_kwargs = dict(potential=False)
    _ = locals()  # quieten flake8 unused variable warning
    del _


@model_comparison_ex.named_config
def no_rescale():
    """Equivalence class are shifts plus potential shaping (no scaling)."""
    model_wrapper_fn = comparisons.equivalence_model_wrapper
    model_wrapper_kwargs = dict(affine_kwargs=dict(scale=False))
    _ = locals()  # quieten flake8 unused variable warning
    del _


@model_comparison_ex.named_config
def shaping_only():
    """Equivalence class consists of just potential shaping."""
    model_wrapper_fn = comparisons.equivalence_model_wrapper
    model_wrapper_kwargs = dict(affine=False)
    pretrain = False
    _ = locals()  # quieten flake8 unused variable warning
    del _


@model_comparison_ex.named_config
def ellp_loss():
    p = 0.5
    # Note if p specified at CLI, it will take priority over p above here
    # (Sacred configuration scope magic).
    loss_fn = functools.partial(comparisons.ellp_norm_loss, p=p)
    _ = locals()  # quieten flake8 unused variable warning
    del _


# TODO(): add a sparsify named config combining ellp_loss, no_rescale
# and Zero target. (Sacred does not currently support combining named configs
# but they're intending to add it.)


@model_comparison_ex.named_config
def fast():
    """Small number of epochs, finish quickly, intended for tests / debugging."""
    pretrain_size = 512
    total_timesteps = 8192
    _ = locals()  # quieten flake8 unused variable warning
    del _


@model_comparison_ex.named_config
def dataset_random_transition():
    """Randomly samples state and action and computes next state from dynamics."""
    dataset_factory = datasets.random_transition_generator
    dataset_factory_kwargs = {}
    _ = locals()  # quieten flake8 unused variable warning
    del _


script_utils.add_logging_config(model_comparison_ex, "model_comparison")


@model_comparison_ex.main
def model_comparison(
    _seed: int,  # pylint:disable=invalid-name
    # Dataset
    env_name: str,
    dataset_factory: datasets.DatasetFactory,
    dataset_factory_kwargs: Dict[str, Any],
    # Source specification
    source_reward_type: str,
    source_reward_path: str,
    # Target specification
    target_reward_type: str,
    target_reward_path: str,
    # Model parameters
    model_wrapper_fn: comparisons.ModelWrapperFn,
    model_wrapper_kwargs: Dict[str, Any],
    pretrain: bool,
    pretrain_size: int,
    total_timesteps: int,
    batch_size: int,
    learning_rate: float,
    # Logging
    log_dir: str,
) -> Mapping[str, Any]:
    """Entry-point into script to regress source onto target reward model."""
    with dataset_factory(env_name, seed=_seed, **dataset_factory_kwargs) as dataset_callable:
        dataset = dataset_callable(total_timesteps, batch_size)

        def make_source(venv):
            return serialize.load_reward(source_reward_type, source_reward_path, venv)

        def make_trainer(model, model_scope, target):
            del model_scope
            model_wrapper = functools.partial(model_wrapper_fn, **model_wrapper_kwargs)
            return comparisons.RegressWrappedModel(
                model, target, model_wrapper=model_wrapper, learning_rate=learning_rate
            )

        def do_training(target, trainer):
            del target
            pretrain_set = None
            if pretrain:
                pretrain_set = next(dataset_callable(pretrain_size, pretrain_size))
            return trainer.fit(dataset, pretrain=pretrain_set)

        return regress_utils.regress(
            seed=_seed,
            env_name=env_name,
            make_source=make_source,
            source_init=False,
            make_trainer=make_trainer,
            do_training=do_training,
            target_reward_type=target_reward_type,
            target_reward_path=target_reward_path,
            log_dir=log_dir,
        )


if __name__ == "__main__":
    script_utils.experiment_main(model_comparison_ex, "model_comparison")
