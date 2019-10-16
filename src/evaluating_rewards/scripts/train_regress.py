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

"""CLI script to regress a model onto another, pre-loaded model."""

import functools
from typing import Any, Dict, Mapping

from absl import app
import sacred

from evaluating_rewards import comparisons, rewards
from evaluating_rewards.experiments import datasets
from evaluating_rewards.scripts import regress_utils, script_utils

train_regress_ex = sacred.Experiment("train_regress")


@train_regress_ex.config
def default_config():
    """Default configuration values."""
    locals().update(**regress_utils.DEFAULT_CONFIG)
    dataset_factory = datasets.rollout_serialized_policy_generator
    dataset_factory_kwargs = dict()

    # Model to train and hyperparameters
    model_reward_type = rewards.MLPRewardModel
    total_timesteps = 1e6
    batch_size = 4096
    learning_rate = 1e-2
    _ = locals()  # quieten flake8 unused variable warning
    del _


@train_regress_ex.config
def default_kwargs(dataset_factory, dataset_factory_kwargs):
    # TODO(): remove this function when Sacred issue #238 is fixed
    if (
        dataset_factory == datasets.rollout_serialized_policy_generator  # pylint:disable=comparison-with-callable
        and not dataset_factory_kwargs
    ):
        dataset_factory_kwargs = dict(policy_type="random", policy_path="dummy")
    _ = locals()  # quieten flake8 unused variable warning
    del _


@train_regress_ex.named_config
def fast():
    """Small number of epochs, finish quickly, intended for tests / debugging."""
    total_timesteps = 8192  # noqa: F841  pylint:disable=unused-variable


@train_regress_ex.named_config
def dataset_random_transition():
    """Randomly samples state and action and computes next state from dynamics."""
    dataset_factory = datasets.random_transition_generator  # noqa: F841  pylint:disable=unused-variable


script_utils.add_logging_config(train_regress_ex, "train_regress")


@train_regress_ex.main
def train_regress(
    _seed: int,  # pylint:disable=invalid-name
    # Dataset
    env_name: str,
    dataset_factory: datasets.DatasetFactory,
    dataset_factory_kwargs: Dict[str, Any],
    # Target specification
    target_reward_type: str,
    target_reward_path: str,
    # Model parameters
    model_reward_type: regress_utils.EnvRewardFactory,
    total_timesteps: int,
    batch_size: int,
    learning_rate: float,
    # Logging
    log_dir: str,
) -> Mapping[str, Any]:
    """Entry-point into script to regress source onto target reward model."""
    with dataset_factory(env_name, seed=_seed, **dataset_factory_kwargs) as dataset_callable:
        dataset = dataset_callable(total_timesteps, batch_size)

        make_source = functools.partial(regress_utils.make_model, model_reward_type)

        def make_trainer(model, model_scope, target):
            del model_scope
            return comparisons.RegressModel(model, target, learning_rate=learning_rate)

        def do_training(target, trainer):
            del target
            return trainer.fit(dataset)

        return regress_utils.regress(
            seed=_seed,
            env_name=env_name,
            make_source=make_source,
            source_init=True,
            make_trainer=make_trainer,
            do_training=do_training,
            target_reward_type=target_reward_type,
            target_reward_path=target_reward_path,
            log_dir=log_dir,
        )


if __name__ == "__main__":
    main = script_utils.make_main(train_regress_ex, "train_regress")
    app.run(main)
