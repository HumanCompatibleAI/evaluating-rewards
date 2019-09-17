# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI script to compare one source model onto a target model."""

import functools
import os
from typing import Any, Callable, Dict, Mapping

from absl import app
from evaluating_rewards import comparisons
from evaluating_rewards import serialize
from evaluating_rewards.experiments import datasets
from evaluating_rewards.scripts import regress_utils
from evaluating_rewards.scripts import script_utils
import gym
import sacred
from stable_baselines.common import vec_env

model_comparison_ex = sacred.Experiment("model_comparison")


# pylint:disable=unused-variable
@model_comparison_ex.config
def default_config():
  """Default configuration values."""
  locals().update(**regress_utils.DEFAULT_CONFIG)
  # TODO(): make these configurable outside of Python?
  dataset_factory = datasets.random_generator

  # Model to fit to target
  source_reward_type = "evaluating_rewards/PointMassSparse-v0"
  source_reward_path = "dummy"

  # Model to train and hyperparameters
  model_wrapper_fn = comparisons.equivalence_model_wrapper  # equivalence class
  model_wrapper_kwargs = dict()
  total_timesteps = 1e6
  batch_size = 4096
  learning_rate = 1e-2

  # Logging
  log_root = os.path.join("output", "train_regress")  # output directory


@model_comparison_ex.named_config
def affine_only():
  """Equivalence class consists of just affine transformations."""
  model_wrapper_fn = comparisons.equivalence_model_wrapper
  model_wrapper_kwargs = dict(potential=False)


@model_comparison_ex.named_config
def no_rescale():
  """Equivalence class are shifts plus potential shaping (no scaling)."""
  model_wrapper_fn = comparisons.equivalence_model_wrapper
  model_wrapper_kwargs = dict(scale=False)


@model_comparison_ex.named_config
def shaping_only():
  """Equivalence class consists of just potential shaping."""
  model_wrapper_fn = comparisons.equivalence_model_wrapper
  model_wrapper_kwargs = dict(affine=False)


@model_comparison_ex.named_config
def fast():
  """Small number of epochs, finish quickly, intended for tests / debugging."""
  total_timesteps = 8192
# pylint:enable=unused-variable


script_utils.add_logging_config(model_comparison_ex, "model_comparison")


@model_comparison_ex.main
def model_comparison(_seed: int,  # pylint:disable=invalid-name
                     # Dataset
                     env_name: str,
                     dataset_factory: Callable[[gym.Env],
                                               datasets.BatchCallable],
                     # Source specification
                     source_reward_type: str,
                     source_reward_path: str,
                     # Target specification
                     target_reward_type: str,
                     target_reward_path: str,
                     # Model parameters
                     model_wrapper_fn: comparisons.ModelWrapperFn,
                     model_wrapper_kwargs: Dict[str, Any],
                     total_timesteps: int,
                     batch_size: int,
                     learning_rate: float,
                     # Logging
                     log_dir: str,
                    ) -> Mapping[str, Any]:
  """Entry-point into script to regress source onto target reward model."""
  env = gym.make(env_name)
  venv = vec_env.DummyVecEnv([lambda: env])
  dataset_callable = dataset_factory(env)
  dataset = dataset_callable(total_timesteps, batch_size)

  def make_source(venv):
    return serialize.load_reward(source_reward_type, source_reward_path, venv)

  def make_trainer(model, model_scope, target):
    del model_scope
    model_wrapper = functools.partial(model_wrapper_fn, **model_wrapper_kwargs)
    return comparisons.RegressWrappedModel(model, target,
                                           model_wrapper=model_wrapper,
                                           learning_rate=learning_rate)

  def do_training(target, trainer):
    del target
    return trainer.fit(dataset)

  return regress_utils.regress(seed=_seed,
                               venv=venv,
                               make_source=make_source,
                               make_trainer=make_trainer,
                               do_training=do_training,
                               target_reward_type=target_reward_type,
                               target_reward_path=target_reward_path,
                               log_dir=log_dir)


if __name__ == "__main__":
  main = script_utils.make_main(model_comparison_ex, "model_comparison")
  app.run(main)
