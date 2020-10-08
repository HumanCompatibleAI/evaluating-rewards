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

"""CLI script to fit a model to synthetically generated preferences."""

import functools
from typing import Any, Mapping

from imitation.policies import serialize as policies_serialize
from imitation.util import util
import sacred

from evaluating_rewards import preferences
from evaluating_rewards.rewards import base
from evaluating_rewards.scripts import regress_utils, script_utils

train_preferences_ex = sacred.Experiment("train_preferences")


@train_preferences_ex.config
def default_config():
    """Default configuration values."""
    locals().update(**regress_utils.DEFAULT_CONFIG)
    num_vec = 8  # number of environments in VecEnv

    # Trajectory specification
    policy_type = "random"  # type of policy to generate comparison trajectories
    policy_path = "dummy"  # path to policy
    trajectory_length = 5  # length of trajectories compared

    # Hyperparameters
    model_reward_type = base.MLPRewardModel
    total_timesteps = 5e6  # total number of steps to train for
    batch_timesteps = 10000  # total number of timesteps in each batch
    learning_rate = 1e-2
    weight_l2_reg = 0.0  # scaling factor for weight/parameter regularization
    reward_l2_reg = 1e-4  # scaling factor for regularization of output
    accuracy_threshold = 0.5  # minimum probability in correct direction to count as success

    _ = locals()  # quieten flake8 unused variable warning
    del _


@train_preferences_ex.named_config
def test():
    """Small number of epochs, finish quickly, intended for tests / debugging."""
    total_timesteps = 1e4
    _ = locals()  # quieten flake8 unused variable warning
    del _


script_utils.add_logging_config(train_preferences_ex, "train_preferences")


@train_preferences_ex.main
def train_preferences(
    _seed: int,  # pylint:disable=invalid-name
    # Dataset
    env_name: str,
    discount: float,
    num_vec: int,
    policy_type: str,
    policy_path: str,
    # Target specification
    target_reward_type: str,
    target_reward_path: str,
    # Model parameters
    model_reward_type: regress_utils.EnvRewardFactory,
    trajectory_length: int,
    total_timesteps: int,
    batch_timesteps: int,
    learning_rate: float,
    weight_l2_reg: float,
    reward_l2_reg: float,
    accuracy_threshold: float,
    # Logging
    log_dir: str,
) -> Mapping[str, Any]:
    """Entry-point into script for synthetic preference comparisons."""
    venv = util.make_vec_env(env_name, n_envs=num_vec, seed=_seed)

    make_source = functools.partial(regress_utils.make_model, model_reward_type)

    def make_trainer(model, model_scope, target):
        del target
        model_params = model_scope.global_variables()
        batch_size = batch_timesteps // trajectory_length
        kwargs = {"learning_rate": learning_rate}
        return preferences.PreferenceComparisonTrainer(
            model,
            model_params,
            batch_size=batch_size,
            optimizer_kwargs=kwargs,
            weight_l2_reg=weight_l2_reg,
            reward_l2_reg=reward_l2_reg,
            accuracy_threshold=accuracy_threshold,
        )

    with policies_serialize.load_policy(policy_type, policy_path, venv) as policy:

        def do_training(target, trainer):
            # Specify in terms of total_timesteps so longer trajectory_length
            # does not give model more data.
            total_comparisons = total_timesteps // trajectory_length
            return trainer.fit_synthetic(
                venv,
                policy=policy,
                target=target,
                trajectory_length=trajectory_length,
                total_comparisons=total_comparisons,
            )

        return regress_utils.regress(
            seed=_seed,
            env_name=env_name,
            discount=discount,
            make_source=make_source,
            source_init=True,
            make_trainer=make_trainer,
            do_training=do_training,
            target_reward_type=target_reward_type,
            target_reward_path=target_reward_path,
            log_dir=log_dir,
        )


if __name__ == "__main__":
    script_utils.experiment_main(train_preferences_ex, "train_preferences")
