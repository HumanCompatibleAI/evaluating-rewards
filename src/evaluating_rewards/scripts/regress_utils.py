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

"""Common training boilerplate shared between CLI scripts."""

import os
import pickle
from typing import Callable, TypeVar

import gym
from imitation import util
from stable_baselines.common import vec_env
import tensorflow as tf

from evaluating_rewards import rewards, serialize

T = TypeVar("T")
V = TypeVar("V")

EnvRewardFactory = Callable[[gym.Space, gym.Space], rewards.RewardModel]


DEFAULT_CONFIG = {
    "env_name": "evaluating_rewards/PointMassLine-v0",
    "discount": 0.99,
    "target_reward_type": "evaluating_rewards/Zero-v0",
    "target_reward_path": None,
    "model_reward_type": rewards.MLPRewardModel,
}


def logging_config(log_root, env_name):
    log_dir = os.path.join(log_root, env_name.replace("/", "_"), util.make_unique_timestamp())
    _ = locals()  # quieten flake8 unused variable warning
    del _


MakeModelFn = Callable[[vec_env.VecEnv], T]
MakeTrainerFn = Callable[[rewards.RewardModel, tf.VariableScope, rewards.RewardModel], T]
DoTrainingFn = Callable[[rewards.RewardModel, T], V]


def make_model(model_reward_type: EnvRewardFactory, venv: vec_env.VecEnv) -> rewards.RewardModel:
    return model_reward_type(venv.observation_space, venv.action_space)


def regress(
    seed: int,
    env_name: str,
    discount: float,
    make_source: MakeModelFn,
    source_init: bool,
    make_trainer: MakeTrainerFn,
    do_training: DoTrainingFn,
    target_reward_type: str,
    target_reward_path: str,
    log_dir: str,
) -> V:
    """Train a model on target and save the results, reporting training stats."""
    # This venv is needed by serialize.load_reward, but is never stepped.
    venv = vec_env.DummyVecEnv([lambda: gym.make(env_name)])

    with util.make_session() as (_, sess):
        tf.random.set_random_seed(seed)

        with tf.variable_scope("source") as model_scope:
            model = make_source(venv)

        with tf.variable_scope("target"):
            target = serialize.load_reward(target_reward_type, target_reward_path, venv, discount)

        with tf.variable_scope("train") as train_scope:
            trainer = make_trainer(model, model_scope, target)

        # Do not initialize any variables from target, which have already been
        # set during serialization.
        init_vars = train_scope.global_variables()
        if source_init:
            init_vars += model_scope.global_variables()
        sess.run(tf.initializers.variables(init_vars))

        stats = do_training(target, trainer)

        # Trainer may wrap source, so save trainer.source not source directly
        # (see e.g. RegressWrappedModel).
        trainer.model.save(os.path.join(log_dir, "model"))

        with open(os.path.join(log_dir, "stats.pkl"), "wb") as f:
            pickle.dump(stats, f)

    return stats
