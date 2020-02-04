# Copyright 2019 DeepMind Technologies Limited and Adam Gleave
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

"""Unit tests for evaluating_rewards.rewards."""

import logging

import gym
import pandas as pd
from stable_baselines.common import vec_env
import tensorflow as tf

# Environments registered as a side-effect of importing
from evaluating_rewards import comparisons, datasets, rewards, serialize
from tests import common

PM_REWARD_TYPES = {
    "ground_truth": {
        "target": "evaluating_rewards/PointMassGroundTruth-v0",
        "loss_ub": 6.5e-3,
        "rel_loss_lb": 10,
    },
    "dense": {
        "target": "evaluating_rewards/PointMassDenseWithCtrl-v0",
        "loss_ub": 4e-2,
        "rel_loss_lb": 10,
    },
    # For sparse and zero, set a low relative error bound, since some
    # random seeds have small scale and so get a low initial loss.
    "sparse": {
        "target": "evaluating_rewards/PointMassSparseWithCtrl-v0",
        "loss_ub": 4e-2,
        "rel_loss_lb": 2,
    },
    "zero": {"target": "evaluating_rewards/Zero-v0", "loss_ub": 2e-4, "rel_loss_lb": 2},
}


@common.mark_parametrize_kwargs(PM_REWARD_TYPES)
def test_regress(
    graph: tf.Graph, session: tf.Session, target: str, loss_ub: float, rel_loss_lb: float
):
    """Test regression onto target.

    Args:
        target: The target reward model type. Must be a hardcoded reward:
            we always load with a path "dummy".
        loss_ub: The maximum loss of the model at the end of training.
        rel_loss_lb: The minimum relative improvement to the initial loss.
    """
    env_name = "evaluating_rewards/PointMassLine-v0"
    venv = vec_env.DummyVecEnv([lambda: gym.make(env_name)])

    with datasets.random_transition_generator(env_name) as dataset_generator:
        with graph.as_default():
            with session.as_default():
                with tf.variable_scope("source") as source_scope:
                    source = rewards.MLPRewardModel(venv.observation_space, venv.action_space)

                with tf.variable_scope("target"):
                    target_model = serialize.load_reward(target, "dummy", venv)

                with tf.variable_scope("match") as match_scope:
                    match = comparisons.RegressModel(source, target_model)

                init_vars = source_scope.global_variables() + match_scope.global_variables()
                session.run(tf.initializers.variables(init_vars))

                stats = match.fit(dataset_generator, total_timesteps=1e5, batch_size=512)

        loss = pd.DataFrame(stats["loss"])["singleton"]
        logging.info(f"Loss: {loss.iloc[::10]}")
        initial_loss = loss.iloc[0]
        logging.info(f"Initial loss: {initial_loss}")
        final_loss = loss.iloc[-10:].mean()
        logging.info(f"Final loss: {final_loss}")

        assert initial_loss / final_loss > rel_loss_lb
        assert final_loss < loss_ub
