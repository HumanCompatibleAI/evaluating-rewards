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

"""Unit tests for evaluating_rewards.rewards."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from evaluating_rewards import comparisons
from evaluating_rewards import rewards
from evaluating_rewards import serialize
from evaluating_rewards.experiments import datasets
from tests import common
import gym
import pandas as pd
from stable_baselines.common import vec_env
import tensorflow as tf


PM_REWARD_TYPES = {
    "ground_truth": {
        "target": "evaluating_rewards/PointMassGroundTruth-v0",
        "loss_ub": 5e-3,
    },
    "sparse": {
        "target": "evaluating_rewards/PointMassSparse-v0",
        "loss_ub": 4e-2,
    },
    "dense": {
        "target": "evaluating_rewards/PointMassDense-v0",
        "loss_ub": 4e-2,
    },
    "zero": {
        "target": "evaluating_rewards/Zero-v0",
        "loss_ub": 1e-4,
    }
}


class RewardTest(common.TensorFlowTestCase):
  """Unit tests for evaluating_rewards.rewards."""

  @parameterized.named_parameters(common.combine_dicts_as_kwargs(
      PM_REWARD_TYPES
  ))
  def test_regress(self, target: str, loss_ub: float):
    env = gym.make("evaluating_rewards/PointMassLineFixedHorizon-v0")
    venv = vec_env.DummyVecEnv([lambda: env])

    dataset_generator = datasets.random_generator(env)
    dataset = dataset_generator(1e5, 128)

    with self.graph.as_default():
      with self.sess.as_default():
        with tf.variable_scope("source") as source_scope:
          source = rewards.MLPRewardModel(env.observation_space,
                                          env.action_space)

        with tf.variable_scope("target"):
          target_model = serialize.load_reward(target, "dummy", venv)

        with tf.variable_scope("match") as match_scope:
          match = comparisons.RegressModel(source, target_model)

        init_vars = (source_scope.global_variables()
                     + match_scope.global_variables())
        self.sess.run(tf.initializers.variables(init_vars))

        stats = match.fit(dataset)

    loss = pd.DataFrame(stats["loss"])["singleton"]
    logging.info(f"Loss: {loss.iloc[::10]}")
    initial_loss = loss.iloc[:5].mean()
    logging.info(f"Initial loss: {initial_loss}")
    final_loss = loss.iloc[-5:].mean()
    logging.info(f"Final loss: {final_loss}")

    assert initial_loss / final_loss > 10
    assert final_loss < loss_ub


if __name__ == "__main__":
  absltest.main()
