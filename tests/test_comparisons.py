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
# Environments registered as a side-effect of importing
from evaluating_rewards import envs  # pylint:disable=unused-import
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
        "rel_loss_lb": 10,
    },
    "dense": {
        "target": "evaluating_rewards/PointMassDense-v0",
        "loss_ub": 4e-2,
        "rel_loss_lb": 10,
    },
    # For sparse and zero, set a low relative error bound, since some
    # random seeds have small scale and so get a low initial loss.
    "sparse": {
        "target": "evaluating_rewards/PointMassSparse-v0",
        "loss_ub": 4e-2,
        "rel_loss_lb": 2,
    },
    "zero": {
        "target": "evaluating_rewards/Zero-v0",
        "loss_ub": 2e-4,
        "rel_loss_lb": 2,
    }
}


class RewardTest(common.TensorFlowTestCase):
  """Unit tests for evaluating_rewards.rewards."""

  @parameterized.named_parameters(common.combine_dicts_as_kwargs(
      PM_REWARD_TYPES
  ))
  def test_regress(self, target: str, loss_ub: float, rel_loss_lb: float):
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
      dataset = dataset_generator(1e5, 512)

      with self.graph.as_default():
        with self.sess.as_default():
          with tf.variable_scope("source") as source_scope:
            source = rewards.MLPRewardModel(venv.observation_space,
                                            venv.action_space)

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
      initial_loss = loss.iloc[0]
      logging.info(f"Initial loss: {initial_loss}")
      final_loss = loss.iloc[-10:].mean()
      logging.info(f"Final loss: {final_loss}")

      assert initial_loss / final_loss > rel_loss_lb
      assert final_loss < loss_ub


if __name__ == "__main__":
  absltest.main()
