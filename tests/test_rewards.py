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

import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from evaluating_rewards import rewards
from evaluating_rewards import serialize
from evaluating_rewards.envs import point_mass
from evaluating_rewards.experiments import datasets
from tests import common
import gym
from imitation.policies import base
from imitation.util import serialize as util_serialize
import numpy as np
from stable_baselines.common import vec_env
import tensorflow as tf


ENVS = {env: {"env_id": env}
        for env in ["FrozenLake-v0", "CartPole-v1", "Pendulum-v0"]}


GENERAL_REWARD_MODELS = {
    "mlp": {
        "model_class": rewards.MLPRewardModel,
    },
    "mlp_wide": {
        "model_class": rewards.MLPRewardModel,
        "kwargs": {"hid_sizes": [64, 64]},
    },
    "potential": {
        "model_class": rewards.PotentialShaping,
    },
    "constant": {
        "model_class": rewards.ConstantReward,
    },
}


POINT_MASS_MODELS = {
    "ground_truth": {"model_class": point_mass.PointMassGroundTruth},
    "sparse": {"model_class": point_mass.PointMassSparseReward},
    "shaping": {"model_class": point_mass.PointMassShaping},
    "dense": {"model_class": point_mass.PointMassDenseReward},
}


STANDALONE_REWARD_MODELS = {}
STANDALONE_REWARD_MODELS.update(common.combine_dicts(
    ENVS, GENERAL_REWARD_MODELS
))
STANDALONE_REWARD_MODELS.update(common.combine_dicts(
    {"pm": {"env_id": "evaluating_rewards/PointMassLineFixedHorizon-v0"}},
    POINT_MASS_MODELS,
))


class RewardTest(common.TensorFlowTestCase):
  """Unit tests for evaluating_rewards.rewards."""

  def _test_serialize_identity(self, env_id, make_model):
    """Creates reward model, saves it, reloads it, and checks for equality."""
    venv = vec_env.DummyVecEnv([lambda: gym.make(env_id)])
    policy = base.RandomPolicy(venv.observation_space, venv.action_space)
    dataset_callable = datasets.rollout_generator(venv, policy)
    batch = next(dataset_callable(1024, 1024))

    with self.graph.as_default(), self.sess.as_default():
      original = make_model(venv)
      self.sess.run(tf.global_variables_initializer())

      with tempfile.TemporaryDirectory(prefix="eval-rew-serialize") as tmpdir:
        original.save(tmpdir)

        with tf.variable_scope("loaded_direct"):
          loaded_direct = util_serialize.Serializable.load(tmpdir)

        model_name = "evaluating_rewards/RewardModel-v0"
        loaded_indirect = serialize.load_reward(model_name, tmpdir, venv)

      models = [original, loaded_direct, loaded_indirect]
      preds = rewards.evaluate_models(models, batch)

    for model in models[1:]:
      assert original.observation_space == model.observation_space
      assert original.action_space == model.action_space

    assert len(preds) == len(models)
    for pred in preds[1:]:
      assert np.allclose(preds[0], pred)

  @parameterized.named_parameters(common.combine_dicts_as_kwargs(
      STANDALONE_REWARD_MODELS
  ))
  def test_serialize_identity_standalone(self, env_id, model_class,
                                         kwargs=None):
    """Creates reward model, saves it, reloads it, and checks for equality."""
    kwargs = kwargs or {}

    def make_model(venv):
      return model_class(venv.observation_space, venv.action_space, **kwargs)

    return self._test_serialize_identity(env_id, make_model)

  @parameterized.named_parameters(common.combine_dicts_as_kwargs(ENVS))
  def test_serialize_identity_linear_combination(self, env_id):
    """Checks for equality between original and reloaded LC of reward models."""
    def make_model(env):
      constant_a = rewards.ConstantReward(env.observation_space,
                                          env.action_space)
      weight_a = tf.constant(42.0)
      constant_b = rewards.ConstantReward(env.observation_space,
                                          env.action_space)
      weight_b = tf.get_variable("weight_b",
                                 initializer=tf.constant(13.37))
      return rewards.LinearCombinationModelWrapper({
          "constant": (constant_a, weight_a),
          "zero": (constant_b, weight_b)
      })

    return self._test_serialize_identity(env_id, make_model)

  @parameterized.named_parameters(common.combine_dicts_as_kwargs(ENVS))
  def test_serialize_identity_affine_transform(self, env_id):
    """Checks for equality between original and loaded affine transformation."""
    def make_model(env):
      mlp = rewards.MLPRewardModel(env.observation_space, env.action_space)
      return rewards.AffineTransform(mlp)

    return self._test_serialize_identity(env_id, make_model)

  @parameterized.named_parameters(common.combine_dicts_as_kwargs(ENVS))
  def test_serialize_identity_potential_wrapper(self, env_id):
    """Checks for equality between original and reloaded potential shaping."""
    def make_model(env):
      mlp = rewards.MLPRewardModel(env.observation_space, env.action_space)
      return rewards.PotentialShapingWrapper(mlp)

    return self._test_serialize_identity(env_id, make_model)


if __name__ == "__main__":
  absltest.main()
