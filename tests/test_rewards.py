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
from evaluating_rewards.envs import mujoco
from evaluating_rewards.envs import point_mass
from evaluating_rewards.experiments import datasets
from tests import common
from imitation.policies import base
from imitation.rewards import reward_net
from imitation.util import rollout
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
        "hid_sizes": [64, 64],
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


STANDALONE_REWARD_MODELS = {
    "halfcheetah_ground_truth": {
        "env_id": "evaluating_rewards/HalfCheetah-v3",
        "model_class": mujoco.HalfCheetahGroundTruthReward,
    },
    "hopper_ground_truth": {
        "env_id": "evaluating_rewards/Hopper-v3",
        "model_class": mujoco.HopperGroundTruthReward,
    },
    "hopper_backflip": {
        "env_id": "evaluating_rewards/Hopper-v3",
        "model_class": mujoco.HopperBackflipReward,
    },
    "point_maze_ground_truth": {
        "env_id": "imitation/PointMazeLeft-v0",
        "model_class": mujoco.PointMazeReward,
        "target": np.array([0.3, 0.3, 0]),
    }
}
STANDALONE_REWARD_MODELS.update(common.combine_dicts(
    ENVS, GENERAL_REWARD_MODELS
))
STANDALONE_REWARD_MODELS.update(common.combine_dicts(
    {"pm": {"env_id": "evaluating_rewards/PointMassLine-v0"}},
    POINT_MASS_MODELS,
))

REWARD_WRAPPERS = {
    "base_wrapper": {"wrapper_cls": rewards.RewardModelWrapper},
    "stop_gradients": {"wrapper_cls": rewards.StopGradientsModelWrapper},
    "affine": {"wrapper_cls": rewards.AffineTransform},
    "potential": {"wrapper_cls": rewards.PotentialShapingWrapper},
}


GROUND_TRUTH = {
    "half_cheetah": {
        "env_id": "evaluating_rewards/HalfCheetah-v3",
        "reward_id": ("evaluating_rewards/HalfCheetahGroundTruth"
                      "ForwardWithCtrl-v0"),
    },
    "hopper": {
        "env_id": "evaluating_rewards/Hopper-v3",
        "reward_id": "evaluating_rewards/HopperGroundTruthForwardWithCtrl-v0",
    },
    "point_maze": {
        "env_id": "imitation/PointMazeLeft-v0",
        "reward_id": "evaluating_rewards/PointMazeGroundTruthWithCtrl-v0",
    }
}


USE_TEST = {
    "test": {"use_test": True},
    "train": {"use_test": False},
}


class RewardTest(common.TensorFlowTestCase):
  """Unit tests for evaluating_rewards.rewards."""

  def _test_serialize_identity(self, env_id, make_model):
    """Creates reward model, saves it, reloads it, and checks for equality."""
    venv = vec_env.DummyVecEnv([lambda: common.make_env(env_id)])
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

      models = {"o": original, "ld": loaded_direct, "li": loaded_indirect}
      preds = rewards.evaluate_models(models, batch)

    for model in models.values():
      assert original.observation_space == model.observation_space
      assert original.action_space == model.action_space

    assert len(preds) == len(models)
    for pred in preds.values():
      assert np.allclose(preds["o"], pred)

  @parameterized.named_parameters(common.combine_dicts_as_kwargs(
      STANDALONE_REWARD_MODELS
  ))
  def test_serialize_identity_standalone(self, env_id, model_class, **kwargs):
    """Creates reward model, saves it, reloads it, and checks for equality."""
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

  @parameterized.named_parameters(common.combine_dicts_as_kwargs(
      REWARD_WRAPPERS, ENVS
  ))
  def test_serialize_identity_wrapper(self, wrapper_cls, env_id):
    """Checks for equality between original and loaded wrapped reward."""
    def make_model(env):
      mlp = rewards.MLPRewardModel(env.observation_space, env.action_space)
      return wrapper_cls(mlp)

    return self._test_serialize_identity(env_id, make_model)

  @parameterized.named_parameters(common.combine_dicts_as_kwargs(
      ENVS, USE_TEST
  ))
  def test_serialize_identity_reward_net(self, env_id, use_test):
    def make_model(env):
      net = reward_net.BasicRewardNet(env.observation_space, env.action_space)
      return rewards.RewardNetToRewardModel(net, use_test=use_test)

    return self._test_serialize_identity(env_id, make_model)

  @parameterized.named_parameters(common.combine_dicts_as_kwargs(GROUND_TRUTH))
  def test_ground_truth_similar_to_gym(self, env_id, reward_id):
    """Checks that reward models predictions match those of Gym reward."""
    # Generate rollouts, recording Gym reward
    venv = vec_env.DummyVecEnv([lambda: common.make_env(env_id)])
    policy = base.RandomPolicy(venv.observation_space, venv.action_space)
    transitions = rollout.generate_transitions(policy, venv, n_timesteps=1024)
    batch = rewards.Batch(obs=transitions.obs,
                          actions=transitions.acts,
                          next_obs=transitions.next_obs)
    gym_reward = transitions.rews

    # Make predictions using reward model
    with self.graph.as_default(), self.sess.as_default():
      reward_model = serialize.load_reward(reward_id, "dummy", venv)
      pred_reward = rewards.evaluate_models({"m": reward_model}, batch)["m"]

    # Are the predictions close to true Gym reward?
    np.testing.assert_allclose(gym_reward, pred_reward, rtol=0, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
