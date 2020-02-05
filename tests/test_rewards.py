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

import tempfile

from imitation.policies import base
from imitation.rewards import reward_net
from imitation.util import rollout
from imitation.util import serialize as util_serialize
import numpy as np
import pytest
import tensorflow as tf

from evaluating_rewards import datasets, rewards, serialize
from evaluating_rewards.envs import mujoco, point_mass
from tests import common

ENVS = ["FrozenLake-v0", "CartPole-v1", "Pendulum-v0"]

STANDALONE_REWARD_MODELS = {
    "halfcheetah_ground_truth": {
        "env_name": "evaluating_rewards/HalfCheetah-v3",
        "model_class": mujoco.HalfCheetahGroundTruthReward,
        "kwargs": {},
    },
    "hopper_ground_truth": {
        "env_name": "evaluating_rewards/Hopper-v3",
        "model_class": mujoco.HopperGroundTruthReward,
        "kwargs": {},
    },
    "hopper_backflip": {
        "env_name": "evaluating_rewards/Hopper-v3",
        "model_class": mujoco.HopperBackflipReward,
        "kwargs": {},
    },
    "point_maze_ground_truth": {
        "env_name": "imitation/PointMazeLeftVel-v0",
        "model_class": mujoco.PointMazeReward,
        "kwargs": {"target": np.array([0.3, 0.3, 0])},
    },
}

GENERAL_REWARD_MODELS = {
    "mlp": {"model_class": rewards.MLPRewardModel, "kwargs": {}},
    "mlp_wide": {"model_class": rewards.MLPRewardModel, "kwargs": {"hid_sizes": [64, 64]}},
    "potential": {"model_class": rewards.PotentialShaping, "kwargs": {}},
    "constant": {"model_class": rewards.ConstantReward, "kwargs": {}},
}
ENVS_KWARGS = {env: {"env_name": env} for env in ENVS}
STANDALONE_REWARD_MODELS.update(common.combine_dicts(ENVS_KWARGS, GENERAL_REWARD_MODELS))

POINT_MASS_MODELS = {
    "ground_truth": {"model_class": point_mass.PointMassGroundTruth},
    "sparse": {"model_class": point_mass.PointMassSparseReward},
    "shaping": {"model_class": point_mass.PointMassShaping},
    "dense": {"model_class": point_mass.PointMassDenseReward},
}
STANDALONE_REWARD_MODELS.update(
    common.combine_dicts(
        {"pm": {"env_name": "evaluating_rewards/PointMassLine-v0", "kwargs": {}}}, POINT_MASS_MODELS
    )
)

REWARD_WRAPPERS = [
    rewards.RewardModelWrapper,
    rewards.StopGradientsModelWrapper,
    rewards.AffineTransform,
    rewards.PotentialShapingWrapper,
]

GROUND_TRUTH = {
    "half_cheetah": (
        "evaluating_rewards/HalfCheetah-v3",
        "evaluating_rewards/HalfCheetahGroundTruthForwardWithCtrl-v0",
    ),
    "hopper": (
        "evaluating_rewards/Hopper-v3",
        "evaluating_rewards/HopperGroundTruthForwardWithCtrl-v0",
    ),
    "point_mass": (
        "evaluating_rewards/PointMassLine-v0",
        "evaluating_rewards/PointMassGroundTruth-v0",
    ),
    "point_maze": (
        "imitation/PointMazeLeftVel-v0",
        "evaluating_rewards/PointMazeGroundTruthWithCtrl-v0",
    ),
}


@pytest.fixture(name="helper_serialize_identity")
def fixture_serialize_identity(graph, session, venv):
    """Creates reward model, saves it, reloads it, and checks for equality."""

    def f(make_model):
        policy = base.RandomPolicy(venv.observation_space, venv.action_space)
        with datasets.rollout_policy_generator(venv, policy) as dataset_callable:
            batch = dataset_callable(1024)

            with graph.as_default(), session.as_default():
                original = make_model(venv)
                session.run(tf.global_variables_initializer())

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

    return f


@common.mark_parametrize_kwargs(STANDALONE_REWARD_MODELS)
def test_serialize_identity_standalone(helper_serialize_identity, model_class, kwargs):
    """Creates reward model, saves it, reloads it, and checks for equality."""

    def make_model(venv):
        return model_class(venv.observation_space, venv.action_space, **kwargs)

    return helper_serialize_identity(make_model)


@pytest.mark.parametrize("env_name", ENVS)
def test_serialize_identity_linear_combination(helper_serialize_identity):
    """Checks for equality between original and reloaded LC of reward models."""

    def make_model(env):
        constant_a = rewards.ConstantReward(env.observation_space, env.action_space)
        weight_a = tf.constant(42.0)
        constant_b = rewards.ConstantReward(env.observation_space, env.action_space)
        weight_b = tf.get_variable("weight_b", initializer=tf.constant(13.37))
        return rewards.LinearCombinationModelWrapper(
            {"constant": (constant_a, weight_a), "zero": (constant_b, weight_b)}
        )

    return helper_serialize_identity(make_model)


@pytest.mark.parametrize("wrapper_cls", REWARD_WRAPPERS)
@pytest.mark.parametrize("env_name", ENVS)
def test_serialize_identity_wrapper(helper_serialize_identity, wrapper_cls):
    """Checks for equality between original and loaded wrapped reward."""

    def make_model(env):
        mlp = rewards.MLPRewardModel(env.observation_space, env.action_space)
        return wrapper_cls(mlp)

    return helper_serialize_identity(make_model)


@pytest.mark.parametrize("env_name", ENVS)
@pytest.mark.parametrize("use_test", [True, False])
def test_serialize_identity_reward_net(helper_serialize_identity, use_test):
    def make_model(env):
        net = reward_net.BasicRewardNet(env.observation_space, env.action_space)
        return rewards.RewardNetToRewardModel(net, use_test=use_test)

    return helper_serialize_identity(make_model)


@pytest.mark.parametrize("env_name,reward_id", GROUND_TRUTH.values(), ids=list(GROUND_TRUTH.keys()))
def test_ground_truth_similar_to_gym(graph, session, venv, reward_id):
    """Checks that reward models predictions match those of Gym reward."""
    # Generate rollouts, recording Gym reward
    policy = base.RandomPolicy(venv.observation_space, venv.action_space)
    transitions = rollout.generate_transitions(policy, venv, n_timesteps=1024)
    batch = rewards.Batch(
        obs=transitions.obs, actions=transitions.acts, next_obs=transitions.next_obs
    )
    gym_reward = transitions.rews

    # Make predictions using reward model
    with graph.as_default(), session.as_default():
        reward_model = serialize.load_reward(reward_id, "dummy", venv)
        pred_reward = rewards.evaluate_models({"m": reward_model}, batch)["m"]

    # Are the predictions close to true Gym reward?
    np.testing.assert_allclose(gym_reward, pred_reward, rtol=0, atol=5e-5)


REWARD_LEN = 10000
NUM_SAMPLES = 10


def test_least_l2_affine_random():
    """Check least_l2_affine recovers random affine transformations."""
    source = np.random.randn(REWARD_LEN)

    shifts = np.random.randn(NUM_SAMPLES)
    scales = np.exp(np.random.randn(NUM_SAMPLES))

    for shift, scale in zip(shifts, scales):
        target = source * scale + shift
        params = rewards.least_l2_affine(source, target)
        assert np.allclose([shift, scale], [params.shift, params.scale])
        assert params.scale >= 0

        for has_shift in [False, True]:
            target = source * scale
            params = rewards.least_l2_affine(source, target, shift=has_shift)
            assert np.allclose([0.0, scale], [params.shift, params.scale])

        for has_scale in [False, True]:
            target = source + shift
            params = rewards.least_l2_affine(source, target, scale=has_scale)
            assert np.allclose([shift, 1.0], [params.shift, params.scale], atol=0.1)


def test_least_l2_affine_zero():
    """Check least_l2_affine finds zero scale and shift for negative and zero target."""
    for _ in range(NUM_SAMPLES):
        source = np.random.randn(REWARD_LEN)

        params = rewards.least_l2_affine(source, -source)
        assert np.allclose([0.0], [params.scale])
        assert params.scale >= 0
        assert np.allclose([0.0], [params.shift], atol=0.1)

        params = rewards.least_l2_affine(source, np.zeros_like(source))
        assert np.allclose([0.0, 0.0], [params.shift, params.scale])
        assert params.scale >= 0
