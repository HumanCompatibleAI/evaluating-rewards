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

import dataclasses
import tempfile

import hypothesis
from hypothesis import strategies as st
from hypothesis.extra import numpy as hp_numpy
from imitation.data import rollout
from imitation.policies import base as base_policies
from imitation.rewards import reward_net
from imitation.util import serialize as util_serialize
import numpy as np
import pytest
from stable_baselines.common import vec_env
import tensorflow as tf

from evaluating_rewards import datasets, serialize
from evaluating_rewards.envs import mujoco, point_mass
from evaluating_rewards.rewards import base
from tests import common

ENVS = ["FrozenLake-v1", "CartPole-v1", "Pendulum-v1"]

STANDALONE_REWARD_MODELS = {
    "halfcheetah_ground_truth": {
        "env_name": "seals/HalfCheetah-v0",
        "model_class": mujoco.HalfCheetahGroundTruthReward,
        "kwargs": {},
    },
    "hopper_ground_truth": {
        "env_name": "seals/Hopper-v0",
        "model_class": mujoco.HopperGroundTruthReward,
        "kwargs": {},
    },
    "hopper_backflip": {
        "env_name": "seals/Hopper-v0",
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
    "mlp": {"model_class": base.MLPRewardModel, "kwargs": {}},
    "mlp_wide": {"model_class": base.MLPRewardModel, "kwargs": {"hid_sizes": [64, 64]}},
    "mlp_potential": {"model_class": base.MLPPotentialShaping, "kwargs": {}},
    "constant": {"model_class": base.ConstantReward, "kwargs": {}},
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
    base.RewardModelWrapper,
    base.StopGradientsModelWrapper,
    base.AffineTransform,
    base.MLPPotentialShapingWrapper,
]

GROUND_TRUTH = {
    "half_cheetah": (
        "seals/HalfCheetah-v0",
        "evaluating_rewards/HalfCheetahGroundTruthForwardWithCtrl-v0",
    ),
    "hopper": ("seals/Hopper-v0", "evaluating_rewards/HopperGroundTruthForwardWithCtrl-v0"),
    "point_mass": (
        "evaluating_rewards/PointMassLine-v0",
        "evaluating_rewards/PointMassGroundTruth-v0",
    ),
    "point_maze": (
        "imitation/PointMazeLeftVel-v0",
        "evaluating_rewards/PointMazeGroundTruthWithCtrl-v0",
    ),
}

ENV_POTENTIALS = [
    # (env_name, potential_class)
    # Tests require the environments are fixed length.
    ("seals/CartPole-v0", base.MLPPotentialShaping),
    ("evaluating_rewards/PointMassLine-v0", point_mass.PointMassShaping),
]
DISCOUNTS = [0.9, 0.99, 1.0]


@pytest.fixture(name="helper_serialize_identity")
def fixture_serialize_identity(
    graph: tf.Graph,
    session: tf.Session,
    venv: vec_env.VecEnv,
):
    """Creates reward model, saves it, reloads it, and checks for equality."""

    def f(make_model):
        policy = base_policies.RandomPolicy(venv.observation_space, venv.action_space)
        with datasets.transitions_factory_from_policy(venv, policy) as dataset_callable:
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
                preds = base.evaluate_models(models, batch)

            for model in models.values():
                assert original.observation_space == model.observation_space
                assert original.action_space == model.action_space

            assert len(preds) == len(models)
            for pred in preds.values():
                assert np.allclose(preds["o"], pred)

    return f


@common.mark_parametrize_kwargs(STANDALONE_REWARD_MODELS)
@pytest.mark.parametrize("discount", [0.9, 0.99, 1.0])
def test_serialize_identity_standalone(
    helper_serialize_identity, model_class, discount: float, kwargs
):
    """Creates reward model, saves it, reloads it, and checks for equality."""

    def make_model(venv):
        model = model_class(venv.observation_space, venv.action_space, **kwargs)
        model.set_discount(discount)
        return model

    return helper_serialize_identity(make_model)


@pytest.mark.parametrize("env_name", ENVS)
def test_serialize_identity_linear_combination(helper_serialize_identity):
    """Checks for equality between original and reloaded LC of reward models."""

    def make_model(env):
        constant_a = base.ConstantReward(env.observation_space, env.action_space)
        weight_a = tf.constant(42.0)
        constant_b = base.ConstantReward(env.observation_space, env.action_space)
        weight_b = tf.get_variable("weight_b", initializer=tf.constant(13.37))
        return base.LinearCombinationModelWrapper(
            {"constant": (constant_a, weight_a), "zero": (constant_b, weight_b)}
        )

    return helper_serialize_identity(make_model)


@pytest.mark.parametrize("wrapper_cls", REWARD_WRAPPERS)
@pytest.mark.parametrize("env_name", ENVS)
@pytest.mark.parametrize("discount", [0.9, 0.99, 1.0])
def test_serialize_identity_wrapper(helper_serialize_identity, wrapper_cls, discount: float):
    """Checks for equality between original and loaded wrapped reward."""

    def make_model(env):
        mlp = base.MLPRewardModel(env.observation_space, env.action_space)
        model = wrapper_cls(mlp)
        model.set_discount(discount)
        return model

    return helper_serialize_identity(make_model)


@pytest.mark.parametrize("cls", [reward_net.BasicRewardNet, reward_net.BasicShapedRewardNet])
@pytest.mark.parametrize("env_name", ENVS)
@pytest.mark.parametrize("use_test", [True, False])
def test_serialize_identity_reward_net(helper_serialize_identity, cls, use_test):
    def make_model(env):
        net = cls(env.observation_space, env.action_space)
        return base.RewardNetToRewardModel(net, use_test=use_test)

    return helper_serialize_identity(make_model)


@pytest.mark.parametrize("env_name,reward_id", GROUND_TRUTH.values(), ids=list(GROUND_TRUTH.keys()))
def test_ground_truth_similar_to_gym(graph, session, venv, reward_id):
    """Checks that reward models predictions match those of Gym reward."""
    # Generate rollouts, recording Gym reward
    policy = base_policies.RandomPolicy(venv.observation_space, venv.action_space)
    transitions = rollout.generate_transitions(policy, venv, n_timesteps=1024)
    gym_reward = transitions.rews

    # Make predictions using reward model
    with graph.as_default(), session.as_default():
        reward_model = serialize.load_reward(reward_id, "dummy", venv, 1.0)
        pred_reward = base.evaluate_models({"m": reward_model}, transitions)["m"]

    # Are the predictions close to true Gym reward?
    np.testing.assert_allclose(gym_reward, pred_reward, rtol=0, atol=5e-5)


@pytest.mark.parametrize("env_name,potential_cls", ENV_POTENTIALS)
@pytest.mark.parametrize("discount", DISCOUNTS)
def test_potential_shaping_cycle(
    graph, session, venv, potential_cls, discount: float, num_episodes: int = 10
) -> None:
    """Test that potential shaping is constant on any fixed-length cycle.

    Specifically, performs rollouts of a random policy in the environment.
    Fixes the starting state for each trajectory at the all-zero state.
    Then computes episode return, and checks they're all equal.

    Requires environment be fixed length, otherwise the episode return will vary
    (except in the undiscounted case).
    """
    policy = base_policies.RandomPolicy(venv.observation_space, venv.action_space)
    trajectories = rollout.generate_trajectories(
        policy, venv, sample_until=rollout.min_episodes(num_episodes)
    )
    transitions = rollout.flatten_trajectories(trajectories)

    # Make initial state fixed as all-zero.
    # Note don't need to change final state, since `dones` being `True` should
    # force potential to be zero at those states.
    obs = np.array(transitions.obs)
    idxs = np.where(transitions.dones)[0] + 1
    idxs = np.pad(idxs[:-1], (1, 0), "constant")
    obs[idxs, :] = 0
    transitions = dataclasses.replace(transitions, obs=obs)

    with graph.as_default(), session.as_default():
        reward_model = potential_cls(venv.observation_space, venv.action_space, discount=discount)
        session.run(tf.global_variables_initializer())
        rews = base.evaluate_models({"m": reward_model}, transitions)

    rets = base.compute_return_from_rews(rews, transitions.dones, discount=discount)["m"]
    if discount == 1.0:
        assert np.allclose(rets, 0.0, atol=1e-5)
    assert np.allclose(rets, np.mean(rets), atol=1e-5)


@pytest.mark.parametrize("env_name,potential_cls", ENV_POTENTIALS)
@pytest.mark.parametrize("discount", DISCOUNTS)
def test_potential_shaping_invariants(
    graph, session, venv, potential_cls, discount: float, num_timesteps: int = 100
):
    """Test that potential shaping obeys several invariants.

    Specifically:
        1. new_potential must be constant when dones is true, and zero when `discount == 1.0`.
        2. new_potential depends only on next observation.
        3. old_potential depends only on current observation.
        4. Shaping is discount * new_potential - old_potential.
    """
    # Invariants:
    # When done, new_potential should always be zero.
    # self.discount * new_potential - old_potential should equal the output
    # Same old_obs should have same old_potential; same new_obs should have same new_potential.
    policy = base_policies.RandomPolicy(venv.observation_space, venv.action_space)
    transitions = rollout.generate_transitions(policy, venv, n_timesteps=num_timesteps)

    with graph.as_default(), session.as_default():
        potential = potential_cls(venv.observation_space, venv.action_space, discount=discount)
        session.run(tf.global_variables_initializer())
        (old_pot,), (new_pot,) = base.evaluate_potentials([potential], transitions)

    # Check invariant 1: new_potential must be zero when dones is true
    transitions_all_done = dataclasses.replace(
        transitions, dones=np.ones_like(transitions.dones, dtype=np.bool)
    )
    with session.as_default():
        _, new_pot_done = base.evaluate_potentials([potential], transitions_all_done)
    expected_new_pot_done = 0.0 if discount == 1.0 else np.mean(new_pot_done)
    assert np.allclose(new_pot_done, expected_new_pot_done)

    # Check invariants 2 and 3: {new,old}_potential depend only on {next,current} observation
    def _shuffle(fld: str):
        arr = np.array(getattr(transitions, fld))
        np.random.shuffle(arr)
        trans = dataclasses.replace(transitions, **{fld: arr})
        with session.as_default():
            return base.evaluate_potentials([potential], trans)

    (old_pot_shuffled,), _ = _shuffle("next_obs")
    _, (new_pot_shuffled,) = _shuffle("obs")
    assert np.all(old_pot == old_pot_shuffled)
    assert np.all(new_pot == new_pot_shuffled)

    # Check invariant 4: that reward output is as expected given potentials
    with session.as_default():
        rew = base.evaluate_models({"m": potential}, transitions)["m"]
    assert np.allclose(rew, discount * new_pot - old_pot)


REWARD_LEN = 10000
NUM_SAMPLES = 10


def test_least_l2_affine_random():
    """Check least_l2_affine recovers random affine transformations."""
    source = np.random.randn(REWARD_LEN)

    shifts = np.random.randn(NUM_SAMPLES)
    scales = np.exp(np.random.randn(NUM_SAMPLES))

    for shift, scale in zip(shifts, scales):
        target = source * scale + shift
        params = base.least_l2_affine(source, target)
        assert np.allclose([shift, scale], [params.shift, params.scale])
        assert params.scale >= 0

        for has_shift in [False, True]:
            target = source * scale
            params = base.least_l2_affine(source, target, shift=has_shift)
            assert np.allclose([0.0, scale], [params.shift, params.scale])

        for has_scale in [False, True]:
            target = source + shift
            params = base.least_l2_affine(source, target, scale=has_scale)
            assert np.allclose([shift, 1.0], [params.shift, params.scale], atol=0.1)


def test_least_l2_affine_zero():
    """Check least_l2_affine finds zero scale and shift for negative and zero target."""
    for _ in range(NUM_SAMPLES):
        source = np.random.randn(REWARD_LEN)

        params = base.least_l2_affine(source, -source)
        assert np.allclose([0.0], [params.scale])
        assert params.scale >= 0
        assert np.allclose([0.0], [params.shift], atol=0.1)

        params = base.least_l2_affine(source, np.zeros_like(source))
        assert np.allclose([0.0, 0.0], [params.shift, params.scale])
        assert params.scale >= 0


def _test_compute_return_from_rews(dones: np.ndarray, discount: float) -> None:
    """Test logic to compute return."""
    increasing = np.array([]) if len(dones) == 0 else np.concatenate(([0], np.cumsum(dones)[:-1]))
    rews = {
        "zero": np.zeros(len(dones)),
        "ones": np.ones(len(dones)),
        "increasing": increasing,
    }
    ep_returns = base.compute_return_from_rews(rews, dones, discount)
    assert ep_returns.keys() == rews.keys()

    num_eps = np.sum(dones)
    for k, v in ep_returns.items():
        assert v.shape == (num_eps,), f"violation at {k}"

    assert np.all(ep_returns["zero"] == 0.0)

    boundaries = np.where(dones)[0]
    idxs = np.array([0] + list(boundaries + 1))
    lengths = idxs[1:] - idxs[:-1]
    if discount == 1.0:
        one_expected_return = lengths
    else:
        one_expected_return = (1 - np.power(discount, lengths)) / (1 - discount)
    assert np.allclose(ep_returns["ones"], one_expected_return)

    ep_idx = rews["increasing"][idxs[:-1]]
    assert np.allclose(ep_returns["increasing"], one_expected_return * ep_idx)


_dones_strategy = hp_numpy.arrays(
    dtype=np.bool, shape=st.integers(min_value=0, max_value=1000), fill=st.booleans()
)


@hypothesis.given(
    dones=_dones_strategy,
    discount=st.floats(
        min_value=0, max_value=1, exclude_max=True, allow_infinity=False, allow_nan=False
    ),
)
def test_compute_return_from_rews_discounted(dones: np.ndarray, discount: float) -> None:
    """Test logic to compute return, in discounted case."""
    return _test_compute_return_from_rews(dones, discount)


@hypothesis.given(dones=_dones_strategy)
def test_compute_return_from_rews_undiscounted(dones: np.ndarray) -> None:
    """Test logic to compute return, in undiscounted case.

    This ensures coverage of a different code path.
    """
    return _test_compute_return_from_rews(dones, discount=1.0)
