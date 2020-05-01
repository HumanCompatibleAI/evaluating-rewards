# Copyright 2020 Adam Gleave
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

"""Unit tests for evaluating_rewards.canonical_sample."""

import itertools
from typing import Mapping

import gym
from imitation.util import data
import numpy as np
import pytest
from stable_baselines.common import vec_env
import tensorflow as tf

from evaluating_rewards import canonical_sample, datasets, rewards, serialize, tabular
from evaluating_rewards import envs  # noqa: F401  # pylint:disable=unused-import


def mesh_evaluate_models_slow(
    models: Mapping[canonical_sample.K, rewards.RewardModel],
    obs: np.ndarray,
    actions: np.ndarray,
    next_obs: np.ndarray,
) -> Mapping[canonical_sample.K, np.ndarray]:
    """
    Evaluate models on the Cartesian product of `obs`, `actions`, `next_obs`.

    Same interface as `canonical_sample.mesh_evaluate_models`. However, this is much simpler, but
    also much slower (around 20x). We use it for testing to verify they produce the same results.
    It might also be useful in the future for other optimisations (e.g. a JIT like Numba).
    """
    transitions = list(itertools.product(obs, actions, next_obs))
    tiled_obs, tiled_acts, tiled_next_obs = (
        np.array([m[i] for m in transitions]) for i in range(3)  # pylint:disable=not-an-iterable
    )
    dones = np.zeros(len(tiled_obs), dtype=np.bool)
    transitions = data.Transitions(
        obs=tiled_obs, acts=tiled_acts, next_obs=tiled_next_obs, dones=dones
    )
    rews = rewards.evaluate_models(models, transitions)
    rews = {k: v.reshape(len(obs), len(actions), len(next_obs)) for k, v in rews.items()}
    return rews


MESH_SPACES = [
    gym.spaces.Box(low=-1, high=1, shape=(2,)),
    gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
]


@pytest.mark.parametrize("space", MESH_SPACES)
@pytest.mark.parametrize("n_models", range(3))
def test_mesh_evaluate_models(
    graph: tf.Graph, session: tf.Session, space: gym.Space, n_models: int, n_mesh: int = 64,
):
    """Checks `canonical_sample.mesh_evaluate_models` agrees with `mesh_evaluate_models_slow`."""
    with datasets.space_to_sample(space) as dist:
        obs = dist(n_mesh)
        actions = dist(n_mesh)
        next_obs = dist(n_mesh)

    with graph.as_default():
        models = {}
        for i in range(n_models):
            with tf.variable_scope(str(i)):
                models[i] = rewards.MLPRewardModel(space, space)

        session.run(tf.global_variables_initializer())
        with session.as_default():
            expected = mesh_evaluate_models_slow(models, obs, actions, next_obs)
            actual = canonical_sample.mesh_evaluate_models(models, obs, actions, next_obs)

    assert expected.keys() == actual.keys()
    for k in expected:
        assert np.allclose(expected[k], actual[k]), f"difference in model {k}"


@pytest.mark.parametrize("discount", [0.9, 0.99, 1.0])
def test_sample_canon_shaping(
    graph: tf.Graph, session: tf.Session, discount: float, eps: float = 1e-4,
):
    """Tests canonical_sample.sample_canon_shaping.

    Specifically, verifies that sparse, sparse affine-transformed and dense rewards in PointMass
    compare equal (distance < eps); and than sparse and the ground-truth (norm) reward are unequal
    (distance > 0.1).
    """
    venv = vec_env.DummyVecEnv([lambda: gym.make("evaluating_rewards/PointMassLine-v0")])
    reward_types = [
        "evaluating_rewards/PointMassSparseWithCtrl-v0",
        "evaluating_rewards/PointMassDenseWithCtrl-v0",
        "evaluating_rewards/PointMassGroundTruth-v0",
    ]
    with graph.as_default():
        with session.as_default():
            models = {k: serialize.load_reward(k, "dummy", venv, discount) for k in reward_types}
            constant = rewards.ConstantReward(venv.observation_space, venv.action_space)
            constant.constant.set_constant(42.0)
            models["big_sparse"] = rewards.LinearCombinationModelWrapper(
                {
                    "model": (
                        models["evaluating_rewards/PointMassSparseWithCtrl-v0"],
                        tf.constant(10.0),
                    ),
                    "shift": (constant, tf.constant(1.0)),
                }
            )

    with datasets.space_to_sample(venv.observation_space) as obs_dist:
        with datasets.space_to_sample(venv.action_space) as act_dist:
            with datasets.iid_transition_generator(obs_dist, act_dist) as iid_generator:
                batch = iid_generator(256)
    canon_rew = canonical_sample.sample_canon_shaping(
        models, batch, act_dist, obs_dist, n_mean_samples=256, discount=discount,
    )

    sparse_vs_affine = tabular.direct_distance(
        canon_rew["evaluating_rewards/PointMassSparseWithCtrl-v0"], canon_rew["big_sparse"], p=1,
    )
    assert sparse_vs_affine < eps
    sparse_vs_dense = tabular.direct_distance(
        canon_rew["evaluating_rewards/PointMassSparseWithCtrl-v0"],
        canon_rew["evaluating_rewards/PointMassDenseWithCtrl-v0"],
        p=1,
    )
    assert sparse_vs_dense < eps
    sparse_vs_gt = tabular.direct_distance(
        canon_rew["evaluating_rewards/PointMassSparseWithCtrl-v0"],
        canon_rew["evaluating_rewards/PointMassGroundTruth-v0"],
        p=1,
    )
    assert sparse_vs_gt > 0.1


CROSS_DISTANCE_TEST_CASES = [
    {"rewxs": {}, "rewys": {"bar": np.zeros(4)}, "expected": {}},
    {
        "rewxs": {"foo": np.zeros(4), 42: np.ones(4)},
        "rewys": {"bar": np.zeros(4), None: np.ones(4)},
        "expected": {("foo", "bar"): 0, ("foo", None): 1, (42, "bar"): 1, (42, None): 0},
    },
]


@pytest.mark.parametrize("test_case", CROSS_DISTANCE_TEST_CASES)
@pytest.mark.parametrize("threading", [False, True])
@pytest.mark.parametrize("parallelism", [None, 1, 2])
def test_cross_distance(test_case, parallelism: int, threading: bool) -> None:
    """Tests canonical_sample.cross_distance on CROSS_DISTANCE_TEST_CASES."""
    actual = canonical_sample.cross_distance(
        test_case["rewxs"],
        test_case["rewys"],
        distance_fn=tabular.direct_distance,
        parallelism=parallelism,
        threading=threading,
    )
    assert test_case["expected"] == actual
