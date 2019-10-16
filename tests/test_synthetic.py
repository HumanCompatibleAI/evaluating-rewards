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

"""Unit tests for evaluating_rewards.synthetic.

Also indirectly tests evaluating_rewards.deep, evaluating_rewards.datasets and
evaluating_rewards.util.
"""

import logging
import math

import gym
from imitation import util
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from evaluating_rewards import rewards
from evaluating_rewards.envs import point_mass
from evaluating_rewards.experiments import datasets, synthetic
from tests import common


def dummy_env_and_dataset(dims: int = 5):
    """Make a simple fake environment with rollouts."""
    obs_space = gym.spaces.Box(low=np.repeat(0.0, dims), high=np.repeat(1.0, dims))
    act_space = gym.spaces.Box(low=np.repeat(0.0, dims), high=np.repeat(1.0, dims))

    def dataset_generator(total_timesteps, batch_size):
        nbatches = math.ceil(total_timesteps / batch_size)
        for _ in range(nbatches):
            obs = np.array([obs_space.sample() for _ in range(batch_size)])
            actions = np.array([act_space.sample() for _ in range(batch_size)])
            next_obs = (obs + actions).clip(0.0, 1.0)
            yield rewards.Batch(obs=obs, actions=actions, next_obs=next_obs)

    return {
        "observation_space": obs_space,
        "action_space": act_space,
        "dataset_generator": dataset_generator,
    }


def make_pm(env_name="evaluating_rewards/PointMassLine-v0"):
    """Make Point Mass environment and dataset generator."""
    venv = util.make_vec_env(env_name)
    obs_space = venv.observation_space
    act_space = venv.action_space

    pm = point_mass.PointMassPolicy(obs_space, act_space)
    with datasets.rollout_policy_generator(venv, pm) as dataset_generator:
        # It's OK to return dataset_generator outside the with context:
        # rollout_policy_generator doesn't actually have any internal resources
        # (some other datasets do).
        return {
            "observation_space": obs_space,
            "action_space": act_space,
            "dataset_generator": dataset_generator,
        }


ENVIRONMENTS = {
    "Uniform5D": dummy_env_and_dataset(dims=5),
    "PointLine": make_pm("evaluating_rewards/PointMassLine-v0"),
    "PointGrid": make_pm("evaluating_rewards/PointMassGrid-v0"),
}

ARCHITECTURES = {
    "Linear": {
        "kwargs": {
            "reward_hids": [],
            "dataset_potential_hids": [],
            "model_potential_hids": [],
            "learning_rate": 1e-2,
            "total_timesteps": 2 ** 18,
            "batch_size": 256,
        },
        "rel_upperbound": 0.1,
    },
    "OneLayer": {
        "kwargs": {
            "reward_hids": [32],
            "dataset_potential_hids": [4],
            "model_potential_hids": [32],
            "learning_rate": 1e-2,
            "total_timesteps": 2 ** 18,
            "batch_size": 512,
        },
        "rel_upperbound": 0.2,
    },
    "TwoLayer": {
        "kwargs": {
            "reward_hids": [32, 32],
            "dataset_potential_hids": [4, 4],
            "model_potential_hids": [32, 32],
            "learning_rate": 1e-2,
            "total_timesteps": 2 ** 18,
            "batch_size": 512,
        },
        "rel_upperbound": 0.2,
    },
}

EQUIV_SCALES = {
    "identity": {},  # reward functions are identical
    "random": {  # reward functions are affine transformations of each other
        "scale_fn": synthetic.log_normal,
        "constant_fn": lambda scale: scale * np.random.normal(),
    },
}

NOISY_AFFINE_ENVIRONMENTS = {
    # It generally does much better on Point{Line,Grid}, which has a much
    # smaller scale of potential noise than Uniform5D. So set higher upper bound
    # for Uniform5D than for Point*.
    "Uniform5D": dict(**ENVIRONMENTS["Uniform5D"], upperbound=2.0),
    "PointLine": dict(**ENVIRONMENTS["PointLine"], upperbound=0.025),
    "PointGrid": dict(**ENVIRONMENTS["PointGrid"], upperbound=0.025),
}


def const_functor(x):
    def f(*args):
        del args
        return x

    return f


AFFINE_TRANSFORMS = {
    "random": {
        "scale_fn": synthetic.log_normal,
        "constant_fn": lambda scale: scale * np.random.normal(),
    },
    "identity": {},
}
for scale, sign in zip([1e-3, 1e-1, 1e1, 1e3], [1, -1, 1, -1]):
    AFFINE_TRANSFORMS[f"{scale}_{sign}"] = {
        "scale_fn": const_functor(scale),
        "constant_fn": const_functor(scale / 2 * sign),
    }


SYNTHETIC_TEST = {
    "same_scale": {
        "kwargs": {},
        "rescale": False,
        "fudge_factor": 1.0,  # follow upper bound in ARCHITECTURES
    },
    "random_scale": {
        "kwargs": {
            "scale_fn": synthetic.log_normal,
            "constant_fn": lambda scale: scale * np.random.normal(),
        },
        "rescale": True,
        "fudge_factor": 2.0,  # be twice as lenient
    },
}


@common.mark_parametrize_dict("env_kwargs", ENVIRONMENTS)
@common.mark_parametrize_dict("scale_kwargs", EQUIV_SCALES)
@common.mark_parametrize_kwargs(ARCHITECTURES)
def test_identical(
    graph: tf.Graph, session: tf.Session, env_kwargs, scale_kwargs, kwargs, rel_upperbound
):
    """Try to minimize the difference between two identical reward models."""
    del rel_upperbound  # not used here, but is used in test_compare_synthetic
    with graph.as_default():
        with session.as_default():
            noise = np.array([0.0])
            _, metrics = synthetic.compare_synthetic(
                reward_noise=noise, potential_noise=noise, **env_kwargs, **scale_kwargs, **kwargs
            )
            loss = pd.DataFrame(metrics["loss"])
            loss = loss[(0.0, 0.0)]
            initial_loss = loss.iloc[0]
            final_loss = loss.iloc[-1]
            assert final_loss < 1e-4
            assert initial_loss / final_loss > 1e2


@pytest.fixture(name="helper_affine")
def fixture_affine(graph, session):
    """Do we recover affine parameters correctly?"""

    def f(upperbound, **kwargs):
        """Helper."""
        with graph.as_default():
            with session.as_default():
                df, _ = synthetic.compare_synthetic(
                    reward_noise=np.array([0.0]),
                    model_affine=True,
                    pretrain=True,
                    pretrain_size=4096,
                    **kwargs,
                )
                rel_error_scale = (df["Inferred Scale"] - df["Real Scale"]) / df["Real Scale"]
                # The constant parameter is in the same scale as the target
                # (which should be consistent across test configurations),
                # so no need to normalize.
                abs_error_constant = df["Inferred Constant"] - df["Real Constant"]

                with pd.option_context("display.max_rows", None, "display.max_columns", None):
                    logging.info("Comparison: %s", df)
                    logging.info("Relative error scale: %s", rel_error_scale)
                    logging.info("Absolute error constant: %s", abs_error_constant)

                assert rel_error_scale.abs().max() < upperbound
                assert abs_error_constant.abs().max() < upperbound

    return f


@common.mark_parametrize_dict("kwargs", dict(common.combine_dicts(ENVIRONMENTS, AFFINE_TRANSFORMS)))
def test_clean_affine(helper_affine, kwargs):
    """Can we get a good initialization when there is no noise?"""
    return helper_affine(
        total_timesteps=0,
        potential_noise=np.array([0.0]),
        model_potential=False,
        upperbound=1e-3,
        **kwargs,
    )


@common.mark_parametrize_dict(
    "kwargs", dict(common.combine_dicts(NOISY_AFFINE_ENVIRONMENTS, AFFINE_TRANSFORMS))
)
def test_pretrain_affine(helper_affine, kwargs):
    """Can we recover good affine parameters in presence of potential noise?"""
    return helper_affine(
        reward_hids=[32, 32],
        dataset_potential_hids=[4, 4],
        model_potential=True,
        model_potential_hids=[32, 32],
        total_timesteps=2 ** 18,
        learning_rate=1e-2,
        potential_noise=np.array([0.0, 1.0]),
        **kwargs,
    )


@common.mark_parametrize_dict("env_kwargs", ENVIRONMENTS)
@common.mark_parametrize_kwargs(dict(common.combine_dicts(ARCHITECTURES, SYNTHETIC_TEST)))
def test_compare_synthetic(
    graph: tf.Graph,
    session: tf.Session,
    rel_upperbound: float,
    fudge_factor: float,
    rescale: bool,
    env_kwargs,
    kwargs,
):
    """Try comparing randomly generated reward models, same scale."""
    with graph.as_default():
        with session.as_default():
            noise = np.array([0.0, 0.5, 1.0])  # coarse-grained for speed
            df, _ = synthetic.compare_synthetic(
                reward_noise=noise,
                potential_noise=noise,
                model_affine=rescale,
                **env_kwargs,
                **kwargs,
            )
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                logging.info("Results: %s", df)

            for k in ["Intrinsic", "Shaping", "Extrinsic"]:
                assert (df[k] >= 0).all(axis=None), f"distances {k} should not be negative"

            # No reward noise, but potential noise
            no_rew_noise = df.loc[(0.0, slice(0.1, None)), :]
            rel = no_rew_noise["Intrinsic"] / no_rew_noise["Extrinsic"]
            assert rel.max(axis=None) < (rel_upperbound * fudge_factor)

            if not rescale:
                # When ground truth and noised reward are on the same scale,
                # shaping distance should increase proportionally with potential
                # magnitude. When reward-noise is non-zero there's a confounder as
                # the shaping noise we add can *cancel* with shaping in the reward
                # noise. So just consider zero reward noise.
                deltas = no_rew_noise["Shaping"].diff().dropna()  # first row is N/A
                assert deltas.min(axis=None) > 0.0
                mean_delta = deltas.mean()
                # Increment should be similar: allow it to vary by 2x up & down
                assert (deltas < mean_delta * 2).all(axis=None)
                assert (deltas > mean_delta * 0.5).all(axis=None)

            # We're no more than 10% of intrinsic upper bound at any point.
            # The upper bound is based on the magnitude of the reward noise
            # we added. It's an upper bound since it may include some potential
            # shaping, so we actually could find a shorter intrinsic distance.
            # Add 10% margin of error since we don't expect perfect optimization.
            some_noise = df.loc[df.index.get_level_values("Reward Noise") > 0.0]
            rel = some_noise["Intrinsic"] / some_noise["Intrinsic Upper Bound"]
            assert rel.max(axis=None) < (1 + 0.1 * fudge_factor)
