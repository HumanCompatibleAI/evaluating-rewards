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

"""Methods to create dataset generators.

These are methods that yield batches of observation-action-next observation
triples, implicitly used to define a distribution which distance metrics
can be taken with respect to.
"""

import contextlib
from typing import Callable, ContextManager, Iterator, Union

import gym
from imitation.policies import serialize
from imitation.util import data, rollout, util
import numpy as np
from stable_baselines.common import base_class, policies, vec_env

SampleDist = Callable[[int], np.ndarray]
TransitionsCallable = Callable[[int], data.Transitions]
# Expect DatasetFactory to accept a str specifying env_name as first argument,
# int specifying seed as second argument and factory-specific keyword arguments
# after this. There is no way to specify this in Python type annotations yet :(
# See https://github.com/python/mypy/issues/5876
TransitionsFactory = Callable[..., ContextManager[TransitionsCallable]]
SampleDistFactory = Callable[..., ContextManager[SampleDist]]


@contextlib.contextmanager
def space_to_sample(space: gym.Space) -> Iterator[SampleDist]:
    """Creates function to sample `n` elements from from `space`."""

    def f(n: int) -> np.ndarray:
        return np.array([space.sample() for _ in range(n)])

    yield f


@contextlib.contextmanager
def env_name_to_sample(env_name: str, obs: bool) -> Iterator[SampleDist]:
    env = gym.make(env_name)
    space = env.observation_space if obs else env.action_space
    with space_to_sample(space) as sample_dist:
        yield sample_dist
    env.close()


@contextlib.contextmanager
def rollout_policy_generator(
    venv: vec_env.VecEnv, policy: Union[base_class.BaseRLModel, policies.BasePolicy]
) -> Iterator[TransitionsCallable]:
    """Generator returning rollouts from a policy in a given environment."""

    def f(total_timesteps: int) -> data.Transitions:
        # TODO(adam): inefficient -- discards partial trajectories and resets environment
        return rollout.generate_transitions(policy, venv, n_timesteps=total_timesteps)

    yield f


@contextlib.contextmanager
def rollout_serialized_policy_generator(
    env_name: str, policy_type: str, policy_path: str, num_vec: int = 8, seed: int = 0
) -> Iterator[TransitionsCallable]:
    venv = util.make_vec_env(env_name, n_envs=num_vec, seed=seed)
    with serialize.load_policy(policy_type, policy_path, venv) as policy:
        with rollout_policy_generator(venv, policy) as generator:
            yield generator


@contextlib.contextmanager
def random_transition_generator(env_name: str, seed: int = 0) -> Iterator[TransitionsCallable]:
    """Randomly samples state and action and computes next state from dynamics.

    This is one of the weakest possible priors, with broad support. It is similar
    to `rollout_generator` with a random policy, with two key differences.
    First, adjacent timesteps are independent from each other, as a state
    is randomly sampled at the start of each transition. Second, the initial
    state distribution is ignored. WARNING: This can produce physically impossible
    states, if there is no path from a feasible initial state to a sampled state.

    Args:
        env_name: The name of a Gym environment. It must be a ResettableEnv.
        seed: Used to seed the dynamics.

    Yields:
        A function that will perform the sampling process described above for a
        number of timesteps specified in the argument.
    """
    env = gym.make(env_name)
    env.seed(seed)

    def f(total_timesteps: int) -> data.Transitions:
        """Helper function."""
        obses = []
        acts = []
        next_obses = []
        for _ in range(total_timesteps):
            old_state = env.state_space.sample()
            obs = env.obs_from_state(old_state)
            act = env.action_space.sample()
            new_state = env.transition(old_state, act)  # may be non-deterministic
            next_obs = env.obs_from_state(new_state)

            obses.append(obs)
            acts.append(act)
            next_obses.append(next_obs)
        dones = np.zeros(total_timesteps, dtype=np.bool)
        return data.Transitions(
            obs=np.array(obses), acts=np.array(acts), next_obs=np.array(next_obses), dones=dones,
        )

    yield f


@contextlib.contextmanager
def iid_transition_generator(
    obs_dist: SampleDist, act_dist: SampleDist
) -> Iterator[TransitionsCallable]:
    """Samples state and next state i.i.d. from `obs_dist` and actions i.i.d. from `act_dist`.

    This is an extremely weak prior. It's most useful in conjunction with methods in
    `canonical_sample` which assume i.i.d. transitions internally."""

    def f(total_timesteps: int) -> data.Transitions:
        obses = obs_dist(total_timesteps)
        acts = act_dist(total_timesteps)
        next_obses = obs_dist(total_timesteps)
        dones = np.zeros(total_timesteps, dtype=np.bool)
        return data.Transitions(
            obs=np.array(obses), acts=np.array(acts), next_obs=np.array(next_obses), dones=dones,
        )

    yield f


def transitions_callable_to_sample_dist(
    transitions_callable: TransitionsCallable, obs: bool
) -> SampleDist:
    """Samples state/actions from batches returned by `batch_callable`.

    If `obs` is true, then samples observations from state and next state.
    If `obs` is false, then samples actions.
    """

    def f(n: int) -> np.ndarray:
        num_timesteps = ((n - 1) // 2 + 1) if obs else n
        transitions = transitions_callable(num_timesteps)
        if obs:
            res = np.concatenate((transitions.obs, transitions.next_obs))
        else:
            res = transitions.acts
        return res[:n]

    return f


@contextlib.contextmanager
def dataset_factory_to_sample_dist_factory(
    dataset_factory: TransitionsFactory, obs: bool, **kwargs
) -> Iterator[SampleDist]:
    with dataset_factory(**kwargs) as transitions_callable:
        yield transitions_callable_to_sample_dist(transitions_callable, obs)
