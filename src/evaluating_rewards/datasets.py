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
import dataclasses
import functools
from typing import Callable, ContextManager, Iterator, Optional, Sequence, TypeVar, Union

import gym
from imitation.data import rollout, types
from imitation.policies import serialize
from imitation.util import util
import numpy as np
from stable_baselines.common import base_class, policies, vec_env
import tensorflow as tf

T = TypeVar("T")
TrajectoryCallable = Callable[[rollout.GenTrajTerminationFn], Sequence[types.Trajectory]]
DatasetCallable = Callable[[int], T]
"""int parameter specifies number of timesteps."""
TransitionsCallable = DatasetCallable[types.Transitions]
SampleDist = DatasetCallable[np.ndarray]

C = TypeVar("C")
Factory = Callable[..., ContextManager[C]]
TrajectoryFactory = Factory[TrajectoryCallable]
TransitionsFactory = Factory[TransitionsCallable]
SampleDistFactory = Factory[SampleDist]

# *** Conversion functions ***


@contextlib.contextmanager
def transitions_factory_from_trajectory_factory(
    trajectory_factory: TrajectoryFactory,
    **kwargs,
) -> Iterator[TransitionsCallable]:
    """Generates and flattens trajectories until target timesteps reached, truncating if overshot.

    Args:
        trajectory_factory: The factory to sample trajectories from.
        kwargs: Passed through to the factory.

    Yields:
        A function that will perform the sampling process described above for a
        number of transitions `n` specified in the argument.
    """
    with trajectory_factory(**kwargs) as trajectory_callable:

        def f(total_timesteps: int) -> types.Transitions:
            trajs = trajectory_callable(sample_until=rollout.min_timesteps(total_timesteps))
            trans = rollout.flatten_trajectories(trajs)
            assert len(trans) >= total_timesteps
            as_dict = dataclasses.asdict(trans)
            truncated = {k: arr[:total_timesteps] for k, arr in as_dict.items()}
            return dataclasses.replace(trans, **truncated)

        yield f


@contextlib.contextmanager
def transitions_factory_iid_from_sample_dist_factory(
    obs_dist_factory: SampleDistFactory,
    act_dist_factory: SampleDistFactory,
    obs_kwargs=None,
    act_kwargs=None,
    **kwargs,
) -> Iterator[TransitionsCallable]:
    """Samples state and next state i.i.d. from `obs_dist` and actions i.i.d. from `act_dist`.

    This is an extremely weak prior. It's most useful in conjunction with methods in
    `canonical_sample` which assume i.i.d. transitions internally.
    """

    obs_kwargs = obs_kwargs or {}
    act_kwargs = act_kwargs or {}
    with obs_dist_factory(**obs_kwargs, **kwargs) as obs_dist:
        with act_dist_factory(**act_kwargs, **kwargs) as act_dist:

            def f(total_timesteps: int) -> types.Transitions:
                obses = obs_dist(total_timesteps)
                acts = act_dist(total_timesteps)
                next_obses = obs_dist(total_timesteps)
                dones = np.zeros(total_timesteps, dtype=np.bool)
                return types.Transitions(
                    obs=np.array(obses),
                    acts=np.array(acts),
                    next_obs=np.array(next_obses),
                    dones=dones,
                    infos=None,
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
def transitions_factory_to_sample_dist_factory(
    transitions_factory: TransitionsFactory, obs: bool, **kwargs
) -> Iterator[SampleDist]:
    """Converts TransitionsFactory to a SampleDistFactory.

    See `transitions_callable_to_sample_dist`.
    """
    with transitions_factory(**kwargs) as transitions_callable:
        yield transitions_callable_to_sample_dist(transitions_callable, obs)


# *** Trajectory factories ***


@contextlib.contextmanager
def _factory_via_serialized(
    factory_from_policy: Callable[[vec_env.VecEnv, policies.BasePolicy], T],
    env_name: str,
    policy_type: str,
    policy_path: str,
    parallel: bool = True,
    **kwargs,
) -> Iterator[T]:
    venv = util.make_vec_env(env_name, parallel=parallel, **kwargs)
    with tf.device("/cpu:0"):
        # It's normally faster to do policy inference on CPU, since batch sizes are small.
        with serialize.load_policy(policy_type, policy_path, venv) as policy:
            with factory_from_policy(venv, policy) as generator:
                yield generator


@contextlib.contextmanager
def trajectory_factory_from_policy(
    venv: vec_env.VecEnv, policy: Union[base_class.BaseRLModel, policies.BasePolicy]
) -> Iterator[TrajectoryCallable]:
    """Generator returning rollouts from a policy in a given environment."""

    def f(sample_until: rollout.GenTrajTerminationFn) -> Sequence[types.Trajectory]:
        return rollout.generate_trajectories(policy, venv, sample_until=sample_until)

    yield f


trajectory_factory_from_serialized_policy = functools.partial(
    _factory_via_serialized, trajectory_factory_from_policy
)


@contextlib.contextmanager
def trajectory_factory_noise_wrapper(
    factory: TrajectoryFactory,
    noise_env_name: str,
    obs_noise: Optional[SampleDist] = None,
    obs_noise_scale: float = 1.0,
    acts_noise: Optional[SampleDist] = None,
    acts_noise_scale: float = 1.0,
    **kwargs,
) -> Iterator[TrajectoryCallable]:
    """Adds noise to transitions from `factory`.

    Args:
        factory: The factory to add noise to.
        noise_env_name: The Gym identifier for the environment.
        obs_noise: A distribution to sample additive noise for `obs` from; if unspecified,
                   then samples from the observation space of `env_name`.
        obs_noise_scale: Multiplier for `obs_noise`.
        acts_noise: A distribution to sample additive noise for `acts` from; if unspecified,
                   then samples from the action space of `env_name`.
        acts_noise_scale: Multiplier for `acts_noise`.

    Yields:
        A function that will perform the sampling process described above for a
        number of trajectories `n` specified in the argument.
    """

    with contextlib.ExitStack() as stack:
        obs_noise = obs_noise or stack.enter_context(
            sample_dist_from_env_name(noise_env_name, obs=True)
        )
        acts_noise = acts_noise or stack.enter_context(
            sample_dist_from_env_name(noise_env_name, obs=False)
        )

        with factory(**kwargs) as trajectory_callable:

            def f(sample_until: rollout.GenTrajTerminationFn) -> Sequence[types.Trajectory]:
                trajs = trajectory_callable(sample_until)
                res = []
                for traj in trajs:
                    new_obs = traj.obs + obs_noise_scale * obs_noise(len(traj.obs))
                    new_acts = traj.acts + acts_noise_scale * acts_noise(len(traj.acts))
                    traj = dataclasses.replace(traj, obs=new_obs, acts=new_acts)
                    res.append(traj)
                return res

            yield f


# *** Transition factories ***


@contextlib.contextmanager
def transitions_factory_from_policy(
    venv: vec_env.VecEnv, policy: Union[base_class.BaseRLModel, policies.BasePolicy]
) -> Iterator[TransitionsCallable]:
    """Generator returning rollouts from a policy in a given environment."""

    def f(total_timesteps: int) -> types.Transitions:
        # TODO(adam): inefficient -- discards partial trajectories and resets environment
        return rollout.generate_transitions(
            policy, venv, n_timesteps=total_timesteps, truncate=True
        )

    yield f


transitions_factory_from_serialized_policy = functools.partial(
    _factory_via_serialized, transitions_factory_from_policy
)


@contextlib.contextmanager
def transitions_factory_from_random_model(
    env_name: str, seed: int = 0
) -> Iterator[TransitionsCallable]:
    """Randomly samples state and action and computes next state from dynamics.

    This is one of the weakest possible priors, with broad support. It is similar
    to `transitions_factory_from_policy` with a random policy, with two key differences.
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

    def f(total_timesteps: int) -> types.Transitions:
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
        return types.Transitions(
            obs=np.array(obses),
            acts=np.array(acts),
            next_obs=np.array(next_obses),
            dones=dones,
            infos=None,
        )

    yield f


@contextlib.contextmanager
def transitions_factory_permute_wrapper(
    factory: TransitionsFactory,
    min_buffer: int = 4096,
    rng: np.random.RandomState = np.random,
    **kwargs,
) -> Iterator[TransitionsCallable]:
    """Permutes states and actions uniformly at random in transitions from `factory`.

    Samples transitions from `factory`, to obtain a buffer of `multiplier` times the number of
    requested transitions `n`. Returns a random slice of the transitions, sampled without
    replacement independently from all states (i.e. starting and next states) and actions.
    The returned transitions are then deleted from the buffer; in subsequent calls, `n`
    transitions are sampled. In this way, the buffer only imposes a one-time overhead.

    The effect of this is to return transitions that `callable` might never generate. This is
    useful when you wish to have counterfactual transitions, but stay close to a realistic
    state and action distribution implicitly given by `callable`.

    Args:
        factory: The factory to sample transitions from.
        min_buffer: Minimum size of buffer to sample from. This is desirable to ensure the samples
            are appropriately mixed, e.g. not all from a single episode.
        rng: Random state.
        kwargs: passed through to factory.

    Yields:
        A function that will perform the sampling process described above for a
        number of timesteps `n` specified in the argument.
    """
    buf = {k: np.empty((0)) for k in ["obs", "acts", "next_obs", "dones"]}
    with factory(**kwargs) as transitions_callable:

        def f(n: int) -> types.Transitions:
            target_size = max(min_buffer, n)
            delta = target_size - len(buf["obs"])
            if delta > 0:
                transitions = transitions_callable(delta)
                assert len(transitions.obs) == delta
                for k, v in buf.items():
                    new_v = getattr(transitions, k)
                    if len(v) > 0:
                        new_v = np.concatenate((v, new_v))
                    buf[k] = new_v

                # Note this assert may not hold outside this branch: if f was previously called
                # with a larger `n`, then `len(buf["obs"])` may be greater than `target_size`.
                assert len(buf["obs"]) == target_size

            assert len(buf["obs"]) >= target_size
            idxs = {k: rng.choice(target_size, size=n, replace=False) for k in buf.keys()}
            res = {k: buf[k][idx] for k, idx in idxs.items()}
            res = types.Transitions(**res, infos=None)

            for k, idx in idxs.items():
                buf[k] = np.delete(buf[k], idx, axis=0)

            return res

        yield f


# *** Sample distribution factories ***


@contextlib.contextmanager
def sample_dist_from_space(space: gym.Space, seed: int = 0) -> Iterator[SampleDist]:
    """Creates function to sample `n` elements from from `space`."""
    space.seed(seed)

    def f(n: int) -> np.ndarray:
        return np.array([space.sample() for _ in range(n)])

    yield f


@contextlib.contextmanager
def sample_dist_from_env_name(env_name: str, obs: bool, **kwargs) -> Iterator[SampleDist]:
    env = gym.make(env_name)
    try:
        space = env.observation_space if obs else env.action_space
    finally:
        env.close()
    with sample_dist_from_space(space, **kwargs) as sample_dist:
        yield sample_dist
