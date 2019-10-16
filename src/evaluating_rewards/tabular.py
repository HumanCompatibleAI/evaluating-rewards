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

"""Experiments with tabular (i.e. finite state) reward models."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def random_state_only_reward(
    n_states: int, n_actions: int, rng: np.random.RandomState = np.random
) -> np.ndarray:
    """Generates a random reward matrix, differing only in first axis.

    Args:
        n_states: The number of states.
        n_actions: The number of actions.
        rng: Random number generator.

    Returns:
        A three-dimensional array R, where R[s,a,s'] is the reward starting at state
        s, taking action a, and transitioning to state s'.
    """
    rew = rng.rand(n_states, 1, 1)
    return np.tile(rew, (1, n_actions, n_states))


def random_reward(
    n_states: int, n_actions: int, rng: np.random.RandomState = np.random
) -> np.ndarray:
    """Generates a random reward matrix.

    Args:
        n_states: The number of states.
        n_actions: The number of actions.
        rng: Random number generator.

    Returns:
        A three-dimensional array R, where R[s,a,s'] is the reward starting at state
        s, taking action a, and transitioning to state s'.
    """
    return rng.rand(n_states, n_actions, n_states)


def random_potential(n_states: int, rng: np.random.RandomState = np.random) -> np.ndarray:
    r"""Generates a random potential function.

    Args:
        n_states: The number of states.
        rng: Random number generator.

    Returns:
        A one-dimensional potential $$\phi$$.
    """
    return rng.rand(n_states)


def shape(reward: np.ndarray, potential: np.ndarray) -> np.ndarray:
    """Adds a potential-based shaping to a reward.

    Args:
        reward: The three-dimensional reward array.
        potential: The state-only potential function.

    Returns:
        reward shaped by potential.
    """
    assert reward.ndim == 3
    assert potential.ndim == 1
    return reward + potential[np.newaxis, np.newaxis, :] - potential[:, np.newaxis, np.newaxis]


def extract_potential(shaped: np.ndarray, unshaped: np.ndarray) -> np.ndarray:
    """Reverses `shape`, up to a constant."""
    assert shaped.ndim == 3
    assert shaped.shape == unshaped.shape
    delta = shaped - unshaped
    potential = delta[0, 0, :]

    # Make sure potential really is a potential for shaped
    assert np.allclose(shape(unshaped, potential), shaped)

    return potential


def closest_potential(reward: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Finds the potential which makes reward closest to target."""
    assert reward.ndim == 3
    assert reward.shape == target.shape

    # Compute coefficients and targets
    n_states, n_actions, _ = reward.shape
    eye = np.eye(n_states)
    x_vals = eye[np.newaxis, np.newaxis, :, :] - eye[:, np.newaxis, np.newaxis, :]
    x_vals = x_vals.repeat(n_actions, axis=1)
    y_vals = target - reward

    # Flatten for linear regression
    x_vals = x_vals.reshape(-1, n_states)
    y_vals = y_vals.flatten()

    # TODO(): report convergence-related statistics such as singular values?
    potential, _, _, _ = np.linalg.lstsq(x_vals, y_vals)

    return potential


def summary_comparison(reward1: np.ndarray, reward2: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Compares rewards in terms of intrinsic and shaping difference."""
    potential = closest_potential(reward1, reward2)
    closest_reward1 = shape(reward1, potential)
    intrinisic_difference = np.linalg.norm(reward2 - closest_reward1)

    potential_2d = potential[:, np.newaxis]
    potential_delta = potential_2d.T - potential_2d
    shaping_difference = np.linalg.norm(potential_delta)

    return intrinisic_difference, shaping_difference, potential


def make_shaped_reward(n_states: int, n_actions: int, seed: Optional[int] = None):
    """Creates random reward, potential and potential-shaped reward."""
    rng = None
    if seed is not None:
        rng = np.random.RandomState(seed=seed)

    reward = random_reward(n_states, n_actions, rng=rng)
    potential = random_potential(n_states, rng=rng)
    shaped = shape(reward, potential)

    return reward, potential, shaped


def potential_difference(p1, p2):
    p1 = p1.flatten()
    p2 = p2.flatten()
    p1 = p1 - p1[0]
    p2 = p2 - p2[0]
    return np.linalg.norm(p1 - p2)


def experiment_shaping_comparison(
    n_states: int,
    n_actions: int,
    reward_noise: Optional[np.ndarray] = None,
    potential_noise: Optional[np.ndarray] = None,
    state_only: bool = True,
) -> pd.DataFrame:
    """Compares rewards with varying noise to a ground-truth reward."""

    if reward_noise is None:
        reward_noise = np.arange(0.0, 1.0, 0.2)
    if potential_noise is None:
        potential_noise = np.arange(0.0, 10.0, 2.0)

    random_reward_fn = random_state_only_reward if state_only else random_reward
    reward = random_reward_fn(n_states, n_actions)
    additive_noise = random_reward_fn(n_states, n_actions)
    noised_reward_potential = random_potential(n_states)

    intrinsics = {}
    shapings = {}
    potential_deltas = {}
    real_intrinsics = {}

    for reward_nm in reward_noise:
        for pot_nm in potential_noise:
            noised_reward = reward + reward_nm * additive_noise
            noised_shaped = shape(noised_reward, pot_nm * noised_reward_potential)

            # These statistics could be computed in a real application
            intrinsic, shaping, potential = summary_comparison(noised_shaped, reward)
            intrinsics[(reward_nm, pot_nm)] = intrinsic
            shapings[(reward_nm, pot_nm)] = shaping

            # These could not be computed 'in the wild', but we can compute them
            # since we know how the reward models were constructed
            potential_delta = potential_difference(-potential, pot_nm * noised_reward_potential)
            potential_deltas[(reward_nm, pot_nm)] = potential_delta
            real_intrinsic = np.linalg.norm(noised_reward - reward)
            real_intrinsics[(reward_nm, pot_nm)] = real_intrinsic

    df = pd.DataFrame(
        {
            "Intrinsic": intrinsics,
            "Shaping": shapings,
            # Note since the reward noise may effectively include shaping,
            # we would expect a non-zero delta (in an l2 norm).
            "Potential Delta": potential_deltas,
            "Real Intrinsic": real_intrinsics,
        }
    )
    df.index.names = ["Reward Noise", "Potential Noise"]
    return df
