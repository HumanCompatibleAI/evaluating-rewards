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

from evaluating_rewards import rewards


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


def shape(reward: np.ndarray, potential: np.ndarray, discount: float) -> np.ndarray:
    """Adds a potential-based shaping to a reward.

    Args:
        reward: The three-dimensional reward array.
        potential: The state-only potential function.
        discount: The discount factor.

    Returns:
        reward shaped by potential.
    """
    assert reward.ndim == 3
    assert potential.ndim == 1
    new_pot = discount * potential[np.newaxis, np.newaxis, :]
    old_pot = potential[:, np.newaxis, np.newaxis]
    return reward + new_pot - old_pot


def closest_potential(reward: np.ndarray, target: np.ndarray, discount: float) -> np.ndarray:
    """Finds the squared-error minimizing potential shaping.

    Args:
        reward: the reward to transform.
        target: the target to match.
        discount: the discount factor.

    Returns:
        A state-array of potentials, such that `reward + discount * potential(s') - potential(s)`
        has minimal least squared-error.
    """
    assert reward.ndim == 3
    assert reward.shape == target.shape

    # Compute coefficients and targets
    n_states, n_actions, _ = reward.shape
    eye = np.eye(n_states)
    new_pot = discount * eye[np.newaxis, np.newaxis, :, :]
    old_pot = eye[:, np.newaxis, np.newaxis, :]
    x_vals = new_pot - old_pot
    x_vals = x_vals.repeat(n_actions, axis=1)
    y_vals = target - reward

    # Flatten for linear regression
    x_vals = x_vals.reshape(-1, n_states)
    y_vals = y_vals.flatten()

    # TODO(): report convergence-related statistics such as singular values?
    potential, _, _, _ = np.linalg.lstsq(x_vals, y_vals, rcond=None)

    return potential


def closest_reward_am(
    source: np.ndarray, target: np.ndarray, n_iter: int = 100, discount: float = 0.99
) -> np.ndarray:
    """Finds the least squared-error reward to target that is equivalent to reward.

    Alternating minimization over `closest_potential` and `closest_affine`.

    Args:
        - source: the source reward.
        - target: the reward to match.
        - n_iter: the number of iterations of expectation-maximization.
        - discount: The discount rate of the MDP.

    Returns:
        A reward that is equivalent to `source` with minimal squared-error to `target`.
    """
    closest_reward = source
    for _ in range(n_iter):
        potential = closest_potential(closest_reward, target, discount)
        closest_reward = shape(closest_reward, potential, discount)
        params = rewards.least_l2_affine(closest_reward.flatten(), target.flatten())
        closest_reward = closest_reward * params.scale + params.shift
    return closest_reward


def _check_rews_dist(rewa: np.ndarray, rewb: np.ndarray, dist: np.ndarray) -> None:
    assert rewa.shape == rewb.shape
    assert rewa.shape == dist.shape
    assert np.allclose(np.sum(dist), 1)
    assert np.all(dist >= 0)


def direct_sq_divergence(rewa: np.ndarray, rewb: np.ndarray, dist: Optional[np.ndarray]) -> float:
    """Direct divergence over uniform transition distribution with squared-error loss."""
    if dist is None:
        dist = np.ones_like(rewa) / np.product(rewa.shape)
    _check_rews_dist(rewa, rewb, dist)
    squared_error = np.square(rewa - rewb)
    weighted_sse = np.sum(squared_error * dist)
    return np.sqrt(weighted_sse)


def epic_distance(
    src_reward: np.ndarray, target_reward: np.ndarray, dist: Optional[np.ndarray] = None, **kwargs
) -> float:
    closest = closest_reward_am(src_reward, target_reward, **kwargs)
    return direct_sq_divergence(closest, target_reward, dist)


def _center(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = np.average(x, weights=weights)
    return x - mean


def pearson_distance(
    rewa: np.ndarray, rewb: np.ndarray, dist: Optional[np.ndarray] = None
) -> float:
    """Computes pseudometric derived from the Pearson correlation coefficient.

    It is invariant to positive affine transformations like the Pearson correlation coefficient.

    Args:
        rewa: One three-dimensional reward array.
        rewb: One three-dimensional reward array.
        dist: Optionally, a probability distribution of the same shape as rewa and rewb.

    Returns:
        Computes the Pearson correlation coefficient rho, optionally weighted by dist.
        Returns the square root of 1 minus rho.
    """
    if dist is None:
        dist = np.ones_like(rewa) / np.product(rewa.shape)
    _check_rews_dist(rewa, rewb, dist)

    dist = dist.flatten()
    rewa = _center(rewa.flatten(), dist)
    rewb = _center(rewb.flatten(), dist)

    vara = np.average(np.square(rewa), weights=dist)
    varb = np.average(np.square(rewb), weights=dist)
    cov = np.average(rewa * rewb, weights=dist)
    corr = cov / (np.sqrt(vara) * np.sqrt(varb))

    return np.sqrt(1 - corr)


def asymmetric_distance(
    source: np.ndarray, target: np.ndarray, dist: Optional[np.ndarray] = None, **kwargs
) -> float:
    """Minimal Pearson distance over rewards equivalent to source. This is a premetric.

    Args:
        source: The three-dimensional source reward array.
        target: The three-dimensional target reward array.
        dist: Optionally, a probability distribution of the same shape as source and target.
        **kwargs: Passed through to `closest_reward_am`.

    Returns:
        The minimal distance to target over rewards equivalent to source.
    """
    source_matched = closest_reward_am(source, target, **kwargs)
    return pearson_distance(source_matched, target, dist)


def symmetric_distance(rewa: np.ndarray, rewb: np.ndarray, **kwargs) -> float:
    """Symmetric version of `asymmetric_distance`. This is a pseudosemimetric.

    Args:
        rewa: One three-dimensional reward array.
        rewb: One three-dimensional reward array.
        **kwargs: Passed through to `asymmetric_distance`.

    Returns:
         The mean of `asymmetric_distance` from `rewa` to `rewb` and `rewb` to `rewa`.
    """
    dista = asymmetric_distance(rewa, rewb, **kwargs)
    distb = asymmetric_distance(rewb, rewa, **kwargs)
    return 0.5 * (dista + distb)


def summary_comparison(
    reward1: np.ndarray, reward2: np.ndarray, discount: float
) -> Tuple[float, float, np.ndarray]:
    """Compares rewards in terms of intrinsic and shaping difference."""
    potential = closest_potential(reward1, reward2, discount)
    closest_reward1 = shape(reward1, potential, discount)
    intrinisic_difference = np.linalg.norm(reward2 - closest_reward1)

    potential_2d = potential[:, np.newaxis]
    potential_delta = potential_2d.T - potential_2d
    shaping_difference = np.linalg.norm(potential_delta)

    return intrinisic_difference, shaping_difference, potential


def make_shaped_reward(
    n_states: int, n_actions: int, discount: float = 1.0, seed: Optional[int] = None
):
    """Creates random reward, potential and potential-shaped reward."""
    rng = None
    if seed is not None:
        rng = np.random.RandomState(seed=seed)

    reward = random_reward(n_states, n_actions, rng=rng)
    potential = random_potential(n_states, rng=rng)
    shaped = shape(reward, potential, discount)

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
    discount: float = 1.0,
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
            noised_shaped = shape(noised_reward, pot_nm * noised_reward_potential, discount)

            # These statistics could be computed in a real application
            intrinsic, shaping, potential = summary_comparison(noised_shaped, reward, discount)
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
