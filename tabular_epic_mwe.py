"""Minimal working example of tabular computation of EPIC on gridworld reward.

This is designed to replicate Figure A.2.(a) from https://arxiv.org/pdf/2006.13900.pdf.
"""

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from evaluating_rewards.analysis import gridworld_rewards

DeshapeFn = Callable[[np.ndarray, float], np.ndarray]


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


def _check_dist(dist: np.ndarray) -> None:
    assert np.allclose(np.sum(dist), 1)
    assert np.all(dist >= 0)


def fully_connected_random_canonical_reward(
    rew: np.ndarray,
    discount: float,
    state_dist: Optional[np.ndarray] = None,
    action_dist: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute version of rew with canonicalized shaping.

    Args:
        rew: The three-dimensional reward array to canonicalize.
        discount: The discount rate of the MDP.
        state_dist: Distribution over next states. Uniform if unspecified.
        action_dist: Distribution over actions. Uniform if unspecified.

    Returns:
        Shaped version of rew. Specifically, this corresponds to the advantage under
        transition dynamics where next states are chosen according to state_dist and a policy
        chooses actions according to action_dist. This return value is the same for any
        shaped version of rew.
    """
    assert 0 <= discount <= 1
    ns, _na, ns2 = rew.shape
    assert ns == ns2

    if state_dist is not None:
        _check_dist(state_dist)
    if action_dist is not None:
        _check_dist(action_dist)

    mean_rew_sa = np.average(rew, axis=2, weights=state_dist)
    mean_rew_s = np.average(mean_rew_sa, axis=1, weights=action_dist)
    mean_rew = np.average(mean_rew_s, axis=0, weights=state_dist)
    # In the infinite-horizon discounted case, the value function is:
    # V(s) = mean_rew_s + discount / (1 - discount) * mean_rew
    # So shaping gives:
    # R^{PC} = shape(rew, mean_rew_s, discount)
    #        + (discount - 1) * discount / (1 - discount) * mean_rew
    #        = shape(rew, mean_rew_s, discount) - mean_rew
    # In the finite-horizon undiscounted case, the value function is:
    # V_T(s) = mean_rew_s[s] + T*mean_rew
    # So shaping gives:
    # R^{PC}(s,a,s') = rew[s,a,s'] + V_{T - 1}(s') - V_{T-1}(s)
    #                = rew[s,a,s'] + mean_rew_s[s'] - mean_rew_s[s] - mean_rew
    #                = shape(rew, mean_rew, 1) - 1 * mean_rew
    # So pleasingly the same formula works for the discounted infinite-horizon and undiscounted
    # finite-horizon case.
    return shape(rew, mean_rew_s, discount) - discount * mean_rew


def state_to_3d(reward: np.ndarray, ns: int, na: int) -> np.ndarray:
    """Convert state-only reward R[s] to 3D reward R[s,a,s'].

    Args:
        - reward: state only reward.
        - ns: number of states.
        - na: number of actions.

    Returns:
        State-action-next state reward from tiling `reward`.
    """
    assert reward.ndim == 1
    assert reward.shape[0] == ns
    return np.tile(reward[:, np.newaxis, np.newaxis], (1, na, ns))


def grid_to_3d(reward: np.ndarray) -> np.ndarray:
    """Convert gridworld state-only reward R[i,j] to 3D reward R[s,a,s']."""
    assert reward.ndim == 2
    reward = reward.flatten()
    ns = reward.shape[0]
    return state_to_3d(reward, ns, 5)


def make_reward(cfg: Dict[str, np.ndarray], discount: float) -> np.ndarray:
    """Create reward from state-only reward and potential."""
    state_reward = grid_to_3d(cfg["state_reward"])
    potential = cfg["potential"]
    assert potential.ndim == 2  # gridworld, (i,j) indexed
    potential = potential.flatten()
    return shape(state_reward, potential, discount)


def _center(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = np.average(x, weights=weights)
    return x - mean


def pearson_distance(
    rewa: np.ndarray, rewb: np.ndarray, dist: Optional[np.ndarray] = None
) -> float:
    """Computes pseudometric derived from the Pearson correlation coefficient.

    It is invariant to positive affine transformations like the Pearson correlation coefficient.

    Args:
        rewa: A reward array.
        rewb: A reward array.
        dist: Optionally, a probability distribution of the same shape as rewa and rewb.

    Returns:
        Computes the Pearson correlation coefficient rho, optionally weighted by dist.
        Returns the square root of 1 minus rho.
    """
    if dist is None:
        dist = np.ones_like(rewa) / np.product(rewa.shape)
    _check_dist(dist)
    assert rewa.shape == dist.shape
    assert rewa.shape == rewb.shape

    dist = dist.flatten()
    rewa = _center(rewa.flatten(), dist)
    rewb = _center(rewb.flatten(), dist)

    vara = np.average(np.square(rewa), weights=dist)
    varb = np.average(np.square(rewb), weights=dist)
    cov = np.average(rewa * rewb, weights=dist)
    corr = cov / (np.sqrt(vara) * np.sqrt(varb))
    corr = min(corr, 1.0)  # floating point error sometimes rounds above 1.0

    return np.sqrt(0.5 * (1 - corr))


def deshape_pearson_distance(
    rewa: np.ndarray,
    rewb: np.ndarray,
    discount: float,
    deshape_fn: DeshapeFn,
    dist: Optional[np.ndarray] = None,
) -> float:
    """
    Computes Pearson distance between deshaped versions of rewa and rewb.

    Args:
        rewa: A three-dimensional reward array.
        rewb: A three-dimensional reward array.
        discount: The discount rate of the MDP.
        deshape_fn: The function to canonicalize the shaping component of the reward.
        dist: The measure for the Pearson distance.

    Returns:
        The Pearson distance between the deshaped versions of `rewa` and `rewb`.
    """
    rewa = deshape_fn(rewa, discount)
    rewb = deshape_fn(rewb, discount)
    return pearson_distance(rewa, rewb, dist)


def construct_rewards(discount: float):
    reward_keys = ("sparse_goal", "transformed_goal", "center_goal",
                   "sparse_penalty", "dirt_path", "cliff_walk")
    rewards = {k: gridworld_rewards.REWARDS[k] for k in reward_keys}
    return {k: make_reward(v, discount=discount) for k, v in rewards.items()}


def main(discount: float = 0.99):
    rewards = construct_rewards(discount)

    divergence = {}
    for src_key, src_rew in rewards.items():
        divergence[src_key] = {}
        for dst_key, dst_rew in rewards.items():
            div = deshape_pearson_distance(
                src_rew,
                dst_rew,
                discount=discount,
                deshape_fn=fully_connected_random_canonical_reward,
            )
            divergence[src_key][dst_key] = div
    pd.set_option('max_columns', 10)
    print(pd.DataFrame(divergence).round(4))


if __name__ == "__main__":
    main()