# Copyright 2019 Adam Gleave
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

"""Unit tests for evaluating_rewards.policies."""

from typing import Sequence, Set, Tuple

import gym
from imitation.policies import base
import numpy as np
import pytest

from evaluating_rewards import policies


class FixedPolicy(base.HardCodedPolicy):  # pylint:disable=abstract-method
    """Policy that always returns a fixed value."""

    def __init__(self, ob_space: gym.Space, ac_space: gym.Space, fixed_val: np.ndarray):
        super().__init__(ob_space, ac_space)
        self.fixed_val = fixed_val
        if not ac_space.contains(fixed_val):
            raise ValueError(f"fixed_val = '{fixed_val}' not contained in ac_space = '{ac_space}'")

    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        return self.fixed_val


def test_policy_mixture_validation():
    """Test input validation."""
    with pytest.raises(ValueError):
        policies.PolicyMixture(pols=[])

    space1 = gym.spaces.Box(low=0, high=1, shape=(2,))
    space2 = gym.spaces.Box(low=0, high=1, shape=(3,))
    space3 = gym.spaces.Box(low=0, high=0.5, shape=(2,))

    with pytest.raises(ValueError):
        policies.PolicyMixture(
            pols=[
                FixedPolicy(space1, space1, [0.5, 0.5]),
                FixedPolicy(space2, space2, [0.5, 0.5, 0.5]),
            ]
        )

    with pytest.raises(ValueError):
        policies.PolicyMixture(
            pols=[FixedPolicy(space1, space1, [0.5, 0.5]), FixedPolicy(space2, space3, [0.5, 0.5])]
        )


def _test_policy_mixture(
    n_policies: int, transition_p: float, n_steps: int, batch_size: int
) -> Tuple[Sequence[np.ndarray], Set[np.ndarray]]:
    space = gym.spaces.Box(low=0, high=1, shape=(2,))
    fixed_vals = np.array([space.sample() for _ in range(n_policies)])
    pols = [FixedPolicy(space, space, fixed_val) for fixed_val in fixed_vals]
    mixture = policies.PolicyMixture(pols=pols, transition_p=transition_p)

    actions = []
    for _ in range(n_steps):
        obs = np.array([space.sample() for _ in range(batch_size)])
        action, _, _, _ = mixture.step(obs)
        assert action.shape[0] == batch_size
        action = np.unique(action, axis=0)
        assert action.shape[0] == 1
        action = action[0, :]
        assert action in fixed_vals
        actions.append(action)

    return fixed_vals, np.unique(actions, axis=0)


def test_policy_mixture():
    """Test returned values are plausible."""
    fixed_vals, seen = _test_policy_mixture(
        n_policies=3, transition_p=0.5, n_steps=100, batch_size=4
    )
    assert np.all(np.sort(seen, axis=0) == np.sort(fixed_vals, axis=0))


def test_policy_mixture_never_change():
    """If transition_p is zero, policy should never switch."""
    _, seen = _test_policy_mixture(n_policies=2, transition_p=0.0, n_steps=100, batch_size=4)
    assert seen.shape[0] == 1
