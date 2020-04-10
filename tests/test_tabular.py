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

"""Test evaluating_rewards.tabular.*"""

import numpy as np
import pytest

from evaluating_rewards import tabular

DESHAPE_FN = [
    tabular.singleton_shaping_canonical_reward,
    tabular.fully_connected_random_canonical_reward,
    tabular.fully_connected_greedy_canonical_reward,
]


@pytest.mark.parametrize("deshape_fn", DESHAPE_FN)
def test_canonical(
    deshape_fn: tabular.DeshapeFn, n_states: int = 10, n_actions: int = 5, discount: float = 0.99,
) -> None:
    """Tests reward canonicalization and distance metrics based off this."""
    base_rew, _potential, shaped_rew = tabular.make_shaped_reward(n_states, n_actions, discount)

    deshaped_base = deshape_fn(base_rew, discount)
    deshaped_shaped = deshape_fn(shaped_rew, discount)
    assert np.allclose(deshaped_base, deshaped_shaped)

    equiv_rew = np.random.uniform(0, 10.0) * shaped_rew
    canon_base = tabular.canonical_reward(base_rew, discount, deshape_fn)
    canon_equiv = tabular.canonical_reward(equiv_rew, discount, deshape_fn)
    assert np.allclose(canon_base, canon_equiv)

    dist_equiv = tabular.canonical_reward_distance(base_rew, equiv_rew, discount, deshape_fn)
    assert np.allclose(dist_equiv, 0)

    dist_opposite = tabular.canonical_reward_distance(base_rew, -equiv_rew, discount, deshape_fn)
    # Distance should be large for opposite rewards.
    # For deshape_fn `fully_connected_random_canonical_reward`, it will be exactly equal to one.
    # For others it may be smaller since they take greedy policies, introducing an asymmetry
    # when rewards from a state differ depending on action.
    assert 0.5 < dist_opposite <= 1
