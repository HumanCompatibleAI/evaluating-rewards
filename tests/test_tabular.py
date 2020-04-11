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

from typing import Tuple

import hypothesis
from hypothesis import strategies as st
from hypothesis.extra import numpy as hp_numpy
import numpy as np
import pytest

from evaluating_rewards import tabular


# pylint:disable=no-value-for-parameter
# pylint gets confused with hypothesis draw magic
@st.composite
def distribution(draw, shape) -> np.ndarray:
    """Search strategy for a probability distribution of given shape."""
    nonneg_elements = st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False)
    arr = draw(hp_numpy.arrays(np.float, shape, elements=nonneg_elements, fill=st.nothing()))
    hypothesis.assume(np.any(arr > 0))
    return arr / np.sum(arr)


def numeric_float(max_abs: float = 1e4) -> st.SearchStrategy:
    """Search strategy for numeric (non-inf, non-NaN) floats with bounded absolute value."""
    return st.floats(min_value=-max_abs, max_value=max_abs, allow_nan=False, allow_infinity=False)


@st.composite
def arr_and_distribution(draw) -> Tuple[np.ndarray, np.ndarray]:
    """Search strategy for array and a probability distribution of same shape as array."""
    shape = draw(hp_numpy.array_shapes())
    arr = draw(
        hp_numpy.arrays(dtype=np.float, shape=shape, elements=numeric_float(), fill=st.nothing())
    )
    dist = draw(distribution(shape))
    return arr, dist


@hypothesis.given(
    arr_dist=arr_and_distribution(), p=st.integers(min_value=1, max_value=10), scale=numeric_float()
)
def test_weighted_lp_norm(arr_dist: Tuple[np.ndarray, np.ndarray], p: int, scale: float) -> None:
    """Test for tabular.weighted_lp_norm."""
    arr, dist = arr_dist
    norm = tabular.weighted_lp_norm(arr, p, dist)
    # Non-negativity
    assert norm >= 0

    # Absolute homogeneity
    norm_scaled = tabular.weighted_lp_norm(scale * arr, p, dist)
    assert norm_scaled == scale * norm
    norm_neg = tabular.weighted_lp_norm(-arr, p, dist)
    assert norm_neg == norm


@st.composite
def reward(
    draw,
    n_states=st.integers(min_value=1, max_value=10),
    n_actions=st.integers(min_value=1, max_value=10),
) -> np.ndarray:
    """Search strategy for reward function with specified numbers of states and actions."""
    ns = draw(n_states)
    na = draw(n_actions)
    shape = (ns, na, ns)
    rew = draw(hp_numpy.arrays(np.float, shape, elements=numeric_float(), fill=st.nothing()))
    return rew


_default_discount = st.floats(min_value=0.0, max_value=1.0)


@st.composite
def shaped_reward_pair(draw, discount=_default_discount) -> Tuple[np.ndarray, np.ndarray]:
    """Search strategy for a pair of rewards equivalent up to potential shaping."""
    rew = draw(reward())
    ns = rew.shape[0]
    potential = draw(hp_numpy.arrays(np.float, (ns,), elements=numeric_float(), fill=st.nothing()))
    gamma = draw(discount)
    return rew, gamma, tabular.shape(rew, potential, gamma)


@st.composite
def equiv_reward_pair(draw, discount=_default_discount) -> Tuple[np.ndarray, np.ndarray]:
    """Search strategy for a pair of rewards equivalent up to shaping and positive rescaling."""
    base_rew, gamma, shaped_rew = draw(shaped_reward_pair(discount))
    scale = draw(st.floats(min_value=1 / 100.0, max_value=100, exclude_min=True))
    equiv_rew = scale * shaped_rew
    return base_rew, gamma, equiv_rew


DESHAPE_FN = [
    tabular.singleton_shaping_canonical_reward,
    tabular.fully_connected_random_canonical_reward,
    tabular.fully_connected_greedy_canonical_reward,
]


@pytest.mark.parametrize("deshape_fn", DESHAPE_FN)
@hypothesis.given(base_shaped=shaped_reward_pair())
def test_deshape(deshape_fn: tabular.DeshapeFn, base_shaped) -> None:
    """Tests potential shaping invariance."""
    base_rew, discount, shaped_rew = base_shaped
    deshaped_base = deshape_fn(base_rew, discount)
    deshaped_shaped = deshape_fn(shaped_rew, discount)
    assert np.allclose(deshaped_base, deshaped_shaped, atol=1e-6)


@pytest.mark.parametrize("deshape_fn", DESHAPE_FN)
@hypothesis.given(base_equiv=equiv_reward_pair())
def test_canonical(deshape_fn: tabular.DeshapeFn, base_equiv) -> None:
    """Tests reward canonicalization."""
    base_rew, discount, equiv_rew = base_equiv
    canon_base = tabular.canonical_reward(base_rew, discount, deshape_fn)
    canon_equiv = tabular.canonical_reward(equiv_rew, discount, deshape_fn)
    assert np.allclose(canon_base, canon_equiv, atol=1e-6)


@pytest.mark.parametrize("deshape_fn", DESHAPE_FN)
@hypothesis.given(base_equiv=equiv_reward_pair())
def test_canonical_dist(deshape_fn: tabular.DeshapeFn, base_equiv) -> None:
    """Test distance from canonicalized rewards."""
    base_rew, discount, equiv_rew = base_equiv
    deshaped_base = deshape_fn(base_rew, discount)
    # Assume not near-zero rewards. This can cause problems with floating point errors
    # if things go slightly above/below round-to-zero thresholds.
    hypothesis.assume(np.linalg.norm(deshaped_base) > 1e-6)

    dist_equiv = tabular.canonical_reward_distance(base_rew, equiv_rew, discount, deshape_fn)
    assert np.allclose(dist_equiv, 0, atol=1e-6)

    dist_opposite = tabular.canonical_reward_distance(base_rew, -equiv_rew, discount, deshape_fn)
    # Distance should be large for opposite rewards.
    # For deshape_fn `fully_connected_random_canonical_reward`, it will be exactly equal to one.
    # For others it may be smaller since they take greedy policies, introducing an asymmetry
    # when rewards from a state differ depending on action.
    assert 0.3 < dist_opposite <= (1 + 1e-10)


@st.composite
def potential_only_reward(
    draw,
    n_states=st.integers(min_value=1, max_value=10),
    n_actions=st.integers(min_value=1, max_value=10),
    discount=_default_discount,
) -> np.ndarray:
    """Search strategy for a all-zero reward and a potential-shaping only reward."""
    ns = draw(n_states)
    na = draw(n_actions)
    rew = np.zeros((ns, na, ns))
    potential = draw(hp_numpy.arrays(np.float, (ns,), elements=numeric_float(), fill=st.nothing()))
    gamma = draw(discount)
    shaped_rew = tabular.shape(rew, potential, gamma)
    return rew, gamma, shaped_rew


@pytest.mark.parametrize("deshape_fn", DESHAPE_FN)
@hypothesis.given(shaped_pair=potential_only_reward())
def test_canonical_dist_near_zero(deshape_fn: tabular.DeshapeFn, shaped_pair) -> None:
    """Test distance from zero-equivalent (i.e. potential-shaping only) rewards."""
    zero_rew, discount, shaped_rew = shaped_pair

    dist_equiv = tabular.canonical_reward_distance(zero_rew, shaped_rew, discount, deshape_fn)
    assert np.allclose(dist_equiv, 0, atol=1e-6)

    # Since it is equivalent to zero, it is also equivalent to the negative of itself!
    dist_opposite = tabular.canonical_reward_distance(zero_rew, -shaped_rew, discount, deshape_fn)
    assert np.allclose(dist_opposite, 0, atol=1e-6)
