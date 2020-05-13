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

"""Unit tests for evaluating_rewards.util."""

import numpy as np
import pytest

from evaluating_rewards import tabular, util

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
    actual = util.cross_distance(
        test_case["rewxs"],
        test_case["rewys"],
        distance_fn=tabular.direct_distance,
        parallelism=parallelism,
        threading=threading,
    )
    assert test_case["expected"] == actual
