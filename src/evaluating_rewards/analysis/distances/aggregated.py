# Copyright 2020 Adam Gleave
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

"""Helper methods to manipulate aggregated values returned by scripts.distances.*"""

from typing import Iterable, Mapping, Tuple

import pandas as pd

from evaluating_rewards.scripts.distances import common


def oned_mapping_to_series(
    vals: Mapping[Tuple[common.RewardCfg, common.RewardCfg], float]
) -> pd.Series:
    """Converts mapping to a series.

    Args:
        vals: A mapping from pairs of configurations to a float.

    Returns:
        A Series with a multi-index based on the configurations.
    """
    vals = {(xtype, xpath, ytype, ypath): v for ((xtype, xpath), (ytype, ypath)), v in vals.items()}
    vals = pd.Series(vals)
    vals.index.names = [
        "target_reward_type",
        "target_reward_path",
        "source_reward_type",
        "source_reward_path",
    ]
    return vals


def select_subset(
    vals: common.AggregatedDistanceReturn,
    x_reward_cfgs: Iterable[common.RewardCfg],
    y_reward_cfgs: Iterable[common.RewardCfg],
) -> common.AggregatedDistanceReturn:
    """Selects the subset of `vals` specified in `{x,y}_reward_cfgs`.

    Raises:
        KeyError if (x,y) not present in `vals` for any pair from `{x,y}_reward_cfgs`.
    """
    return {
        k: {(x, y): v[(x, y)] for x in x_reward_cfgs for y in y_reward_cfgs}
        for k, v in vals.items()
    }
