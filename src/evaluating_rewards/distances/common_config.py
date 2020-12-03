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

"""Sacred configs for distances: transitions factories and per-environment reward configurations.

Used in `plot_heatmap`, `epic`, `npec` and `erc`.
"""

import os
from typing import Any, Iterable, Mapping, Tuple

RewardCfg = Tuple[str, str]  # (type, path)
AggregatedDistanceReturn = Mapping[str, Mapping[Tuple[RewardCfg, RewardCfg], float]]


def _config_from_kinds(kinds: Iterable[str], **kwargs) -> Mapping[str, Any]:
    cfgs = [(kind, "dummy") for kind in kinds]
    res = dict(kwargs)
    res.update({"x_reward_cfgs": cfgs, "y_reward_cfgs": cfgs})
    return res


POINT_MASS_KINDS = [
    f"evaluating_rewards/PointMass{label}-v0"
    for label in ["SparseNoCtrl", "SparseWithCtrl", "DenseNoCtrl", "DenseWithCtrl", "GroundTruth"]
]
POINT_MAZE_KINDS = [
    "evaluating_rewards/PointMazeGroundTruthWithCtrl-v0",
    "evaluating_rewards/PointMazeGroundTruthNoCtrl-v0",
    "evaluating_rewards/PointMazeRepellentWithCtrl-v0",
    "evaluating_rewards/PointMazeRepellentNoCtrl-v0",
]
POINT_MAZE_LEARNED_CFGS = [
    ("evaluating_rewards/RewardModel-v0", "transfer_point_maze/reward/regress/model"),
    ("evaluating_rewards/RewardModel-v0", "transfer_point_maze/reward/preferences/model"),
    (
        "imitation/RewardNet_unshaped-v0",
        "transfer_point_maze/reward/irl_state_only/checkpoints/final/discrim/reward_net",
    ),
    (
        "imitation/RewardNet_unshaped-v0",
        "transfer_point_maze/reward/irl_state_action/checkpoints/final/discrim/reward_net",
    ),
]
MUJOCO_STANDARD_ORDER = [
    "ForwardNoCtrl",
    "ForwardWithCtrl",
    "BackwardNoCtrl",
    "BackwardWithCtrl",
]
COMMON_CONFIGS = {
    # evaluating_rewards/PointMass* environments.
    "point_mass": _config_from_kinds(
        POINT_MASS_KINDS, env_name="evaluating_rewards/PointMassLine-v0"
    ),
    # imitation/PointMaze{Left,Right}-v0 environments
    "point_maze": _config_from_kinds(
        POINT_MAZE_KINDS,
        env_name="imitation/PointMazeLeft-v0",
    ),
    # Compare rewards learned in imitation/PointMaze* to the ground-truth reward
    "point_maze_learned": {
        "env_name": "imitation/PointMazeLeftVel-v0",
        "x_reward_cfgs": [("evaluating_rewards/PointMazeGroundTruthWithCtrl-v0", "dummy")],
        "y_reward_cfgs": [
            ("evaluating_rewards/PointMazeBetterGoalWithCtrl-v0", "dummy"),
        ]
        + POINT_MAZE_LEARNED_CFGS,
    },
    # seals version of the canonical MuJoCo tasks
    "half_cheetah": _config_from_kinds(
        [
            f"evaluating_rewards/HalfCheetahGroundTruth{suffix}-v0"
            for suffix in MUJOCO_STANDARD_ORDER
        ],
        env_name="seals/HalfCheetah-v0",
    ),
    "hopper": _config_from_kinds(
        kinds=[
            f"evaluating_rewards/Hopper{prefix}{suffix}-v0"
            for prefix in ["GroundTruth", "Backflip"]
            for suffix in MUJOCO_STANDARD_ORDER
        ],
        env_name="seals/Hopper-v0",
    ),
}


def canonicalize_reward_cfg(reward_cfg: RewardCfg, data_root: str) -> RewardCfg:
    """Canonicalize path in reward configuration.

    Specifically, join paths with the `data_root`, unless they are the special "dummy" path.
    Also ensure the return value is actually of type RewardCfg: it is forgiving and will accept
    any iterable pair as input `reward_cfg`. This is important since Sacred has the bad habit of
    converting tuples to lists in configurations.

    Args:
        reward_cfg: Iterable of configurations to canonicailze.
        data_root: The root to join paths to.

    Returns:
        Canonicalized RewardCfg.
    """
    kind, path = reward_cfg
    if path != "dummy":
        path = os.path.join(data_root, path)
    return (kind, path)
