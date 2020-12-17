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

import glob
import itertools
import math
import os
from typing import Any, Iterable, List, Mapping, Optional, Tuple

from evaluating_rewards import serialize

RewardCfg = Tuple[str, str]  # (type, path)
AggregatedDistanceReturn = Mapping[str, Mapping[Tuple[RewardCfg, RewardCfg], float]]


def _config_from_kinds(kinds: Iterable[str], **kwargs) -> Mapping[str, Any]:
    cfgs = [(kind, "dummy") for kind in kinds]
    res = dict(kwargs)
    res.update({"x_reward_cfgs": cfgs, "y_reward_cfgs": cfgs})
    return res


_POINT_MAZE_CFG = [
    ("evaluating_rewards/RewardModel-v0", "regress/checkpoints/{}"),
    ("evaluating_rewards/RewardModel-v0", "preferences/checkpoints/{}"),
    ("imitation/RewardNet_unshaped-v0", "irl_state_only/checkpoints/{}/discrim/reward_net"),
    ("imitation/RewardNet_unshaped-v0", "irl_state_action/checkpoints/{}/discrim/reward_net"),
]


def point_maze_learned_cfgs(prefix: str = "transfer_point_maze") -> List[RewardCfg]:
    "Configurations for learned rewards in PointMaze."
    return [
        (kind, os.path.join(prefix, "reward", path.format("final")))
        for (kind, path) in _POINT_MAZE_CFG
    ]


def point_maze_learned_checkpoint_cfgs(
    prefix: str = "transfer_point_maze", target_num: Optional[int] = None
) -> List[RewardCfg]:
    """Configurations for learned rewards in PointMaze for each checkpoint.

    Args:
        prefix: The directory to locate results under.
        target_num: The target number of checkpoints to return for each algorithm.
            May return less than this if there are fewer checkpoints than `target_num`.
            May return up to twice this number.

    Returns:
        Reward configurations for checkpoints for each reward model.
    """
    res = {}
    for kind, fstr_path in _POINT_MAZE_CFG:
        glob_path = os.path.join(
            serialize.get_output_dir(), prefix, "reward", fstr_path.format("[0-9].*")
        )
        paths = sorted(glob.glob(glob_path))
        cfgs = [(kind, path) for path in paths]

        if target_num and len(cfgs) > target_num:
            subsample = math.floor(len(cfgs) / target_num)
            cfgs = cfgs[::subsample]
            assert target_num <= len(cfgs) <= 2 * target_num

        res[kind] = cfgs

    empty = {k: bool(v) for k, v in res.items()}
    if any(empty) and not all(empty):
        raise ValueError(f"No checkpoints found for some algorithms: {empty}")

    return list(itertools.chain(*res.values()))


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


def _update_common_configs() -> None:
    base_cfg = {
        "env_name": "imitation/PointMazeLeftVel-v0",
        "x_reward_cfgs": [("evaluating_rewards/PointMazeGroundTruthWithCtrl-v0", "dummy")],
    }
    for suffix in ("", "_fast"):
        prefix = f"transfer_point_maze{suffix}"

        std_key = f"point_maze_learned{suffix}"
        std_cfgs = point_maze_learned_cfgs(prefix)
        COMMON_CONFIGS[std_key] = dict(
            **base_cfg,
            y_reward_cfgs=[
                ("evaluating_rewards/PointMazeBetterGoalWithCtrl-v0", "dummy"),
            ]
            + std_cfgs,
        )

        chk_key = f"point_maze_checkpoints{suffix}"
        chk_cfgs = point_maze_learned_checkpoint_cfgs(prefix)
        COMMON_CONFIGS[chk_key] = dict(**base_cfg, y_reward_cfgs=chk_cfgs)


_update_common_configs()


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
