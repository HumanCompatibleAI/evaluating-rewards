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

"""Find hardcoded rewards associated with particular environments."""

import re
from typing import Iterable, Set

from evaluating_rewards import serialize

REWARDS_BY_ENV = {
    "seals/HalfCheetah-v0": ["evaluating_rewards/HalfCheetahGroundTruth.*-v0"],
    "seals/Hopper-v0": ["evaluating_rewards/Hopper.*-v0"],
    "evaluating_rewards/PointMassLine-v0": ["evaluating_rewards/PointMass.*-v0"],
    "imitation/PointMazeLeftVel-v0": ["evaluating_rewards/PointMaze.*-v0"],
    "imitation/PointMazeRightVel-v0": ["evaluating_rewards/PointMaze.*-v0"],
}
GENERIC_REWARDS = ["evaluating_rewards/Zero-v0"]
for env_name in REWARDS_BY_ENV:
    REWARDS_BY_ENV[env_name] += GENERIC_REWARDS

GROUND_TRUTH_REWARDS_BY_ENV = {
    "seals/HalfCheetah-v0": "evaluating_rewards/HalfCheetahGroundTruthForwardWithCtrl-v0",
    "seals/Hopper-v0": "evaluating_rewards/HopperGroundTruthForwardWithCtrl-v0",
    "evaluating_rewards/PointMassLine-v0": "evaluating_rewards/PointMassGroundTruth-v0",
    "imitation/PointMazeLeftVel-v0": "evaluating_rewards/PointMazeGroundTruthWithCtrl-v0",
    "imitation/PointMazeRightVel-v0": "evaluating_rewards/PointMazeGroundTruthWithCtrl-v0",
}

# Minimum return we expect to receive from an expert policy.
THRESHOLDS = {
    # PPO paper reports ~2000; SAC reports ~15,000; we've reached over 3000
    ("seals/HalfCheetah-v0", "evaluating_rewards/HalfCheetahGroundTruthForwardWithCtrl-v0"): 1800,
    # We've reached 2082; varies between seeds.
    ("seals/Hopper-v0", "evaluating_rewards/HopperGroundTruthForwardWithCtrl-v0"): 2000,
    # Performance is very variable. Can usually get below -80 with RL, have got as low as -48.
    # Random policies get around -550. PointMassPolicy (hardcoded) gets around -160.
    # Things above -80 look reasonably good but suboptimal, sometimes stop before touching the goal.
    # The -48 policy seemed close to optimal.
    ("evaluating_rewards/PointMassLine-v0", "evaluating_rewards/PointMassGroundTruth-v0"): -80,
    # Typically get reward between -8 and -10, some seed should get >-9
    ("imitation/PointMazeLeftVel-v0", "evaluating_rewards/PointMazeGroundTruthWithCtrl-v0"): -9,
    # PointMazeRight is similar.
    ("imitation/PointMazeRightVel-v0", "evaluating_rewards/PointMazeGroundTruthWithCtrl-v0"): -9,
}


def _matched(pattern: str, strings: Iterable[str]) -> Set[str]:
    pattern = re.compile(pattern)
    return set(x for x in strings if pattern.match(x))


def find_rewards(patterns: Iterable[str]) -> Set[str]:
    """Find all rewards in serialize.reward_registry matching one of patterns."""
    all_rewards = serialize.reward_registry.keys()
    matched_rewards = set()
    for pattern in patterns:
        matched = _matched(pattern, all_rewards)
        matched_rewards = matched_rewards.union(matched)
    return matched_rewards
