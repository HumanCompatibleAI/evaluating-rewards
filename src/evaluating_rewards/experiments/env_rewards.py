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
    ("seals/HalfCheetah-v0", "evaluating_rewards/HalfCheetahGroundTruthForwardWithCtrl-v0"): 1800,
    # We've reached 2082
    ("seals/Hopper-v0", "evaluating_rewards/HopperGroundTruthForwardWithCtrl-v0"): 2000,
    # We've reached -48 with RL; rendering this looked close to optimal,
    # it went directly to the goal and stayed there with little oscillation.
    # PointMassPolicy (hardcoded) gets around -160.
    ("evaluating_rewards/PointMassLine-v0", "evaluating_rewards/PointMassGroundTruth-v0"): -55,
    # We've reached -5 in PointMaze*, but it's very variable across seed
    ("imitation/PointMazeLeftVel-v0", "evaluating_rewards/PointMazeGroundTruthWithCtrl-v0"): -6.5,
    ("imitation/PointMazeRightVel-v0", "evaluating_rewards/PointMazeGroundTruthWithCtrl-v0"): -6.5,
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
