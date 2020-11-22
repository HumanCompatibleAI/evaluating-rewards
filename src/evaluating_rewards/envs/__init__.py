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

"""Module for Gym environments. __init__ registers environments."""

import gym
import imitation.envs.examples  # noqa: F401  pylint:disable=unused-import
import seals  # noqa: F401  pylint:disable=unused-import

from evaluating_rewards.envs import mujoco, point_mass  # noqa: F401  pylint:disable=unused-import

PROJECT_ROOT = "evaluating_rewards.envs"
PM_ROOT = f"{PROJECT_ROOT}.point_mass"


# Register environments in Gym
def register_point_mass(suffix, **kwargs):
    gym.register(
        id=f"evaluating_rewards/PointMass{suffix}-v0",
        entry_point=f"{PM_ROOT}:PointMassEnv",
        max_episode_steps=100,
        kwargs=kwargs,
    )


register_point_mass("Line", ndim=1)
for i in range(10):
    register_point_mass(f"LineVar0.{i}Start", ndim=1, var=i / 10)
register_point_mass("LineVariableHorizon", ndim=1, threshold=0.05)
register_point_mass("LineStateOnly", ndim=1, ctrl_coef=0.0)
register_point_mass("Grid", ndim=2)


def register_point_maze():
    # Maze true length is 0.1+maze_length-2*0.02, since walls span from -0.1 to maze_length with
    # 0.02 radius. (It's a bit tighter than this up-down, because of the extra dividing wall.)
    # Agent particle radius is 0.03, so 0.01 gives width of 0.1+0.01-2*0.02=0.09 minus 2*0.03
    # equals 0.03 space for the particle to move in -- just enough to vibrate, but not go anywhere.
    # 0.6 is the default length.
    for length in [0.01, 0.05, 0.1, 0.2, 0.3]:
        for dname, dval in {"Left": 0, "Right": 1}.items():
            gym.register(
                id=f"evaluating_rewards/PointMaze{length}{dname}-v0",
                entry_point="imitation.envs.examples.airl_envs.point_maze_env:PointMazeEnv",
                kwargs={"maze_length": length, "direction": dval, "include_vel": True},
            )


register_point_maze()
