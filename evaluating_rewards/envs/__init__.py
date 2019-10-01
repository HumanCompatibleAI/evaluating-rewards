# Copyright 2019 DeepMind Technologies Limited
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

"""Module for Gym environments. __init__ registers environments."""

from evaluating_rewards.envs import mujoco
from evaluating_rewards.envs import point_mass
import gym

PROJECT_ROOT = "evaluating_rewards.envs"
PM_ROOT = f"{PROJECT_ROOT}.point_mass"


# Register environments in Gym
def register_point_mass(suffix, **kwargs):
  gym.register(id=f"evaluating_rewards/PointMass{suffix}-v0",
               entry_point=f"{PM_ROOT}:PointMassEnv",
               max_episode_steps=100,
               kwargs=kwargs)

register_point_mass("Line", ndim=1, threshold=0.05)
register_point_mass("LineVariableHorizon", ndim=1, threshold=-1)
register_point_mass("LineStateOnly", ndim=1, ctrl_coef=0.0)
register_point_mass("Grid", ndim=2)


def register_similar(existing_name: str, new_name: str, **kwargs_delta):
  """Registers a gym environment at new_id modifying existing_id by kwargs.

  Args:
    existing_name: The name of an environment registered in Gym.
    new_name: The new name to register with Gym.
    **kwargs_delta: Arguments to override the specification from existing_id.
  """
  existing_spec = gym.spec(existing_name)
  fields = ["entry_point", "reward_threshold", "nondeterministic",
            "tags", "max_episode_steps"]
  kwargs = {k: getattr(existing_spec, k) for k in fields}
  kwargs["kwargs"] = existing_spec._kwargs  # pylint:disable=protected-access
  kwargs.update(**kwargs_delta)
  gym.register(id=new_name, **kwargs)


GYM_MUJOCO_V3_ENVS = ["Ant-v3", "HalfCheetah-v3", "Hopper-v3",
                      "Humanoid-v3", "Swimmer-v3", "Walker2d-v3"]


def register_mujoco():
  kwargs = dict(exclude_current_positions_from_observation=False)
  for env_name in GYM_MUJOCO_V3_ENVS:
    register_similar(existing_name=env_name,
                     new_name=f"evaluating_rewards/{env_name}",
                     kwargs=kwargs)


register_mujoco()
