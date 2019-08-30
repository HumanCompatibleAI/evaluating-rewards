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

from evaluating_rewards.serialize import reward_registry
import gym
from imitation.policies.serialize import policy_registry

PROJECT_ROOT = "evaluating_rewards.envs"
PM_ROOT = f"{PROJECT_ROOT}.point_mass"


# Register environments in Gym
def register_point_mass(suffix, **kwargs):
  gym.register(id=f"evaluating_rewards/PointMass{suffix}-v0",
               entry_point=f"{PM_ROOT}:PointMassEnv",
               max_episode_steps=100,
               kwargs=kwargs)

register_point_mass("Line", ndim=1)
register_point_mass("LineFixedHorizon", ndim=1, threshold=-1)
register_point_mass("LineFixedHorizonStateOnly", ndim=1,
                    threshold=-1, ctrl_coef=0.0)
register_point_mass("Grid", ndim=2)

# Register custom policies with imitation
policy_registry.register(key="evaluating_rewards/PointMassHardcoded-v0",
                         indirect=f"{PM_ROOT}:load_point_mass_policy")

# Register custom rewards with evaluating_rewards
reward_registry.register(key="evaluating_rewards/PointMassGroundTruth-v0",
                         indirect=f"{PM_ROOT}:load_point_mass_ground_truth")
reward_registry.register(key="evaluating_rewards/PointMassSparseReward-v0",
                         indirect=f"{PM_ROOT}:load_point_mass_sparse_reward")
reward_registry.register(
    key="evaluating_rewards/PointMassSparseRewardNoCtrl-v0",
    indirect=f"{PM_ROOT}:load_point_mass_sparse_reward_no_ctrl")
reward_registry.register(key="evaluating_rewards/PointMassDenseReward-v0",
                         indirect=f"{PM_ROOT}:load_point_mass_dense_reward")
reward_registry.register(
    key="evaluating_rewards/PointMassDenseRewardNoCtrl-v0",
    indirect=f"{PM_ROOT}:load_point_mass_dense_reward_no_ctrl")
