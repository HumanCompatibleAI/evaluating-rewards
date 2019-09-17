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

"""CLI script to visualize a reward model for PointMassLine*-v0.

Note most environments are too high-dimensional to directly visualize reward.
"""

import os

from absl import app
from evaluating_rewards import serialize
from evaluating_rewards.experiments import point_mass_analysis
from evaluating_rewards.experiments import visualize
from evaluating_rewards.scripts import script_utils
import gym
from imitation import util
import numpy as np
import sacred
from stable_baselines.common import vec_env
import xarray as xr

visualize_pm_reward_ex = sacred.Experiment("visualize_pm_reward")


# pylint:disable=unused-variable
@visualize_pm_reward_ex.config
def default_config():
  """Default configuration values."""
  env_name = "evaluating_rewards/PointMassLineFixedHorizon-v0"
  reward_type = "evaluating_rewards/PointMassSparse-v0"
  reward_path = "dummy"
  density = 11  # number of points along each axis to sample
  lim = 1.0  # points span [-lim, lim]


script_utils.add_logging_config(visualize_pm_reward_ex, "visualize_pm_reward")


@visualize_pm_reward_ex.config
def logging_config(log_root, reward_type, reward_path):
  save_path = os.path.join(log_root,
                           reward_type.replace("/", "_"),
                           reward_path.replace("/", "_"))


@visualize_pm_reward_ex.named_config
def fast():
  """Small config, intended for tests / debugging."""
  density = 5
# pylint:enable=unused-variable


@visualize_pm_reward_ex.main
def visualize_pm_reward(env_name: str,
                        reward_type: str,
                        reward_path: str,
                        density: int,
                        lim: float,
                        save_path: str,
                       ) -> xr.DataArray:
  """Entry-point into script to visualize a reward model for point mass."""
  env = gym.make(env_name)
  venv = vec_env.DummyVecEnv([lambda: env])
  goal = np.array([0.0])

  with util.make_session():
    model = serialize.load_reward(reward_type, reward_path, venv)
    reward = point_mass_analysis.evaluate_reward_model(env, model,
                                                       goal=goal,
                                                       density=density,
                                                       pos_lim=lim,
                                                       vel_lim=lim,
                                                       act_lim=lim)

  fig = point_mass_analysis.plot_reward(reward, goal, zaxis="position")
  visualize.save_fig(f"{save_path}.pdf", fig)

  return reward


if __name__ == "__main__":
  main = script_utils.make_main(visualize_pm_reward_ex, "visualize_pm_reward")
  app.run(main)
