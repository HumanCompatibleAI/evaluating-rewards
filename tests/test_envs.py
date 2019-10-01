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

"""Test evaluating_rewards.envs.*.

Runs simple smoke tests against any environments registered starting with
"evaluating_rewards/".
"""

from absl.testing import absltest
from absl.testing import parameterized
from evaluating_rewards import envs  # pylint:disable=unused-import
from tests import common
import gym
from imitation.testing import envs as test_envs


ENVS = [env_spec.id for env_spec in gym.envs.registration.registry.all()
        if env_spec.id.startswith("evaluating_rewards/")]
DETERMINISTIC_ENVS = [f"evaluating_rewards/{mujoco_env}"
                      for mujoco_env in envs.GYM_MUJOCO_V3_ENVS]


class EnvsTest(parameterized.TestCase):
  """Simple smoke tests for custom environments."""

  @parameterized.parameters(ENVS)
  def test_seed(self, env_name):
    env = common.make_env(env_name)
    test_envs.test_seed(env, env_name, DETERMINISTIC_ENVS)

  @parameterized.parameters(ENVS)
  def test_rollout(self, env_name):
    env = common.make_env(env_name)
    test_envs.test_rollout(env)

  @parameterized.parameters(ENVS)
  def test_model_based(self, env_name):
    """Smoke test for each of the ModelBasedEnv methods with type checks."""
    env = common.make_env(env_name)
    if not hasattr(env, "state_space"):  # pragma: no cover
      self.skipTest("This test is only for subclasses of ModelBasedEnv.")
    test_envs.test_model_based(env)


if __name__ == "__main__":
  absltest.main()
