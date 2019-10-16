# Copyright 2019 DeepMind Technologies Limited and Adam Gleave
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

import gym
from imitation.testing import envs as test_envs
import pytest

from evaluating_rewards import envs  # noqa: F401 pylint:disable=unused-import

ENV_NAMES = [
    env_spec.id
    for env_spec in gym.envs.registration.registry.all()
    if env_spec.id.startswith("evaluating_rewards/")
]
DETERMINISTIC_ENVS = []


@pytest.mark.parametrize("env_name", ENV_NAMES)
# pylint:disable=no-self-use
class TestEnvs:
    """Simple smoke tests for custom environments."""

    def test_seed(self, env, env_name):
        test_envs.test_seed(env, env_name, DETERMINISTIC_ENVS)

    def test_rollout(self, env):
        test_envs.test_rollout(env)

    def test_model_based(self, env):
        """Smoke test for each of the ModelBasedEnv methods with type checks."""
        if not hasattr(env, "state_space"):  # pragma: no cover
            pytest.skip("This test is only for subclasses of ModelBasedEnv.")
        test_envs.test_model_based(env)
# pylint:enable=no-self-use
