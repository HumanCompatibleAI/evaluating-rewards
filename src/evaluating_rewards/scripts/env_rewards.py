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

"""Find hardcoded rewards associated with particular environments.

Outputs a bash-style dictionary suitable for experiment scripts.
"""

from absl import app

# Imported for side-effect of registering custom environments with Gym
from evaluating_rewards import envs  # noqa: F401
from evaluating_rewards.experiments import env_rewards


def main(argv):
    del argv
    print("declare -A REWARDS_BY_ENV=(")
    for env_name, patterns in env_rewards.REWARDS_BY_ENV.items():
        rewards = env_rewards.find_rewards(patterns)
        rewards_joined = " ".join(rewards)
        print(f'    ["{env_name}"]="{rewards_joined}"')
    print(")")


if __name__ == "__main__":
    app.run(main)
