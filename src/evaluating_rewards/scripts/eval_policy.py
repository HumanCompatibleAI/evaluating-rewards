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

"""Thin wrapper around imitation.scripts.eval_policy."""

from imitation.scripts import eval_policy

# Imported for side-effect (registers policies we may want to use)
from evaluating_rewards import policies  # pylint:disable=unused-import # noqa: F401
from evaluating_rewards.scripts import script_utils

if __name__ == "__main__":
    script_utils.add_logging_config(eval_policy.eval_policy_ex, "eval_policy")
    script_utils.experiment_main(eval_policy.eval_policy_ex, "eval_policy")
