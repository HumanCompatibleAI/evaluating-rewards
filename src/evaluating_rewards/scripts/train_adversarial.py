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

"""Thin wrapper around imitation.scripts.train_adversarial."""

from absl import app
from imitation.scripts import train_adversarial

from evaluating_rewards.scripts import script_utils

if __name__ == "__main__":
  script_utils.add_logging_config(train_adversarial.train_ex,
                                  "train_adversarial")
  main = script_utils.make_main(train_adversarial.train_ex, "train_adversarial")
  app.run(main)
