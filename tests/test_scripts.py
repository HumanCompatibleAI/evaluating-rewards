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

"""Smoke tests for CLI scripts."""

import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from evaluating_rewards.scripts.model_comparison import model_comparison_ex
from evaluating_rewards.scripts.train_preferences import train_preferences_ex
from evaluating_rewards.scripts.train_regress import train_regress_ex
from tests import common
import pandas as pd


EXPERIMENTS = {
    "comparison": {
        "experiment": model_comparison_ex,
        "expected_type": dict,
    },
    "regress": {
        "experiment": train_regress_ex,
        "expected_type": dict,
    },
    "preferences": {
        "experiment": train_preferences_ex,
        "expected_type": pd.DataFrame,
    },
}


class ScriptTest(parameterized.TestCase):
  """Smoke tests for CLI scripts."""

  @parameterized.named_parameters(common.combine_dicts_as_kwargs(EXPERIMENTS))
  def test_experiment(self, experiment, expected_type):
    with tempfile.TemporaryDirectory(prefix="eval-rewards-exp") as tmpdir:
      run = experiment.run(
          named_configs=["fast"],
          config_updates=dict(log_root=tmpdir),
      )
    assert run.status == "COMPLETED"
    assert isinstance(run.result, expected_type)


if __name__ == "__main__":
  absltest.main()
