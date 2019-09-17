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

"""Utility functions to aid in constructing Sacred experiments."""

import os

# Imported for side-effects (registers with Gym)
from evaluating_rewards import envs  # pylint:disable=unused-import
from imitation.util import util
from sacred import observers


def _get_output_dir():
  return os.path.join(os.getenv('HOME'), 'output')


def make_main(experiment, name):
  """Returns a main function for experiment."""

  experiment.add_config({
      'log_root': os.path.join(_get_output_dir(), name)
  })

  @experiment.config
  def logging(log_root, env_name):
    # pylint: disable=unused-variable
    log_dir = os.path.join(log_root, env_name.replace('/', '_'),
                           util.make_unique_timestamp())
    # pylint: enable=unused-variable

  def main(argv):
    # TODO(): this writes output to disk, which may fail on some VMs
    sacred_dir = os.path.join(_get_output_dir(), 'sacred', name)
    observer = observers.FileStorageObserver.create(sacred_dir)
    experiment.observers.append(observer)
    experiment.run_commandline(argv)

  return main
