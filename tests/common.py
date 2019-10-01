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

"""Common functionality for tests."""
import copy
from typing import Dict, Iterator, Tuple, TypeVar

from absl.testing import absltest
from absl.testing import parameterized
import gym
import tensorflow as tf


class TensorFlowTestCase(parameterized.TestCase):
  """Base class for parameterized tests involving TensorFlow."""

  def setUp(self):
    super().setUp()
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)

  def tearDown(self):
    super().tearDown()
    self.sess.close()
    self.graph = None


K = TypeVar("K")
V = TypeVar("V")


def combine_dicts(*dicts: Dict[str, Dict[K, V]],
                 ) -> Iterator[Tuple[str, Dict[K, V]]]:
  """Return a generator merging together the dictionary arguments.

  Usage example:

    > list(combine_dicts({'a': {'x': 1}, 'b': {'x': 2}},
                         {'c': {'y': 2}, 'd': {'x': 4, 'z': 3}}))
    [('a_c', {'x': 1, 'y': 2}),
     ('a_d', {'x': 4, 'z': 3}),
     ('b_c', {'x': 2, 'y': 2}),
     ('b_d', {'x': 4, 'z': 3})]


  Arguments:
    *dicts: A list of dictionaries, mapping from strings to inner dictionaries.

  Yields:
    Pairs (key, merged) computed from the cartesian product over the inner
    dictionaries. Specifically, if *dicts is an n-length list of dictionaries,
    then for each n-wise combination of inner dictionaries it yields
    (key, merged) where key is the keys of each of the inner dictionaries
    joined with "_" and merged is a dictionary containing key-value pairs from
    each of the inner dictionaries. (If a key occurs multiple times, the
    right-most occurrence takes precedence.)
  """
  head, *tail = dicts
  if not tail:
    for name, cfg in head.items():
      yield name, cfg
  else:
    for head_name, head_cfg in head.items():
      for tail_name, tail_cfg in combine_dicts(*tail):
        name = head_name + "_" + tail_name
        cfg = copy.deepcopy(head_cfg)
        cfg.update(tail_cfg)
        yield name, cfg


def combine_dicts_as_kwargs(*dicts):
  for name, cfg in combine_dicts(*dicts):
    yield {
        "testcase_name": name,
        **cfg
    }


def make_env(env_name: str) -> gym.Env:
  """Wrapper on gym.make, skipping test if simulator not installed."""
  try:
    env = gym.make(env_name)
  except gym.error.DependencyNotInstalled as e:  # pragma: no cover
    if e.args[0].find("mujoco_py") != -1:
      raise absltest.SkipTest("Requires `mujoco_py`, which isn't installed.")
    else:
      raise
  return env
