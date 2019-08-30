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

"""Miscellaneous utility methods."""

import contextlib
import copy
from typing import Any, Dict, Iterator, Optional, Tuple, TypeVar

from evaluating_rewards import comparisons
from evaluating_rewards import rewards
import tensorflow as tf


@contextlib.contextmanager
def fresh_sess(intra_op: int = 0,
               inter_op: int = 0,
               log_device_placement: bool = True,
               config_kwargs: Optional[Dict[str, Any]] = None):
  """Context manager setting fresh TensorFlow graph and session."""
  config_kwargs = config_kwargs or {}
  config = tf.ConfigProto(intra_op_parallelism_threads=intra_op,
                          inter_op_parallelism_threads=inter_op,
                          log_device_placement=log_device_placement,
                          **config_kwargs)
  g = tf.Graph()
  with g.as_default():
    sess = tf.Session(graph=g, config=config)
    with sess.as_default():
      yield


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


def get_affine(matching: comparisons.ModelMatch) -> rewards.AffineParameters:
  """Extract affine parameters from a model.

  Arguments:
    matching: The ModelMatch object fitting a model to a target.

  Returns:
    The current affine parameters (scale and shift), from the perspective of
    mapping the *original* onto the *target*; that is, the inverse of the
    transformation actually performed in the model. (This is for ease of
    comparison with results returned by other methods.)
  """
  sess = tf.get_default_session()
  models = matching.model_extra
  scale = 1 / sess.run(models["affine"].models["wrapped"][1])
  const = -sess.run(models["affine"].models["constant"][0].constant) * scale
  return rewards.AffineParameters(constant=const, scale=scale)
