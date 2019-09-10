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
from typing import Any, Dict, Optional

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


def get_affine(matching: comparisons.RegressWrappedModel,
              ) -> rewards.AffineParameters:
  """Extract affine parameters from a model.

  Arguments:
    matching: The RegressWrappedModel object fitting a model to a target.

  Returns:
    The current affine parameters (scale and shift), from the perspective of
    mapping the *original* onto the *target*; that is, the inverse of the
    transformation actually performed in the model. (This is for ease of
    comparison with results returned by other methods.)
  """
  sess = tf.get_default_session()
  models = matching.model_extra
  scale_tensor = models["affine"].models["wrapped"][1]
  # .constant is a ConstantLayer, .constant.constant is the tensor
  const_tensor = models["affine"].models["constant"][0].constant.constant
  scale, const = sess.run([scale_tensor, const_tensor])
  return rewards.AffineParameters(constant=const, scale=scale)
