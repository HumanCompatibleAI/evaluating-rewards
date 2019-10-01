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

"""Experiment helper methods to compare two reward models.

See Colab notebook for use cases.
"""

from typing import Any, Dict, Optional
import uuid

from evaluating_rewards import comparisons
from evaluating_rewards import rewards
from evaluating_rewards.experiments import datasets
import numpy as np
import tensorflow as tf


def build_graph(original: rewards.RewardModel,
                target: rewards.RewardModel,
                name: str,
                wrapper=comparisons.equivalence_model_wrapper,
                **kwargs):
  """Build graph to match original to target.

  Arguments:
    original: A reward model.
    target: The reward model we wish to try and make original close to.
    name: A name describing the graph.
    wrapper: A function passed to `comparisons.RegressWrappedModel` to wrap
        `original`. This defines the equivalence class of reward models we
        can search over.
    **kwargs: Passed through to `comparisons.RegressWrappedModel`.

  Returns:
    A tuple (model_match, init) where `model_match` is the RegressWrappedModel
    object and `init` is a TensorFlow operation that initializes the
    variables in `model_match`.
  """
  scope_name = "matching_" + name
  with tf.variable_scope(scope_name):
    match = comparisons.RegressWrappedModel(original,
                                            target,
                                            model_wrapper=wrapper,
                                            learning_rate=1e-2,
                                            **kwargs)
  match_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=scope_name)
  match_vars_init = tf.initializers.variables(match_vars,
                                              name="matching_init_" + name)

  return match, match_vars_init


def fit_match(original: rewards.RewardModel,
              target: rewards.RewardModel,
              dataset: datasets.BatchCallable,
              match: comparisons.RegressWrappedModel,
              match_vars_init: tf.Tensor,
              pretrain: bool = True,
              pretrain_size: int = 4096,
              total_timesteps: int = 2**20,
              batch_size: int = 1024) -> Dict[str, Any]:
  """Fits a comparisons.RegressWrappedModel and evaluates the result.

  Arguments:
    original: The original reward model.
    target: The reward model to make target close to.
    dataset: The dataset on which to fit on.
    match: The training object.
    match_vars_init: An operation to initialize the variables of matching.
    pretrain: If True, initialize the affine scaling parameters based on
        the mean and standard deviation of original and target.
    pretrain_size: The number of samples to use in pretraining.
    total_timesteps: The total number of samples.
    batch_size: The number of samples in each batch.

  Returns:
    A dictionary containing statistics on the training and evaluation.
  """
  # Initialize model parameters
  tf.get_default_session().run(match_vars_init)

  if pretrain:
    pretrain_set = next(dataset(pretrain_size, pretrain_size))
    initial_affine = match.model_extra["affine"].pretrain(pretrain_set,
                                                          target=target,
                                                          original=original)
  else:
    initial_affine = None

  # Train potential shaping and other parameters
  training_metrics = match.fit(dataset(total_timesteps, batch_size))
  final_affine = match.model_extra["affine"].get_weights()

  test_set = next(dataset(4096, 4096))
  stats = comparisons.summary_comparison(original=original,
                                         matched=match.model,
                                         target=target,
                                         test_set=test_set)

  return {
      "affine": {
          "initial": initial_affine,
          "final": final_affine,
      },
      "stats": stats,
      "training_metrics": training_metrics,
  }


def match_pipeline(original: rewards.RewardModel,
                   target: rewards.RewardModel,
                   dataset: datasets.BatchCallable,
                   name: Optional[str] = None,
                   construct_kwargs: Optional[Dict[str, Any]] = None,
                   **kwargs) -> Dict[str, Any]:
  """Compare original to target. Calls `build_graph` and `fit_match`.

  Arguments:
    original: The original reward model.
    target: The reward model to make target close to.
    dataset: The dataset on which to fit on.
    name: The name of the comparison, used in building the graph.
        If not specified, will randomly generate one.
    construct_kwargs: Passed through to `build_graph`.
    **kwargs: Passed through to `fit_match`.

  Returns:
    A dictionary containing the results from `fit_match`, augmented with the
    `deep.RegressWrappedModel` object and name.
  """
  if name is None:
    name = uuid.uuid4().hex

  # TODO(): avoid repeatedly constructing a new subgraph?
  construct_kwargs = construct_kwargs or {}
  match, matching_vars_init = build_graph(original, target,
                                          name, **construct_kwargs)
  match_fitted = fit_match(original, target, dataset,
                           match, matching_vars_init,
                           **kwargs)

  res = dict(match_fitted)
  res.update({
      "match": match,
      "name": name
  })
  return res


def norm_diff(predicted: np.ndarray,
              actual: np.ndarray,
              norm: int):
  r"""Computes the mean $$\ell_p$$ norm of the delta between predicted and actual.

  Normalizes depending on the norm: i.e. for $$\ell_1$$ will divide by the
  number of elements, for $$\ell_2$$ by the square root.

  Arguments:
    predicted: A 1-dimensional array.
    actual: A 1-dimensoinal array.
    norm: The power p in $$\ell_p$$-norm.

  Returns:
    The normalized norm difference between predicted and actual.
  """
  delta = predicted - actual
  if delta.ndim != 1:
    raise TypeError("'predicted' and 'actual' must be 1-dimensional arrays.")
  n = actual.shape[0]
  scale = np.power(n, 1 / norm)
  return np.linalg.norm(delta, ord=norm) / scale


def constant_baseline(match: comparisons.RegressModel,
                      target: rewards.RewardModel,
                      dataset: datasets.BatchCallable,
                      test_size: int = 4096) -> Dict[str, Any]:
  """Computes the error in predictions of the model matched and some baselines.

  Arguments:
    match: The (fitted) match object.
    target: The reward model we are trying to predict.
    dataset: The dataset to evaluate on.
    test_size: The number of samples to evaluate on.

  Returns:
    A dictionary containing summary statistics.
  """
  test_set = next(dataset(test_size, test_size))
  models = {"matched": match.model, "target": target}
  preds = rewards.evaluate_models(models, test_set)

  actual_delta = preds["matched"] - preds["target"]
  return {
      "int_l1": norm_diff(actual_delta, preds["target"], norm=1),
      "int_l2": norm_diff(actual_delta, preds["target"], norm=2),
      "baseline_l1": norm_diff(np.median(preds["target"]),
                               preds["target"], norm=1),
      "baseline_l2": norm_diff(np.mean(preds["target"]),
                               preds["target"], norm=2),
  }
