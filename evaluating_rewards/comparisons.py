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

"""Methods to compare reward models."""
import logging
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Tuple, Type, TypeVar

from evaluating_rewards import rewards
import numpy as np
import pandas as pd
import tensorflow as tf


class ModelMatch(object):
  """Builds a model wrapping a source model and fits it to match target."""

  def __init__(self,
               source: rewards.RewardModel,
               target: rewards.RewardModel,
               model_wrapper: Callable[[rewards.RewardModel],
                                       Tuple[rewards.RewardModel, Any]],
               loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] =
               tf.losses.mean_squared_error,
               optimizer: Type[tf.train.Optimizer] = tf.train.AdamOptimizer,
               learning_rate: float = 1e-2):
    """Constructs ModelMatch.

    Args:
      source: The original model.
      target: The model we want to match.
      model_wrapper: A wrapper applied to source. This wrapper will be fit
          to target. Typically the wrapper is constrained to not change the
          equivalence class of source.
      loss_fn: A function computing the loss from labels and predictions.
      optimizer: The type of optimizer to use.
      learning_rate: Hyperparameter for optimizer.
    """
    assert source.observation_space == target.observation_space
    assert source.action_space == target.action_space
    self.source = rewards.StopGradientsModelWrapper(source)
    self.target = rewards.StopGradientsModelWrapper(target)
    self.learning_rate = learning_rate

    self.model, self.model_extra = model_wrapper(self.source)

    self.loss = loss_fn(self.target.reward, self.model.reward)
    self.unshaped_mse = loss_fn(self.target.reward, self.source.reward)

    self._opt = optimizer(learning_rate=self.learning_rate)  # pytype: disable=wrong-keyword-args
    self._grads = self._opt.compute_gradients(self.loss)
    self.grad_norm = {variable: tf.norm(gradient)
                      for variable, gradient in self._grads
                      if gradient is not None}
    self.opt_op = self._opt.apply_gradients(self._grads)

  def build_feed_dict(self, batch: rewards.Batch):
    """Construct feed dict given a batch of data."""
    models = [self.source, self.target, self.model]
    return rewards.make_feed_dict(models, batch)

  def fit(self, dataset: Iterator[rewards.Batch],
          log_interval: int = 10) -> Mapping[str, pd.DataFrame]:
    """Fits shaping to target."""
    return potential_fit({"singleton": self},
                         dataset=dataset,
                         log_interval=log_interval)


def _scaled_norm(x):
  """l2 norm, normalized to be invariant to length of vectors."""
  return np.linalg.norm(x) / np.sqrt(len(x))


def summary_comparison(original: rewards.RewardModel,
                       matched: rewards.RewardModel,
                       target: rewards.RewardModel,
                       test_set: rewards.Batch,
                       shaping: Optional[rewards.RewardModel] = None,
                      ) -> Tuple[float, float, float]:
  """Compare rewards in terms of intrinsic and shaping difference.

  Args:
    original: The inferred reward model.
    matched: The reward model after trying to match target via shaping.
    target: The target reward model (e.g. ground truth, if available).
    test_set: A dataset to evaluate on.
    shaping: A reward model adding potential shaping to original.
        If unspecified, will return 0 for the shaping component.

  Returns:
    A tuple (intrinsic, shaping, extrinsic). The intrinsic difference is the
    approximation of the nearest point between the equivalence classes for
    original and target. Shaping is the magnitude of the potential shaping
    term we are adding. Extrinsic is the raw difference between original and
    target without any transformations.
  """
  models = {
      "original": original,
      "matched": matched,
      "target": target,
  }

  if shaping is not None:
    models["shaping"] = shaping

  preds = rewards.evaluate_models(models, test_set)
  intrinsic_l2 = _scaled_norm(preds["matched"] - preds["target"])
  if "shaping" in preds:
    shaping_l2 = _scaled_norm(preds["shaping"])
  else:
    shaping_l2 = 0.0
  extrinsic_l2 = _scaled_norm(preds["original"] - preds["target"])

  return intrinsic_l2, shaping_l2, extrinsic_l2


def equivalence_model_wrapper(wrapped: rewards.RewardModel,
                              potential: bool = True,
                              affine: bool = True,
                              affine_kwargs: Optional[Dict[str, Any]] = None,
                              **kwargs,
                             ) -> Tuple[rewards.RewardModel,
                                        Mapping[str,
                                                rewards.RewardModel]]:
  """Affine transform model and add potential shaping.

  That is, all transformations that are guaranteed to preserve optimal policy.

  Args:
    wrapped: The model to wrap.
    potential: If true, add potential shaping.
    affine: If true, add affine transformation.
    affine_kwargs: Passed through to AffineTransform.
    **kwargs: Passed through to PotentialShapingWrapper.

  Returns:
    A transformed version of wrapped.
  """

  model = wrapped
  models = {"original": wrapped}

  if affine:
    affine_kwargs = affine_kwargs or {}
    model = rewards.AffineTransform(model, **affine_kwargs)
    models["affine"] = model

  if potential:
    model = rewards.PotentialShapingWrapper(model, **kwargs)
    models["shaping"] = model

  return model, models


K = TypeVar("K")


def potential_fit(potentials: Mapping[K, ModelMatch],
                  dataset: Iterator[rewards.Batch],
                  log_interval: int = 10) -> Mapping[K, pd.DataFrame]:
  """Fits potentials to dataset.

  Args:
    potentials: A mapping from strings to a potential-shaped reward model.
    dataset: An iterator returning batches of old obs-act-next obs tuples.
    log_interval: The frequency with which to print.

  Returns:
    Metrics from training.
  """
  sess = tf.get_default_session()
  ops = {k: [p.opt_op, p.grad_norm, p.loss, p.unshaped_mse]
         for k, p in potentials.items()}
  grad_norms = []
  losses = []
  unshaped_mses = []
  for i, batch in enumerate(dataset):
    feed_dict = {}
    for potential in potentials.values():
      feed_dict.update(potential.build_feed_dict(batch))

    outputs = sess.run(ops, feed_dict=feed_dict)
    grad_norm = {k: v[1] for k, v in outputs.items()}
    loss = {k: v[2] for k, v in outputs.items()}
    unshaped_mse = {k: v[3] for k, v in outputs.items()}
    grad_norms.append(grad_norm)
    losses.append(loss)
    unshaped_mses.append(unshaped_mse)

    if i % log_interval == 0:
      logging.info(f"{i}: grad norm = {grad_norm}, shaped MSE = {loss}, "
                   f"unshaped MSE = {unshaped_mse}")

  # TODO(): better logging method, e.g. TensorBoard summaries?
  return {
      "grad_norm": grad_norms,
      "loss": pd.DataFrame(losses),
      "unshaped_mse": pd.DataFrame(unshaped_mses),
  }
