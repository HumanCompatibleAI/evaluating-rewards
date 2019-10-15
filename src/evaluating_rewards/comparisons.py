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
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Type, TypeVar

from evaluating_rewards import rewards
import numpy as np
import tensorflow as tf


FitStats = Mapping[str, List[Mapping[str, Any]]]


def ellp_norm_loss(labels, predictions, p=0.5):
  """Loss based on L^p norm between `labels` and `predictions`."""
  delta = labels - predictions
  delta = tf.abs(delta)
  delta_pow = tf.pow(delta, p)
  mean_delta_pow = tf.reduce_mean(delta_pow)
  return tf.pow(mean_delta_pow, 1 / p)


class RegressModel:
  """Regress source model onto target."""

  def __init__(self,
               model: rewards.RewardModel,
               target: rewards.RewardModel,
               *,
               loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] =
               tf.losses.mean_squared_error,
               optimizer: Type[tf.train.Optimizer] = tf.train.AdamOptimizer,
               # TODO(): change to optimizer_kwargs?
               learning_rate: float = 1e-2):
    """Constructs RegressModel.

    Args:
      model: The model to fit.
      target: The target we want to match.
      loss_fn: A function computing the loss from labels and predictions.
      optimizer: The type of optimizer to use.
      learning_rate: Hyperparameter for optimizer.
    """
    assert model.observation_space == target.observation_space
    assert model.action_space == target.action_space

    self.model = model
    self.target = rewards.StopGradientsModelWrapper(target)
    self.learning_rate = learning_rate

    self.loss = loss_fn(self.target.reward, self.model.reward)

    self._opt = optimizer(learning_rate=self.learning_rate)  # pytype: disable=wrong-keyword-args
    self._grads = self._opt.compute_gradients(self.loss)
    self.opt_op = self._opt.apply_gradients(self._grads)

    self.metrics = {}
    self.metrics["grad_norm"] = {
        variable.name: tf.norm(gradient)
        for gradient, variable in self._grads
        if gradient is not None
    }

  def build_feed_dict(self, batch: rewards.Batch):
    """Construct feed dict given a batch of data."""
    models = [self.model, self.target]
    return rewards.make_feed_dict(models, batch)

  def fit(self, dataset: Iterator[rewards.Batch],
          log_interval: int = 10) -> FitStats:
    """Fits shaping to target.

    Args:
      dataset: iterator of batches of data to fit to.
      log_interval: reports statistics every log_interval batches.

    Returns:
      Training statistics.
    """
    return fit_models({"singleton": self},
                      dataset=dataset,
                      log_interval=log_interval)


ModelWrapperRet = Tuple[rewards.RewardModel,
                        Any,
                        Mapping[str, tf.Tensor]]
ModelWrapperFn = Callable[[rewards.RewardModel], ModelWrapperRet]


class RegressWrappedModel(RegressModel):
  """Wrap a source model and regress the wrapped model onto target.

  Does not change the source model: only the wrapper.
  """

  def __init__(self,
               model: rewards.RewardModel,
               target: rewards.RewardModel,
               *,
               model_wrapper: ModelWrapperFn,
               loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] =
               tf.losses.mean_squared_error,
               **kwargs):
    """Constructs RegressWrappedModel.

    Args:
      model: The original model.
      target: The model we want to match.
      model_wrapper: A wrapper applied to source. This wrapper will be fit
          to target. Typically the wrapper is constrained to not change the
          equivalence class of source.
      loss_fn: A function computing the loss from labels and predictions.
      **kwargs: Passed through to super-class.
    """
    self.unwrapped_source = rewards.StopGradientsModelWrapper(model)
    model, self.model_extra, metrics = model_wrapper(self.unwrapped_source)
    super().__init__(model=model, target=target, loss_fn=loss_fn, **kwargs)
    self.metrics["unwrapped_loss"] = loss_fn(self.target.reward,
                                             self.unwrapped_source.reward)
    self.metrics.update(metrics)

  def pretrain(self, batch: rewards.Batch):
    affine_model = self.model_extra["affine"]
    return affine_model.pretrain(batch,
                                 target=self.target,
                                 original=self.unwrapped_source)

  def fit(self,
          dataset: Iterator[rewards.Batch],
          pretrain: Optional[rewards.Batch],
          **kwargs) -> FitStats:
    """Fits shaping to target.

    Args:
      dataset: iterator of batches of data to fit to.
      pretrain: if provided, warm-start affine parameters from estimates
          computed from this batch. (Requires that model_wrapper adds
          affine parameters.)
      **kwargs: passed through to super().fit.

    Returns:
      Training statistics.
    """
    if pretrain:
      self.pretrain(pretrain)
    return super().fit(dataset, **kwargs)


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
                             ) -> ModelWrapperRet:
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
  metrics = {}

  if affine:
    affine_kwargs = affine_kwargs or {}
    model = rewards.AffineTransform(model, **affine_kwargs)
    models["affine"] = model
    metrics["constant"] = model.constant
    metrics["scale"] = model.scale

  if potential:
    model = rewards.PotentialShapingWrapper(model, **kwargs)
    models["shaping"] = model

  return model, models, metrics


K = TypeVar("K")


def fit_models(potentials: Mapping[K, RegressModel],
               dataset: Iterator[rewards.Batch],
               log_interval: int = 10) -> Mapping[str, List[Mapping[K, Any]]]:
  """Regresses model(s).

  Each training step is executed concurrently for all the models, enabling
  TensorFlow to exploit any available concurrency.

  Args:
    potentials: A mapping from strings to a potential-shaped reward model.
    dataset: An iterator returning batches of old obs-act-next obs tuples.
    log_interval: The frequency with which to print.

  Returns:
    Metrics from training.
  """
  sess = tf.get_default_session()
  ops = {k: [p.opt_op, p.loss, p.metrics]
         for k, p in potentials.items()}
  losses = []
  metrics = []
  for i, batch in enumerate(dataset):
    feed_dict = {}
    for potential in potentials.values():
      feed_dict.update(potential.build_feed_dict(batch))

    outputs = sess.run(ops, feed_dict=feed_dict)
    loss = {k: v[1] for k, v in outputs.items()}
    metric = {k: v[2] for k, v in outputs.items()}
    losses.append(loss)
    metrics.append(metric)

    if i % log_interval == 0:
      logging.info(f"{i}: loss = {loss}, "
                   f"metrics = {metric}")

  # TODO(): better logging method, e.g. TensorBoard summaries?
  return {
      "loss": losses,
      "metrics": metrics,
  }
