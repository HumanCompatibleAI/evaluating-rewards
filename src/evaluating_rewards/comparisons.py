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

"""Methods to compare reward models."""

import collections
import functools
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, TypeVar

import numpy as np
import tensorflow as tf

from evaluating_rewards import datasets, rewards

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

    def __init__(
        self,
        model: rewards.RewardModel,
        target: rewards.RewardModel,
        *,
        loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = tf.losses.mean_squared_error,
        optimizer: Type[tf.train.Optimizer] = tf.train.AdamOptimizer,
        # TODO(): change to optimizer_kwargs?
        learning_rate: float = 1e-2,
    ):
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

        self._opt = optimizer(
            learning_rate=self.learning_rate
        )  # pytype: disable=wrong-keyword-args
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

    def fit(
        self,
        dataset: datasets.BatchCallable,
        total_timesteps: int = int(1e6),
        batch_size: int = 4096,
        log_interval: int = 10,
    ) -> FitStats:
        """Fits shaping to target.

        Args:
            dataset: a callable returning batches of the specified size.
            total_timesteps: the total number of timesteps to train for.
            batch_size: the number of timesteps in each training batch.
            log_interval: reports statistics every log_interval batches.

        Returns:
            Training statistics.
        """
        return fit_models(
            {"singleton": self},
            dataset=dataset,
            total_timesteps=total_timesteps,
            batch_size=batch_size,
            log_interval=log_interval,
        )


ModelWrapperRet = Tuple[rewards.RewardModel, Any, Mapping[str, tf.Tensor]]
ModelWrapperFn = Callable[[rewards.RewardModel], ModelWrapperRet]


class RegressWrappedModel(RegressModel):
    """Wrap a source model and regress the wrapped model onto target.

    Does not change the source model: only the wrapper.
    """

    def __init__(
        self,
        model: rewards.RewardModel,
        target: rewards.RewardModel,
        *,
        model_wrapper: ModelWrapperFn,
        loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = tf.losses.mean_squared_error,
        **kwargs,
    ):
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
        self.metrics["unwrapped_loss"] = loss_fn(self.target.reward, self.unwrapped_source.reward)
        self.metrics.update(metrics)

    def fit_affine(self, batch: rewards.Batch):
        """Fits affine parameters only (not e.g. potential)."""
        affine_model = self.model_extra["affine"]
        return affine_model.fit_lstsq(batch, target=self.target, shaping=None)

    def fit(
        self, dataset: datasets.BatchCallable, affine_size: Optional[int] = 4096, **kwargs,
    ) -> FitStats:
        """Fits shaping to target.

        If `affine_size` is specified, initializes affine parameters using `self.fit_affine`.

        Args:
            dataset: a callable returning batches of the specified size.
            affine_size: the size of the batch to pretrain affine parameters.

        Returns:
            Training statistics.
        """
        if affine_size:
            affine_batch = dataset(affine_size)
            self.fit_affine(affine_batch)
        return super().fit(dataset, **kwargs)


class RegressEquivalentLeastSqModel(RegressWrappedModel):
    """Least-squares regression from source model wrapped with affine and potential shaping.

    Positive affine transformations and potential shaping are optimal policy preserving
    transformations, and so the rewards are considered equivalent (in the sense of Ng et al, 1999).

    The regression is solved via alternating minimization. Since the regression is least-squares,
    the affine parameters can be computed analytically. The potential shaping must be computed
    with gradient descent.

    Does not change the source model: only the wrapper.
    """

    def __init__(self, model: rewards.RewardModel, target: rewards.RewardModel, **kwargs):
        """Constructs RegressEquivalentLeastSqModel.

        Args:
            model: The original model to wrap.
            target: The model we want to match.
            **kwargs: Passed through to super-class.
        """
        model_wrapper = functools.partial(equivalence_model_wrapper, affine_stopgrad=True)
        super().__init__(
            model=model,
            target=target,
            model_wrapper=model_wrapper,
            loss_fn=tf.losses.mean_squared_error,
            **kwargs,
        )

    def fit_affine(self, batch: rewards.Batch) -> rewards.AffineParameters:
        """
        Set affine transformation parameters to analytic least-squares solution.

        Does not update potential parameters.

        Args:
            batch: The batch to compute the affine parameters over.

        Returns:
            The optimal affine parameters (also updates as side-effect).
        """
        affine_model = self.model_extra["affine"]
        shaping_model = self.model_extra["shaping"].models["shaping"][0]
        return affine_model.fit_lstsq(batch, target=self.target, shaping=shaping_model)

    def fit(
        self,
        dataset: datasets.BatchCallable,
        total_timesteps: int = int(1e6),
        epoch_timesteps: int = 16384,
        affine_size: int = 4096,
        **kwargs,
    ) -> FitStats:
        """Fits shaping to target.

        Args:
            dataset: a callable returning batches of the specified size.
            total_timesteps: the total number of timesteps to train for.
            epoch_timesteps: the number of timesteps to train shaping for; the optimal affine
                parameters are set analytically at the start of each epoch.
            affine_size: the size of the batch to pretrain affine parameters.

        Returns:
            Training statistics.

        Raises:
            ValueError if total_timesteps < epoch_timesteps.
        """
        if total_timesteps < epoch_timesteps:
            raise ValueError("total_timesteps must be at least as large as epoch_timesteps.")

        stats = collections.defaultdict(list)
        nepochs = int(total_timesteps) // int(epoch_timesteps)
        for epoch in range(nepochs):
            affine_batch = dataset(affine_size)
            affine_stats = self.fit_affine(affine_batch)
            logging.info(f"Epoch {epoch}: {affine_stats}")

            epoch_stats = super().fit(
                dataset, total_timesteps=epoch_timesteps, affine_size=None, **kwargs
            )

            for k, v in epoch_stats.items():
                stats[k] += v

        return stats


def _scaled_norm(x):
    """l2 norm, normalized to be invariant to length of vectors."""
    return np.linalg.norm(x) / np.sqrt(len(x))


def summary_comparison(
    original: rewards.RewardModel,
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
    models = {"original": original, "matched": matched, "target": target}

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


def equivalence_model_wrapper(
    wrapped: rewards.RewardModel,
    potential: bool = True,
    affine: bool = True,
    affine_stopgrad: bool = False,
    affine_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> ModelWrapperRet:
    """Affine transform model and add potential shaping.

    That is, all transformations that are guaranteed to preserve optimal policy.

    Args:
        wrapped: The model to wrap.
        potential: If true, add potential shaping.
        affine: If true, add affine transformation.
        affine_stopgrad: If true, do not propagate gradients to affine.
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
        metrics["constant"] = model.shift
        metrics["scale"] = model.scale
        if affine_stopgrad:
            model = rewards.StopGradientsModelWrapper(model)

    if potential:
        model = rewards.PotentialShapingWrapper(model, **kwargs)
        models["shaping"] = model

    return model, models, metrics


K = TypeVar("K")


def fit_models(
    potentials: Mapping[K, RegressModel],
    dataset: datasets.BatchCallable,
    total_timesteps: int,
    batch_size: int,
    log_interval: int = 10,
) -> Mapping[str, List[Mapping[K, Any]]]:
    """Regresses model(s).

    Each training step is executed concurrently for all the models, enabling
    TensorFlow to exploit any available concurrency.

    Args:
        potentials: A mapping from strings to a potential-shaped reward model.
        dataset: An iterator returning batches of old obs-act-next obs tuples.
        total_timesteps: the total number of timesteps to train for.
        batch_size: the number of timesteps in each training batch.
        log_interval: The frequency with which to print.

    Returns:
        Metrics from training.

    Raises:
        ValueError if total_timesteps < batch_size.
    """
    if total_timesteps < batch_size:
        raise ValueError("total_timesteps must be at least as larger as batch_size.")

    sess = tf.get_default_session()
    ops = {k: [p.opt_op, p.loss, p.metrics] for k, p in potentials.items()}
    losses = []
    metrics = []

    nbatches = int(total_timesteps) // int(batch_size)
    for i in range(nbatches):
        batch = dataset(batch_size)
        feed_dict = {}
        for potential in potentials.values():
            feed_dict.update(potential.build_feed_dict(batch))

        outputs = sess.run(ops, feed_dict=feed_dict)
        loss = {k: v[1] for k, v in outputs.items()}
        metric = {k: v[2] for k, v in outputs.items()}
        losses.append(loss)
        metrics.append(metric)

        if i % log_interval == 0:
            logging.info(f"{i}: loss = {loss}, " f"metrics = {metric}")

    # TODO(): better logging method, e.g. TensorBoard summaries?
    return {"loss": losses, "metrics": metrics}
