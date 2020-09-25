# Copyright 2019, 2020 DeepMind Technologies Limited and Adam Gleave
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

"""Helper methods for computing NPEC distance.

See also `scripts.npec_comparison` and `analysis.dissimilarity_heatmaps.plot_npec_heatmap`.
"""
import collections
import functools
import logging
from typing import Any, Callable, Dict, Optional

from imitation.data import types
import tensorflow as tf

from evaluating_rewards import datasets
from evaluating_rewards.rewards import base, comparisons


class RegressWrappedModel(comparisons.RegressModel):
    """Wrap a source model and regress the wrapped model onto target.

    Does not change the source model: only the wrapper.
    """

    def __init__(
        self,
        model: base.RewardModel,
        target: base.RewardModel,
        *,
        model_wrapper: comparisons.ModelWrapperFn,
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
        self.unwrapped_source = base.StopGradientsModelWrapper(model)
        model, self.model_extra, metrics = model_wrapper(self.unwrapped_source)
        super().__init__(model=model, target=target, loss_fn=loss_fn, **kwargs)
        self.metrics["unwrapped_loss"] = loss_fn(self.target.reward, self.unwrapped_source.reward)
        self.metrics.update(metrics)

    def fit_affine(self, batch: types.Transitions):
        """Fits affine parameters only (not e.g. potential)."""
        affine_model = self.model_extra["affine"]
        return affine_model.fit_lstsq(batch, target=self.target, shaping=None)

    def fit(
        self,
        dataset: datasets.TransitionsCallable,
        affine_size: Optional[int] = 4096,
        **kwargs,
    ) -> comparisons.FitStats:
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

    def __init__(self, model: base.RewardModel, target: base.RewardModel, **kwargs):
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

    def fit_affine(self, batch: types.Transitions) -> base.AffineParameters:
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
        dataset: datasets.TransitionsCallable,
        total_timesteps: int = int(1e6),
        epoch_timesteps: int = 16384,
        affine_size: int = 4096,
        **kwargs,
    ) -> comparisons.FitStats:
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


def equivalence_model_wrapper(
    wrapped: base.RewardModel,
    potential: bool = True,
    affine: bool = True,
    affine_stopgrad: bool = False,
    affine_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> comparisons.ModelWrapperRet:
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
        model = base.AffineTransform(model, **affine_kwargs)
        models["affine"] = model
        metrics["constant"] = model.shift
        metrics["scale"] = model.scale
        if affine_stopgrad:
            model = base.StopGradientsModelWrapper(model)

    if potential:
        model = base.MLPPotentialShapingWrapper(model, **kwargs)
        models["shaping"] = model

    return model, models, metrics
