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

import logging
from typing import Any, Callable, List, Mapping, Optional, Tuple, Type, TypeVar

from imitation.data import types
import tensorflow as tf

from evaluating_rewards import datasets
from evaluating_rewards.rewards import base

FitStats = Mapping[str, List[Mapping[str, Any]]]


def ellp_norm_loss(labels: tf.Tensor, predictions: tf.Tensor, p: float = 0.5) -> tf.Tensor:
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
        model: base.RewardModel,
        target: base.RewardModel,
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
        self.target = base.StopGradientsModelWrapper(target)
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

    def build_feed_dict(self, batch: types.Transitions):
        """Construct feed dict given a batch of data."""
        models = [self.model, self.target]
        return base.make_feed_dict(models, batch)

    def fit(
        self,
        dataset: datasets.TransitionsCallable,
        total_timesteps: int = int(1e6),
        batch_size: int = 4096,
        **kwargs,
    ) -> FitStats:
        """Fits shaping to target.

        Args:
            dataset: a callable returning batches of the specified size.
            total_timesteps: the total number of timesteps to train for.
            batch_size: the number of timesteps in each training batch.
            kwargs: passed through to `fit_models`.

        Returns:
            Training statistics.
        """
        return fit_models(
            {"singleton": self},
            dataset=dataset,
            total_timesteps=total_timesteps,
            batch_size=batch_size,
            **kwargs,
        )


ModelWrapperRet = Tuple[base.RewardModel, Any, Mapping[str, tf.Tensor]]
ModelWrapperFn = Callable[[base.RewardModel], ModelWrapperRet]

K = TypeVar("K")


def fit_models(
    potentials: Mapping[K, RegressModel],
    dataset: datasets.TransitionsCallable,
    total_timesteps: int,
    batch_size: int,
    log_interval: int = 10,
    callback: Optional[base.Callback] = None,
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
        callback: If not None, called each epoch with the current epoch number.

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

        if callback:
            callback(i)

    # TODO(): better logging method, e.g. TensorBoard summaries?
    return {"loss": losses, "metrics": metrics}
