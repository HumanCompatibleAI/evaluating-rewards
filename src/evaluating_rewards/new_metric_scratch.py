"""Temporary file for implementation of new metric in continuous control environments."""

import logging
from typing import Callable, Type

import numpy as np
import tensorflow as tf

from evaluating_rewards import rewards, tabular

SampleDist = Callable[[int], np.ndarray]


def compute_mean(
    model: rewards.RewardModel,
    obs: np.ndarray,
    obs_dist: SampleDist,
    act_dist: SampleDist,
    n_samples: int = 128,
) -> np.ndarray:
    """Computes the mean reward of `model` starting from `obs`.

    Args:
        model: The reward model to compute the mean of.
        obs: An array of shape (batch_size, ) + obs_shape.
        obs_dist: A function to sample next-observations from.
        act_dist: A function to sample actions from.
        n_samples: The number of samples to take for actions and next-observation.

    Returns:
        An array of shape (batch_size,) containing the mean reward output by `model` over
        `n_samples` transitions `(old_obs,action,new_obs)` where `old_obs` is from `obs` and
        `action` and `new_obs` are sampled from `act_dist` and `obs_dist`.
    """
    # TODO(adam): may be faster to push this computation into TensorFlow instead?
    actions = act_dist(n_samples)
    next_obs = obs_dist(n_samples)

    n_obs = len(obs)
    obs = np.repeat(obs, n_samples, axis=0)
    actions_rep = (n_obs,) + (1,) * (actions.ndim - 1)
    actions = np.tile(actions, actions_rep)
    next_obs_rep = (n_obs,) + (1,) * (next_obs.ndim - 1)
    next_obs = np.tile(next_obs, next_obs_rep)
    assert obs.shape == next_obs.shape
    assert len(obs) == len(actions) == n_samples * n_obs

    batch = rewards.Batch(obs=obs, actions=actions, next_obs=next_obs)
    fd = rewards.make_feed_dict([model], batch)

    sess = tf.get_default_session()
    rew = sess.run(model.reward, feed_dict=fd)
    rew = rew.reshape((n_obs, n_samples))

    return rew.mean(axis=1)


class RegressMeanModel:
    """Regress shaping onto mean of target."""

    def __init__(
        self,
        shaping: rewards.PotentialShaping,
        target: rewards.RewardModel,
        *,
        loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = tf.losses.mean_squared_error,
        optimizer: Type[tf.train.Optimizer] = tf.train.AdamOptimizer,
        # TODO(): change to optimizer_kwargs?
        learning_rate: float = 1e-2,
    ):
        """
        Constructs RegressMeanModel.

        Args:
             shaping: Potential shaping to fit.
             target: The reward model, from which labels are derived using `compute_mean`.
             loss_fn: The loss function to use.
             optimizer: The optimization algorithm to use.
             learning_rate: The learning rate.
        """
        # TODO(adam): significant code duplication with RegressModel
        assert shaping.observation_space == target.observation_space
        assert shaping.action_space == target.action_space

        self.shaping = shaping
        self.target = target
        self.learning_rate = learning_rate

        self.labels_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name="mean_reward_ph")
        self.loss = loss_fn(self.labels_ph, self.shaping._old_potential)

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

    def fit_batch(
        self, batch_size, obs_dist: SampleDist, act_dist: SampleDist, n_target_samples: int = 128,
    ):
        """Fits parameters to one batch of data."""
        obs = obs_dist(batch_size)
        assert self.shaping.observation_space.shape == obs.shape[1:]
        feed_dict = {ph: obs for ph in self.shaping.obs_ph}

        labels = compute_mean(self.target, obs, obs_dist, act_dist, n_target_samples)
        feed_dict[self.labels_ph] = labels
        ops = [self.opt_op, self.loss, self.metrics]

        sess = tf.get_default_session()
        _, loss, metric = sess.run(ops, feed_dict=feed_dict)

        return loss, metric

    def fit(
        self, total_timesteps: int = 1e6, batch_size: int = 128, log_interval: int = 10, **kwargs
    ):
        """Fits parameters of the model."""
        nbatches = int(total_timesteps) // int(batch_size)
        losses = []
        metrics = []
        for i in range(nbatches):
            loss, metric = self.fit_batch(batch_size, **kwargs)
            losses.append(loss)
            metrics.append(metric)

            if i % log_interval == 0:
                logging.info(f"{i}: loss = {loss}, metrics = {metric}")

        # TODO(): better logging method, e.g. TensorBoard summaries?
        return {"loss": losses, "metrics": metrics}


class CanonicalizeRewardWrapper(rewards.LinearCombinationModelWrapper):
    """Canonicalizes the wrapped reward model.

    That is, after fitting this model should be a unique representative of the equivalence class
    formed by adding potential shaping to the wrapped reward model and rescaling by a positive
    constant.

    In practice, the canonical reward will not be recovered exactly. But this model will always be
    in the same equivalence class as the wrapped reward.
    """

    def __init__(self, model: rewards.RewardModel):
        """
        Constructs CanonicalizeRewardWrapper.

        Args:
            model: The model to canonicalize.
        """
        self.model = model
        self.shaping = rewards.PotentialShaping(model.observation_space, model.action_space)
        self.scale = rewards.ConstantLayer("scale", initializer=tf.keras.initializers.Ones())
        self.scale.build(())
        self.shift = rewards.ConstantReward(model.observation_space, model.action_space)
        self.regress_mean = RegressMeanModel(self.shaping, self.model)
        super().__init__(
            {
                "model": (model, self.scale.constant),
                "shaping": (self.shaping, self.scale.constant),
                "shift": (self.shift, self.scale.constant),
            }
        )

    def fit(
        self,
        obs_dist: SampleDist,
        act_dist: SampleDist,
        affine_n_samples: int = 16384,
        p: int = 1,
        **kwargs,
    ):
        """
        Fits the parameters to the canonical representative of the equivalence class.

        Args:
            obs_dist: The distribution over observations.
            act_dist: The distribution over actions.
            affine_n_samples: The number of samples to use to fit affine parameters, that is
                scaling and the constant shift.
            p: The order of the L^p norm.
            kwargs: Passed through to `RegressMeanModel.fit`.

        Returns:
            Statistics from `RegressMeanModel.fit`.
        """
        stats = self.regress_mean.fit(obs_dist=obs_dist, act_dist=act_dist, **kwargs)

        obs_samples = obs_dist(affine_n_samples)
        action_samples = act_dist(affine_n_samples)
        next_obs_samples = obs_dist(affine_n_samples)
        batch = rewards.Batch(obs=obs_samples, actions=action_samples, next_obs=next_obs_samples)

        feed_dict = {ph: obs_samples for ph in self.shaping.obs_ph}
        sess = tf.get_default_session()
        total_mean_samples = sess.run(self.shaping.old_potential, feed_dict=feed_dict)
        total_mean = total_mean_samples.mean()
        self.shift.constant.set_constant(-total_mean)

        self.scale.set_constant(1)  # for evaluation  # TODO(adam): this is hacky
        rew = rewards.evaluate_models({"x": self}, batch)["x"]
        scale = tabular.lp_norm(rew, p=p)
        self.scale.set_constant(1 / scale)

        return stats
