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

"""Deep neural network reward models."""

import abc
import collections
import itertools
import os
import pickle
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
import gym
from imitation.rewards import reward_net
from imitation.util import serialize
import numpy as np
from stable_baselines.common import input as env_in  # avoid name clash
import tensorflow as tf

AffineParameters = collections.namedtuple("AffineParameters",
                                          ["constant", "scale"])
Batch = collections.namedtuple("Batch",
                               ["old_obs", "actions", "new_obs"])


class RewardModel(serialize.Serializable):
  """Abstract reward model."""

  @property
  @abc.abstractmethod
  def reward(self) -> tf.Tensor:
    """Gets the reward output tensor."""

  @property
  @abc.abstractmethod
  def observation_space(self) -> gym.Space:
    """Gets the space of observations."""

  @property
  @abc.abstractmethod
  def action_space(self) -> gym.Space:
    """Gets the space of actions."""

  # TODO(): Avoid using multiple placeholders?
  # When dependencies move to TF 2.x API we can upgrade and this will be
  # unnecessary. Alternatively, we could create placeholders just once
  # and feed them in at construction, but this might require modification
  # to third-party codebases.
  @property
  @abc.abstractmethod
  def old_obs_ph(self) -> Iterable[tf.Tensor]:
    """Gets the old observation placeholder(s)."""

  @property
  @abc.abstractmethod
  def act_ph(self) -> Iterable[tf.Tensor]:
    """Gets the action placeholder(s)."""

  @property
  @abc.abstractmethod
  def new_obs_ph(self) -> Iterable[tf.Tensor]:
    """Gets the new observation placeholder(s)."""


class BasicRewardModel(RewardModel):
  """Abstract reward model class with basic default implementations."""

  def __init__(self, obs_space: gym.Space, act_space: gym.Space):
    RewardModel.__init__(self)
    self._obs_space = obs_space
    self._act_space = act_space
    self._old_obs_ph, self._proc_old_obs = env_in.observation_input(obs_space)
    self._new_obs_ph, self._proc_new_obs = env_in.observation_input(obs_space)
    self._act_ph, self._proc_act = env_in.observation_input(act_space)

  @property
  def observation_space(self):
    return self._obs_space

  @property
  def action_space(self):
    return self._act_space

  @property
  def old_obs_ph(self):
    return (self._old_obs_ph,)

  @property
  def new_obs_ph(self):
    return (self._new_obs_ph,)

  @property
  def act_ph(self):
    return (self._act_ph,)


class MLPRewardModel(BasicRewardModel,
                     serialize.LayersSerializable):
  """Feed-forward reward model r(s,a,s')."""

  def __init__(self, obs_space: gym.Space, act_space: gym.Space,
               hid_sizes: Optional[Iterable[int]] = None, use_act: bool = True,
               use_old_obs: bool = True, use_new_obs: bool = True):
    BasicRewardModel.__init__(self, obs_space, act_space)
    if hid_sizes is None:
      hid_sizes = [32, 32]
    params = locals()

    kwargs = {
        "old_obs_input": self._proc_old_obs if use_old_obs else None,
        "new_obs_input": self._proc_new_obs if use_new_obs else None,
        "act_input": self._proc_act if use_act else None,
    }
    self._reward, layers = reward_net.build_basic_theta_network(hid_sizes,
                                                                **kwargs)
    serialize.LayersSerializable.__init__(**params, layers=layers)

  @property
  def reward(self):
    return self._reward


class PotentialShaping(BasicRewardModel,
                       serialize.LayersSerializable):
  r"""Models a state-only potential, reward is the difference in potential.

  Specifically, contains a state-only potential $$\phi(s)$$. The reward
  $$r(s,a,s') = \gamma \phi(s') - \phi(s)$$ where $$\gamma$$ is the discount.
  """

  def __init__(self, obs_space: gym.Space, act_space: gym.Space,
               hid_sizes: Optional[Iterable[int]] = None,
               discount: float = 0.99, **kwargs):
    BasicRewardModel.__init__(self, obs_space, act_space)

    if hid_sizes is None:
      hid_sizes = [32, 32]
    params = locals()
    del params["kwargs"]
    params.update(**kwargs)

    res = reward_net.build_basic_phi_network(hid_sizes,
                                             self._proc_old_obs,
                                             self._proc_new_obs,
                                             **kwargs)
    self._old_potential, self._new_potential, layers = res
    self.discount = discount
    self._reward_output = discount * self._new_potential - self.old_potential

    serialize.LayersSerializable.__init__(**params, layers=layers)

  @property
  def reward(self):
    return self._reward_output

  @property
  def old_potential(self):
    return self._old_potential

  @property
  def new_potential(self):
    return self._new_potential


class ConstantLayer(tf.keras.layers.Layer):
  """A layer that computes the same output, regardless of input.

  The output is a constant, repeated to the same shape as the input.
  The constant is a trainable variable, and can also be assigned to explicitly.
  """

  def __init__(self,
               name: str = None,
               initializer: Optional[tf.keras.initializers.Initializer] = None,
               dtype: tf.dtypes.DType = tf.float32):
    """Constructs a ConstantLayer.

    Args:
      name: String name of the layer.
      initializer: The initializer to use for the constant weight.
      dtype: dtype of the constant weight.
    """
    if initializer is None:
      initializer = tf.zeros_initializer
    self.initializer = initializer

    super().__init__(trainable=True, name=name, dtype=dtype)

  def build(self, input_shape):
    self._constant = self.add_weight(name="constant",
                                     shape=(),
                                     initializer=self.initializer,
                                     use_resource=True)
    super().build(input_shape)

  def _check_built(self):
    if not self.built:
      raise ValueError("Must call build() before calling this function.")

  @property
  def constant(self):
    self._check_built()
    return self._constant

  def set_constant(self, val):
    self._check_built()
    self.set_weights([np.array(val)])

  def call(self, inputs):
    return inputs * 0 + self.constant

  def get_config(self):
    return {
        "name": self.name,
        "initializer": self.initializer,
        "dtype": self.dtype
    }


class ConstantReward(BasicRewardModel,
                     serialize.LayersSerializable):
  """Outputs a constant reward value. Constant is a (trainable) variable."""

  def __init__(self, obs_space: gym.Space, act_space: gym.Space,
               initializer: Optional[tf.keras.initializers.Initializer] = None):
    BasicRewardModel.__init__(self, obs_space, act_space)
    params = locals()

    self._constant = ConstantLayer(name="constant", initializer=initializer)
    n_batch = tf.shape(self._proc_old_obs)[0]
    obs = tf.reshape(self._proc_old_obs, [n_batch, -1])
    # self._reward_output is a scalar constant repeated (n_batch, ) times
    self._reward_output = self._constant(obs[:, 0])

    serialize.LayersSerializable.__init__(**params,
                                          layers={"constant": self._constant})

  @property
  def constant(self):
    return self._constant

  @property
  def reward(self):
    return self._reward_output


class ZeroReward(BasicRewardModel,
                 serialize.LayersSerializable):
  """A reward model that always outputs zero."""

  def __init__(self, obs_space: gym.Space, act_space: gym.Space):
    serialize.LayersSerializable.__init__(**locals(), layers={})
    BasicRewardModel.__init__(self, obs_space, act_space)

    n_batch = tf.shape(self._proc_old_obs)[0]
    self._reward_output = tf.fill((n_batch,), 0.0)

  @property
  def reward(self):
    return self._reward_output


class RewardNetToRewardModel(RewardModel):
  """Adapts an (imitation repo) RewardNet to our RewardModel type."""

  def __init__(self, network: reward_net.RewardNet, use_test: bool = True):
    """Builds a RewardNet from a RewardModel.

    Args:
      network: A RewardNet.
      use_test: if True, uses `reward_output_test`; otherwise, uses
          `reward_output_train`.
    """
    self.reward_net = network
    self.use_test = use_test

  @property
  def reward(self):
    if self.use_test:
      return self.reward_net.reward_output_test
    else:
      return self.reward_net.reward_output_train

  @property
  def observation_space(self):
    return self.reward_net.observation_space

  @property
  def action_space(self):
    return self.reward_net.action_space

  @property
  def old_obs_ph(self):
    return (self.reward_net.old_obs_ph,)

  @property
  def act_ph(self):
    return (self.reward_net.act_ph,)

  @property
  def new_obs_ph(self):
    return (self.reward_net.new_obs_ph,)

  def _load(self, cls, directory: str) -> "RewardNetToRewardModel":
    with open(os.path.join(directory, "use_test"), "rb") as f:
      use_test = pickle.load(f)

    net = reward_net.RewardNet.load(os.path.join(directory, "net"))
    return cls(net, use_test=use_test)

  def _save(self, directory: str) -> None:
    with open(os.path.join(directory, "use_test"), "rb") as f:
      pickle.dump(self.use_test, f)

    self.reward_net.save(os.path.join(directory, "net"))


class RewardModelWrapper(RewardModel):
  """Wraper for RewardModel objects.

  This wraper is the identity; it is intended to be subclassed.
  """

  def __init__(self, model: RewardModel):
    """Builds a RewardNet from a RewardModel.

    Args:
      model: A RewardNet.
    """
    RewardModel.__init__(self)
    self.model = model

  @property
  def reward(self):
    return self.model.reward

  @property
  def observation_space(self):
    return self.model.observation_space

  @property
  def action_space(self):
    return self.model.action_space

  @property
  def old_obs_ph(self):
    return self.model.old_obs_ph

  @property
  def act_ph(self):
    return self.model.act_ph

  @property
  def new_obs_ph(self):
    return self.model.new_obs_ph

  def _load(self, cls, directory: str) -> "RewardModelWrapper":
    model = RewardModel.load(os.path.join(directory, "model"))
    return cls(model)

  def _save(self, directory: str) -> None:
    self.model.save(os.path.join(directory, "model"))


class StopGradientsModelWrapper(RewardModelWrapper):
  """Stop gradients propagating through a reward model."""

  @property
  def reward(self):
    return tf.stop_gradient(super().reward)


class LinearCombinationModelWrapper(RewardModelWrapper):
  """Builds a linear combination of different reward models."""

  def __init__(self, models: Mapping[str, Tuple[RewardModel, tf.Tensor]]):
    """Constructs a reward model that linearly combines other reward models.

    Args:
      models: A mapping from ids to a tuple of a reward model and weight.
    """
    model = list(models.values())[0][0]
    for m, _ in models.values():
      assert model.action_space == m.action_space
      assert model.observation_space == m.observation_space
    super().__init__(model)
    self._models = models

    weighted = [weight * model.reward for model, weight in models.values()]
    self._reward_output = tf.reduce_sum(weighted, axis=0)

  @property
  def models(self) -> Mapping[str, Tuple[RewardModel, tf.Tensor]]:
    """Models we are linearly combining."""
    return self._models

  @property
  def reward(self):
    return self._reward_output

  @property
  def old_obs_ph(self):
    return tuple(itertools.chain(*[m.old_obs_ph
                                   for m, _ in self.models.values()]))

  @property
  def new_obs_ph(self):
    return tuple(itertools.chain(*[m.new_obs_ph
                                   for m, _ in self.models.values()]))

  @property
  def act_ph(self):
    return tuple(itertools.chain(*[m.act_ph for m, _ in self.models.values()]))

  @classmethod
  def _load(cls, directory: str) -> "LinearCombinationModelWrapper":
    """Restore dehydrated LinearCombinationModelWrapper.

    This should preserve the outputs of the original model, but the model
    itself may differ in two ways. The returned model is always an instance
    of LinearCombinationModelWrapper, and *not* any subclass it may have
    been created by (unless that subclass overrides save and load explicitly).
    Furthermore, the weights are frozen, and so will not be updated with
    training.

    Args:
      directory: The root of the directory to load the model from.

    Returns:
      An instance of LinearCombinationModelWrapper, making identical
      predictions as the saved model.
    """
    with open(os.path.join(directory, "linear_combination.pkl"), "rb") as f:
      loaded = pickle.load(f)

    models = {}
    for model_name, frozen_weight in loaded.items():
      model = RewardModel.load(os.path.join(directory, model_name))
      models[model_name] = (model, tf.constant(frozen_weight))

    return LinearCombinationModelWrapper(models)

  def _save(self, directory) -> None:
    """Save weights and the constituent models.

    WARNING: the weights will be evaluated and their values saved. This
    method makes no attempt to distinguish between constant weights (the common
    case) and variables or other tensors.

    Args:
      directory: The root of the directory to save the model to.
    """
    weights = {}
    for model_name, (model, weight) in self.models.items():
      model.save(os.path.join(directory, model_name))
      weights[model_name] = weight

    sess = tf.get_default_session()
    evaluated_weights = sess.run(weights)

    with open(os.path.join(directory, "linear_combination.pkl"), "wb") as f:
      pickle.dump(evaluated_weights, f)


class AffineTransform(LinearCombinationModelWrapper):
  """Positive affine transformation of a reward model.

  The scale and shift parameter are initialized to be the identity (scale one,
  shift zero).
  """

  def __init__(self,
               wrapped: RewardModel,
               scale: bool = True,
               shift: bool = True):
    """Wraps wrapped, adding a shift and scale parameter if specified.

    Args:
      wrapped: The RewardModel to wrap.
      scale: If true, adds a positive scale parameter.
      shift: If true, adds a shift parameter.
    """
    self.wrapped = wrapped

    if scale:
      self._log_scale = ConstantLayer("log_scale")
      self._log_scale.build(())
      model_scale = tf.exp(self._log_scale.constant)  # force to be non-negative
    else:
      self._log_scale = None
      model_scale = tf.constant(1.0)

    models = {
        "wrapped": (self.wrapped, model_scale),
    }

    if shift:
      self._constant = ConstantReward(wrapped.observation_space,
                                      wrapped.action_space)
      models["constant"] = (self._constant, tf.constant(1.0))
    else:
      self._constant = None

    super().__init__(models)

  def pretrain(self,
               batch: Batch,
               target: RewardModel,
               original: Optional[RewardModel] = None) -> AffineParameters:
    """Initializes the shift and scale parameter to try to match target.

    Computes the mean and standard deviation of the wrapped reward model
    and target on batch, and sets the shift and scale parameters so that the
    output of this model has the same mean and standard deviation as target.

    If the wrapped model is just an affine transformation of target, this
    should get the correct values (given adequate data). However, if they differ
    -- even if just by potential shaping -- it can deviate substantially. It's
    generally still better than just the identity initialization.

    Args:
      batch: Data to evaluate the reward models on.
      target: A RewardModel to match the mean and standard deviation of.
      original: If specified, a RewardModel to rescale to match target.
          Defaults to using the reward model this object wraps, `self.wrapped`.
          This can be undesirable if `self.wrapped` includes some randomly
          initialized model elements, such as potential shaping, that would
          be better to treat as mean-zero.

    Returns:
      The initial shift and scale parameters.
    """
    if original is None:
      original = self.wrapped

    feed_dict = make_feed_dict([original, target], batch)
    sess = tf.get_default_session()
    preds = sess.run([original.reward, target.reward], feed_dict=feed_dict)
    original_mean, target_mean = np.mean(preds, axis=-1)
    original_std, target_std = np.std(preds, axis=-1)

    log_scale = 0.0
    if self._log_scale is not None:
      log_scale = np.log(target_std) - np.log(original_std)
      logging.info("Assigning log scale: %f", log_scale)
      self._log_scale.set_constant(log_scale)
    scale = np.exp(log_scale)

    constant = 0.0
    if self._constant is not None:
      constant = -original_mean * target_std / original_std + target_mean
      logging.info("Assigning shift: %f", constant)
      self._constant.constant.set_constant(constant)

    return AffineParameters(constant=constant, scale=scale)


class PotentialShapingWrapper(LinearCombinationModelWrapper):
  """Adds potential shaping to an underlying reward model."""

  def __init__(self, wrapped: RewardModel, **kwargs):
    """Wraps wrapped with a PotentialShaping instance.

    Args:
      wrapped: The model to add shaping to.
      **kwargs: Passed through to PotentialShaping.
    """
    shaping = PotentialShaping(wrapped.observation_space, wrapped.action_space,
                               **kwargs)

    super().__init__({
        "wrapped": (wrapped, tf.constant(1.0)),
        "shaping": (shaping, tf.constant(1.0)),
    })


def make_feed_dict(models: Iterable[RewardModel],
                   batch: Batch) -> Dict[tf.Tensor, np.ndarray]:
  """Construct a feed dictionary for models for data in batch."""
  assert batch.old_obs.shape == batch.new_obs.shape
  assert batch.old_obs.shape[0] == batch.actions.shape[0]
  a_model = next(iter(models))
  assert batch.old_obs.shape[1:] == a_model.observation_space.shape
  assert batch.actions.shape[1:] == a_model.action_space.shape
  for m in models:
    assert a_model.observation_space == m.observation_space
    assert a_model.action_space == m.action_space

  feed_dict = {}
  for m in models:
    feed_dict.update({ph: batch.old_obs for ph in m.old_obs_ph})
    feed_dict.update({ph: batch.actions for ph in m.act_ph})
    feed_dict.update({ph: batch.new_obs for ph in m.new_obs_ph})

  return feed_dict


def evaluate_models(models: Union[Mapping[str, RewardModel],
                                  Sequence[RewardModel]],
                    batch: Batch) -> np.ndarray:
  """Computes prediction of reward models."""
  if isinstance(models, collections.abc.Mapping):
    reward_outputs = {k: m.reward for k, m in models.items()}
    seq_models = models.values()
  elif isinstance(models, collections.abc.Sequence):
    reward_outputs = [m.reward for m in models]
    seq_models = models
  feed_dict = make_feed_dict(seq_models, batch)
  return tf.get_default_session().run(reward_outputs, feed_dict=feed_dict)


def evaluate_potentials(potentials: Iterable[PotentialShaping],
                        batch: Batch) -> np.ndarray:
  """Computes prediction of potential shaping models."""
  old_pots = [p.old_potential for p in potentials]
  new_pots = [p.new_potential for p in potentials]
  feed_dict = make_feed_dict(potentials, batch)
  return tf.get_default_session().run([old_pots, new_pots], feed_dict=feed_dict)
