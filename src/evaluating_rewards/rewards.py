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

"""Deep neural network reward models."""

import abc
import itertools
import os
import pickle
from typing import Dict, Iterable, Mapping, NamedTuple, Optional, Sequence, Tuple, Type, TypeVar

import gym
from imitation.data import rollout, types
from imitation.rewards import reward_net
from imitation.util import networks, serialize
import numpy as np
import scipy.optimize
from stable_baselines.common import input as env_in  # avoid name clash
import tensorflow as tf

K = TypeVar("K")


class AffineParameters(NamedTuple):
    """Parameters of an affine transformation.

    Attributes:
        shift: The additive shift.
        scale: The multiplicative dilation factor.
    """

    shift: float
    scale: float


# abc.ABC inheritance is strictly unnecessary as serialize.Serializable is
# abstract, but pytype gets confused without this.
class RewardModel(serialize.Serializable, abc.ABC):
    """Abstract reward model."""

    @property
    @abc.abstractmethod
    def reward(self) -> tf.Tensor:
        """Gets the reward output tensor."""

    @property
    def discount(self) -> Optional[tf.Tensor]:
        """Tensor specifying discount rate of reward model, or None if not applicable.

        Generally reward models that involve explicit shaping will have a discount parameter
        while others will not.
        """
        return None

    def set_discount(self, discount: float) -> None:
        """Set the discount rate; no-op in models without internal discounting."""

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
    def obs_ph(self) -> Iterable[tf.Tensor]:
        """Gets the current observation placeholder(s)."""

    @property
    @abc.abstractmethod
    def act_ph(self) -> Iterable[tf.Tensor]:
        """Gets the action placeholder(s)."""

    @property
    @abc.abstractmethod
    def next_obs_ph(self) -> Iterable[tf.Tensor]:
        """Gets the next observation placeholder(s)."""

    @property
    @abc.abstractmethod
    def dones_ph(self) -> Iterable[tf.Tensor]:
        """Gets the terminal state placeholder(s)."""


# pylint:disable=abstract-method
# These classes are abstract but PyLint don't understand them; see pylint GH #179
# TODO(adam): in pylint 2.5.x adding abc.ABC as an inheritance should fix this
class BasicRewardModel(RewardModel):
    """Abstract reward model class with basic default implementations."""

    def __init__(self, obs_space: gym.Space, act_space: gym.Space):
        """Builds BasicRewardModel: adding placeholders and spaces but nothing else.

        The spaces passed are used to define the `observation_space` and `action_space`
        properties, and also are used to determine how to preprocess the observation
        and action placeholders, made available as `self._proc_{obs,act,next_obs}`.

        Args:
            obs_space: The observation space.
            act_space: The action space.
        """
        RewardModel.__init__(self)
        self._obs_space = obs_space
        self._act_space = act_space
        self._obs_ph, self._proc_obs = env_in.observation_input(obs_space)
        self._next_obs_ph, self._proc_next_obs = env_in.observation_input(obs_space)
        self._act_ph, self._proc_act = env_in.observation_input(act_space)
        self._dones_ph = tf.placeholder(name="dones", shape=(None,), dtype=tf.bool)
        self._proc_dones = tf.cast(self._dones_ph, dtype=tf.float32)

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._act_space

    @property
    def obs_ph(self):
        return (self._obs_ph,)

    @property
    def next_obs_ph(self):
        return (self._next_obs_ph,)

    @property
    def act_ph(self):
        return (self._act_ph,)

    @property
    def dones_ph(self):
        return (self._dones_ph,)


class PotentialShaping(RewardModel):
    """Mix-in to add potential shaping.

    We follow Ng et al (1999)'s definition of potential shaping. In the discounted
    infinite-horizon case, this is simple. Define a state-only function `pot(s)`,
    and then the reward output is `discount * pot(s') - pot(s)` where `s'` is the
    next state and `s` the current state.

    In finite-horizon cases, however, we must always transition into a special
    absorbing state when the episode terminates. This ensures that you pay back
    the potential gained earlier in the episode -- otherwise potential shaping
    would change the optimal policy. We handle this by introducing a special
    end_potential Tensor, which should *not* depend on the state (so may be a
    trainable scalar variable or constant).

    In the undiscounted finite-horizon case, a constant shift in potential has
    no effect on the shaping output. We follow Ng et al in assuming WLOG that the
    potential is zero at the terminal state. Ng et al need this for their derivation,
    but it isn't needed computationally. However, removing an unnecessary degree of
    freedom does make the learning problem better conditioned.

    Andrew Y. Ng, Daishi Harada & Stuart Russell (1999). ICML.
    """

    def __init__(
        self,
        old_potential: tf.Tensor,
        new_potential: tf.Tensor,
        end_potential: tf.Tensor,
        dones: tf.Tensor,
        discount: float = 0.99,
    ):
        """
        Builds PotentialShaping mix-in, adding reward in terms of {old,new}_potential.

        Args:
            old_potential: The potential of the observation.
            new_potential: The potential of the next observation.
            end_potential: The potential of a terminal state at the end of an episode.
                If discount is 1.0, this is ignored and it is fixed at 0.0 instead,
                following Ng et al (1999).
            dones: Indicator variable (0 or 1 floating point tensor) for episode termination.
            discount: The initial discount rate to use.

        Raises:
            ValueError if self.dones_ph is empty.
        """
        self._discount = ConstantLayer("discount", initializer=tf.constant_initializer(discount))
        self._discount.build(())

        self._old_potential = old_potential
        is_discounted = tf.cast(self.discount == 1.0, dtype=tf.float32)
        end_potential = is_discounted * end_potential
        self._new_potential = end_potential * dones + new_potential * (1 - dones)
        self._reward_output = self.discount * self.new_potential - self.old_potential

    @property
    def reward(self):
        """The reward: discount * new_potential - old_potential."""
        return self._reward_output

    @property
    def discount(self) -> tf.Tensor:
        return tf.stop_gradient(self._discount.constant)

    def set_discount(self, discount: float) -> None:
        self._discount.set_constant(discount)

    @property
    def old_potential(self) -> tf.Tensor:
        """The potential of the observation."""
        return self._old_potential

    @property
    def new_potential(self) -> tf.Tensor:
        """The potential of the next observation.

        This is fixed to zero when dones_ph is True.
        """
        return self._new_potential


# pylint:enable=abstract-method


class MLPRewardModel(BasicRewardModel, serialize.LayersSerializable):
    """Feed-forward reward model r(s,a,s')."""

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        hid_sizes: Iterable[int] = (32, 32),
        *,
        use_obs: bool = True,
        use_act: bool = True,
        use_next_obs: bool = True,
        use_dones: bool = True,
    ):
        """Builds MLPRewardModel.

        Args:
            obs_space: The observation space.
            act_space: The action space.
            hid_sizes: The number of hidden units at each layer in the network.
            use_obs: Whether to include the observation in the input.
            use_act: Whether to include actions in the input.
            use_next_obs: Whether to include the next observation in the input.
            use_dones: Whether to include episode termination in the input.

        Raises:
            ValueError if none of `use_obs`, `use_act` or `use_next_obs` are True.
        """
        BasicRewardModel.__init__(self, obs_space, act_space)
        params = dict(locals())

        inputs = []
        if use_obs:
            inputs.append(self._proc_obs)
        if use_act:
            inputs.append(self._proc_act)
        if use_next_obs:
            inputs.append(self._proc_next_obs)
        if len(inputs) == 0:
            msg = "At least one of `use_act`, `use_obs` and `use_next_obs` must be true."
            raise ValueError(msg)
        if use_dones:
            inputs.append(self._proc_dones)

        self._reward, self.layers = networks.build_and_apply_mlp(inputs, hid_sizes)
        serialize.LayersSerializable.__init__(**params, layers=self.layers)

    @property
    def reward(self):
        return self._reward


class MLPPotentialShaping(BasicRewardModel, PotentialShaping, serialize.LayersSerializable):
    """Potential shaping using MLP to calculate potential."""

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        hid_sizes: Iterable[int] = (32, 32),
        discount: float = 0.99,
        **kwargs,
    ):
        """Builds MLPPotentialShaping.

        Args:
            obs_space: The observation space.
            act_space: The action space.
            hid_sizes: The number of hidden units at each layer in the network.
            discount: The initial discount rate to use.
        """
        params = dict(locals())
        del params["kwargs"]
        params.update(**kwargs)

        BasicRewardModel.__init__(self, obs_space, act_space)
        res = reward_net.build_basic_phi_network(
            hid_sizes, self._proc_obs, self._proc_next_obs, **kwargs
        )
        old_potential, new_potential, layers = res
        end_potential = ConstantLayer("end_potential")
        end_potential.build(())
        PotentialShaping.__init__(
            self, old_potential, new_potential, end_potential.constant, self._proc_dones, discount,
        )

        layers["end_potential"] = end_potential
        layers["discount"] = self._discount
        serialize.LayersSerializable.__init__(**params, layers=layers)


class ConstantLayer(tf.keras.layers.Layer):
    """A layer that computes the same output, regardless of input.

    The output is a constant, repeated to the same shape as the input.
    The constant is a trainable variable, and can also be assigned to explicitly.
    """

    def __init__(
        self,
        name: str = None,
        initializer: Optional[tf.keras.initializers.Initializer] = None,
        dtype: tf.dtypes.DType = tf.float32,
    ):
        """Constructs a ConstantLayer.

        Args:
            name: String name of the layer.
            initializer: The initializer to use for the constant weight.
            dtype: dtype of the constant weight.
        """
        if initializer is None:
            initializer = tf.zeros_initializer()
        self.initializer = initializer

        self._constant = None

        super().__init__(trainable=True, name=name, dtype=dtype)

    def build(self, input_shape):
        self._constant = self.add_weight(
            name="constant", shape=(), initializer=self.initializer, use_resource=True
        )
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
        return {"name": self.name, "initializer": self.initializer, "dtype": self.dtype}


class ConstantReward(BasicRewardModel, serialize.LayersSerializable):
    """Outputs a constant reward value. Constant is a (trainable) variable."""

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        initializer: Optional[tf.keras.initializers.Initializer] = None,
    ):
        BasicRewardModel.__init__(self, obs_space, act_space)
        params = dict(locals())

        self._constant = ConstantLayer(name="constant", initializer=initializer)
        n_batch = tf.shape(self._proc_obs)[0]
        obs = tf.reshape(self._proc_obs, [n_batch, -1])
        # self._reward_output is a scalar constant repeated (n_batch, ) times
        self._reward_output = self._constant(obs[:, 0])

        serialize.LayersSerializable.__init__(**params, layers={"constant": self._constant})

    @property
    def constant(self):
        return self._constant

    @property
    def reward(self):
        return self._reward_output


class ZeroReward(BasicRewardModel, serialize.LayersSerializable):
    """A reward model that always outputs zero."""

    def __init__(self, obs_space: gym.Space, act_space: gym.Space):
        serialize.LayersSerializable.__init__(**locals(), layers={})
        BasicRewardModel.__init__(self, obs_space, act_space)

        n_batch = tf.shape(self._proc_obs)[0]
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
        RewardModel.__init__(self)
        self.reward_net = network
        self.use_test = use_test

    # SOMEDAY(adam): support discount/set_discount for RewardNetShaped

    @property
    def reward(self):
        net = self.reward_net
        return net.reward_output_test if self.use_test else net.reward_output_train

    @property
    def observation_space(self):
        return self.reward_net.observation_space

    @property
    def action_space(self):
        return self.reward_net.action_space

    @property
    def obs_ph(self):
        return (self.reward_net.obs_ph,)

    @property
    def act_ph(self):
        return (self.reward_net.act_ph,)

    @property
    def next_obs_ph(self):
        return (self.reward_net.next_obs_ph,)

    @property
    def dones_ph(self):
        # RewardNet does not handle terminal states.
        # TODO(adam): add support to imitation for this so I can do it for IRL too?
        return ()

    @classmethod
    def _load(cls, directory: str) -> "RewardNetToRewardModel":
        with open(os.path.join(directory, "use_test"), "rb") as f:
            use_test = pickle.load(f)

        net = reward_net.RewardNet.load(os.path.join(directory, "net"))
        return cls(net, use_test=use_test)

    def _save(self, directory: str) -> None:
        with open(os.path.join(directory, "use_test"), "wb") as f:
            pickle.dump(self.use_test, f)

        self.reward_net.save(os.path.join(directory, "net"))


class RewardModelWrapper(RewardModel):
    """Wrapper for RewardModel objects.

    This wrapper is the identity; it is intended to be subclassed.
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
    def discount(self):
        return self.model.discount

    def set_discount(self, discount):
        return self.model.set_discount(discount)

    @property
    def observation_space(self):
        return self.model.observation_space

    @property
    def action_space(self):
        return self.model.action_space

    @property
    def obs_ph(self):
        return self.model.obs_ph

    @property
    def act_ph(self):
        return self.model.act_ph

    @property
    def next_obs_ph(self):
        return self.model.next_obs_ph

    @property
    def dones_ph(self):
        return self.model.dones_ph

    @classmethod
    def _load(cls: Type[serialize.T], directory: str) -> serialize.T:
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
    def discount(self):
        # SOMEDAY(adam): single representative is misleading, but can't do better with a scalar
        return self.model.discount

    def set_discount(self, discount):
        for m, _ in self._models.values():
            m.set_discount(discount)

    @property
    def obs_ph(self):
        return tuple(itertools.chain(*[m.obs_ph for m, _ in self.models.values()]))

    @property
    def next_obs_ph(self):
        return tuple(itertools.chain(*[m.next_obs_ph for m, _ in self.models.values()]))

    @property
    def act_ph(self):
        return tuple(itertools.chain(*[m.act_ph for m, _ in self.models.values()]))

    @property
    def dones_ph(self):
        return tuple(itertools.chain(*[m.dones_ph for m, _ in self.models.values()]))

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

    def __init__(self, wrapped: RewardModel, scale: bool = True, shift: bool = True):
        """Wraps wrapped, adding a shift and scale parameter if specified.

        Args:
            wrapped: The RewardModel to wrap.
            scale: If true, adds a positive scale parameter.
            shift: If true, adds a shift parameter.
        """
        self._log_scale_layer = None
        if scale:
            self._log_scale_layer = ConstantLayer("log_scale")
            self._log_scale_layer.build(())
            scale = tf.exp(self._log_scale_layer.constant)  # force to be non-negative
        else:
            scale = tf.constant(1.0)

        models = {"wrapped": (wrapped, scale)}

        self._shift = None
        if shift:
            self._shift = shift = ConstantReward(wrapped.observation_space, wrapped.action_space)
        else:
            shift = ZeroReward(wrapped.observation_space, wrapped.action_space)
        models["constant"] = (shift, tf.constant(1.0))

        super().__init__(models)

    def fit_lstsq(
        self, batch: types.Transitions, target: RewardModel, shaping: Optional[RewardModel]
    ) -> AffineParameters:
        """Sets the shift and scale parameters to try and match target, given shaping.

        Uses `least_l2_affine` to find the least-squares affine parameters between
        `scale * source + shift + shaping` and `target`.

        If one of `scale` or `shift` is not present in this `AffineParameter`, it will be skipped,
        and the value corresponding to the identify transformation will be returned.

        Args:
            batch: A batch of transitions to estimate the affine parameters on.
            target: The reward model to try and match.
            shaping: Optionally, potential shaping to add to the source. If omitted, will default
                to all-zero.

        Returns:
            The least-squares affine parameters. They will also be set as a side-effect.
        """
        sess = tf.get_default_session()
        source = self.models["wrapped"][0]
        target_tensor = target.reward
        models = [source, target]
        if shaping is not None:
            target_tensor -= shaping.reward
            models.append(shaping)

        reward_tensors = [source.reward, target_tensor]
        feed_dict = make_feed_dict(models, batch)
        source_reward, target_reward = sess.run(reward_tensors, feed_dict=feed_dict)

        has_shift = self._shift is not None
        has_scale = self._log_scale_layer is not None
        # Find affine parameters minimizing L2 distance of `scale * source + shift + shaping`
        # to `target`. Note if `shaping` is present, have subtracted it from `target` above.
        params = least_l2_affine(source_reward, target_reward, shift=has_shift, scale=has_scale)
        scale = max(params.scale, np.finfo(params.scale).eps)  # ensure strictly positive
        if has_shift:
            self.set_shift(params.shift)
        if has_scale:
            self.set_log_scale(np.log(scale))

        return params

    @property
    def shift(self) -> tf.Tensor:
        """The additive shift."""
        return self.models["constant"][0].constant.constant

    @property
    def scale(self) -> tf.Tensor:
        """The multiplicative dilation."""
        return self.models["wrapped"][1]

    def set_shift(self, constant: float) -> None:
        """Sets the constant shift.

        Args:
            constant: The shift factor.

        Raises:
            TypeError if the AffineTransform does not have a shift parameter.
        """
        if self._shift is not None:
            return self._shift.constant.set_constant(constant)
        else:
            raise TypeError("Calling `set_shift` on AffineTransform with `shift=False`.")

    def set_log_scale(self, log_scale: float) -> None:
        """Sets the log of the scale factor.

        Args:
            log_scale: The log scale factor.

        Raises:
            TypeError if the AffineTransform does not have a scale parameter.
        """
        if self._log_scale_layer is not None:
            return self._log_scale_layer.set_constant(log_scale)
        else:
            raise TypeError("Calling `set_log_scale` on AffineTransform with `scale=False`.")

    def get_weights(self) -> AffineParameters:
        """Extract affine parameters from a model.

        Returns:
            The current affine parameters (scale and shift), from the perspective of
            mapping the *original* onto the *target*; that is, the inverse of the
            transformation actually performed in the model. (This is for ease of
            comparison with results returned by other methods.)
        """
        sess = tf.get_default_session()
        shift, scale = sess.run([self.shift, self.scale])
        return AffineParameters(shift=shift, scale=scale)

    @classmethod
    def _load(cls, directory: str) -> "AffineTransform":
        """Load an AffineTransform.

        We use the same saving logic as LinearCombinationModelWrapper. This works
        as AffineTransform does not have any extra state needed for inference.
        (There is self._log_scale which is used in pretraining.)

        Args:
            directory: The directory to load from.

        Returns:
            The deserialized AffineTransform instance.
        """
        obj = cls.__new__(cls)
        lc = LinearCombinationModelWrapper._load(directory)
        LinearCombinationModelWrapper.__init__(obj, lc.models)
        return obj


class MLPPotentialShapingWrapper(LinearCombinationModelWrapper):
    """Adds potential shaping to an underlying reward model."""

    def __init__(self, wrapped: RewardModel, **kwargs):
        """Wraps wrapped with a PotentialShaping instance.

        Args:
            wrapped: The model to add shaping to.
            **kwargs: Passed through to PotentialShaping.
        """
        shaping = MLPPotentialShaping(wrapped.observation_space, wrapped.action_space, **kwargs)

        super().__init__(
            {"wrapped": (wrapped, tf.constant(1.0)), "shaping": (shaping, tf.constant(1.0))}
        )


def make_feed_dict(
    models: Iterable[RewardModel], batch: types.Transitions
) -> Dict[tf.Tensor, np.ndarray]:
    """Construct a feed dictionary for models for data in batch."""
    assert batch.obs.shape == batch.next_obs.shape
    assert batch.obs.shape[0] == batch.acts.shape[0]

    if models:
        a_model = next(iter(models))
        assert batch.obs.shape[1:] == a_model.observation_space.shape
        assert batch.acts.shape[1:] == a_model.action_space.shape
        for m in models:
            assert a_model.observation_space == m.observation_space
            assert a_model.action_space == m.action_space

    feed_dict = {}
    for m in models:
        feed_dict.update({ph: batch.obs for ph in m.obs_ph})
        feed_dict.update({ph: batch.acts for ph in m.act_ph})
        feed_dict.update({ph: batch.next_obs for ph in m.next_obs_ph})
        feed_dict.update({ph: batch.dones for ph in m.dones_ph})

    return feed_dict


def evaluate_models(
    models: Mapping[K, RewardModel], batch: types.Transitions
) -> Mapping[K, np.ndarray]:
    """Computes prediction of reward models."""
    reward_outputs = {k: m.reward for k, m in models.items()}
    feed_dict = make_feed_dict(models.values(), batch)
    return tf.get_default_session().run(reward_outputs, feed_dict=feed_dict)


def compute_return_from_rews(
    rews: Mapping[K, np.ndarray], dones: np.ndarray, discount: float = 1.0
) -> Mapping[K, np.ndarray]:
    """Computes the returns from rewards `rews` with episode endings given by `dones`.

    Args:
        rews: A mapping of reward arrays, each of shape `(trajectory_len,)`.
        dones: A boolean array of shape `(trajectory_len,)`.
        discount: The discount rate; defaults to undiscounted.

    Returns:
        A collection of NumPy arrays containing the returns from each model.
    """
    for v in rews.values():
        assert v.shape == (len(dones),)

    # To compute returns, we must sum rewards belonging to each episode in the flattened
    # sequence. First, find the episode boundaries.
    ep_boundaries = np.where(dones)[0]
    # Convert ep_boundaries from inclusive to exclusive range.
    idxs = ep_boundaries + 1
    # NumPy equivalent of Python idxs = [0] + ep_boundaries
    idxs = np.pad(idxs, (1, 0), "constant")

    start_idxs = idxs[:-1]
    end_idxs = idxs[1:]
    if discount < 1.0:
        # Discounted -- need to do the slow but general thing.
        ep_returns = {}
        for k, v in rews.items():
            rets = []
            for start, end in zip(start_idxs, end_idxs):
                rets.append(np.polyval(np.flip(v[start:end]), discount))
            ep_returns[k] = np.array(rets)
    else:
        # Fast path for undiscounted case: sum over the slices.
        if len(start_idxs) == 0:
            # No completed episodes: nothing to compute returns over.
            ep_returns = {k: np.array([]) for k in rews.keys()}
        else:
            # Truncate at last episode completion index.
            last_idx = idxs[-1]
            rews = {k: v[:last_idx] for k, v in rews.items()}
            # Now add over each interval split by the episode boundaries.
            ep_returns = {k: np.add.reduceat(v, start_idxs) for k, v in rews.items()}

    return ep_returns


def compute_return_of_models(
    models: Mapping[K, RewardModel],
    trajectories: Sequence[types.Trajectory],
    discount: float = 1.0,
) -> Mapping[K, np.ndarray]:
    """Computes the returns of each trajectory under each model.

    Args:
        models: A collection of reward models.
        trajectories: A sequence of trajectories.
        discount: The discount rate; defaults to undiscounted.

    Returns:
        A collection of NumPy arrays containing the returns from each model.
    """
    # Reward models are Markovian so only operate on a timestep at a time,
    # expecting input shape (batch_size, ) + {obs,act}_shape. Flatten the
    # trajectories to accommodate this.
    transitions = rollout.flatten_trajectories(trajectories)
    preds = evaluate_models(models, transitions)

    return compute_return_from_rews(preds, transitions.dones, discount)


def evaluate_potentials(
    potentials: Iterable[PotentialShaping], transitions: types.Transitions
) -> np.ndarray:
    """Computes prediction of potential shaping models."""
    old_pots = [p.old_potential for p in potentials]
    new_pots = [p.new_potential for p in potentials]
    feed_dict = make_feed_dict(potentials, transitions)
    return tf.get_default_session().run([old_pots, new_pots], feed_dict=feed_dict)


def least_l2_affine(
    source: np.ndarray, target: np.ndarray, shift: bool = True, scale: bool = True
) -> AffineParameters:
    """Finds the squared-error minimizing affine transform.

    Args:
        source: a 1D array consisting of the reward to transform.
        target: a 1D array consisting of the target to match.
        shift: affine includes constant shift.
        scale: affine includes rescale.

    Returns:
        (shift, scale) such that (scale * reward + shift) has minimal squared-error from target.

    Raises:
        ValueError if source or target are not 1D arrays, or if neither shift or scale are True.
    """
    if source.ndim != 1:
        raise ValueError("source must be vector.")
    if target.ndim != 1:
        raise ValueError("target must be vector.")
    if not (shift or scale):
        raise ValueError("At least one of shift and scale must be True.")

    a_vals = []
    if shift:
        # Positive and negative constant.
        # The shift will be the sum of the coefficients of these terms.
        a_vals += [np.ones_like(source), -np.ones_like(source)]
    if scale:
        a_vals += [source]
    a_vals = np.stack(a_vals, axis=1)
    # Find x such that a_vals.dot(x) has least-squared error from target, where x >= 0.
    coefs, _ = scipy.optimize.nnls(a_vals, target)

    shift_param = 0.0
    scale_idx = 0
    if shift:
        shift_param = coefs[0] - coefs[1]
        scale_idx = 2

    scale_param = 1.0
    if scale:
        scale_param = coefs[scale_idx]

    return AffineParameters(shift=shift_param, scale=scale_param)
