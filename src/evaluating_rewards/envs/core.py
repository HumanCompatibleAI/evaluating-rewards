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

"""Base classes for environment rewards."""

import abc

import gym
from imitation.util import serialize
import tensorflow as tf

from evaluating_rewards import rewards


class HardcodedReward(rewards.BasicRewardModel, serialize.LayersSerializable):
    """Hardcoded (non-trainable) reward model for a Gym environment."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, **kwargs):
        """Constructs the reward model.

        Args:
            observation_space: The observation space of the environment.
            action_space: The action space of the environment.
            **kwargs: Extra parameters to serialize and store in the instance,
                    accessible as attributes.
        """
        rewards.BasicRewardModel.__init__(self, observation_space, action_space)
        serialize.LayersSerializable.__init__(
            self,
            layers={},
            observation_space=observation_space,
            action_space=action_space,
            **kwargs,
        )
        self._reward = self.build_reward()

    def __getattr__(self, name):
        try:
            return self._kwargs[name]
        except KeyError:
            raise AttributeError(f"Attribute '{name}' not present in self._kwargs")

    @abc.abstractmethod
    def build_reward(self) -> tf.Tensor:
        """Computes reward from observation, action and next observation.

        Returns:
            A tensor containing reward, shape (batch_size,).
        """

    @property
    def reward(self):
        """Reward tensor, shape (batch_size,)."""
        return self._reward
