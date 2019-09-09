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

"""Load reward models of different types."""

import contextlib
from typing import Callable, Iterator
import uuid

from absl import logging
from evaluating_rewards import rewards
from imitation.rewards import reward_net
from imitation.rewards import serialize
from imitation.util import registry
from imitation.util import serialize as util_serialize
from imitation.util import util
import numpy as np
from stable_baselines.common import vec_env
import tensorflow as tf

RewardLoaderFn = Callable[[str, vec_env.VecEnv], rewards.RewardModel]


class RewardRegistry(registry.Registry[RewardLoaderFn]):
  """Registry for RewardModel objects.

  Differs from a usual registry by, on insertion, also inserting a reward
  function that wraps the reward model into
  `imitation.rewards.serialize.reward_registry`.
  """

  def register(self, key, *, value=None, indirect=None):
    super().register(key, value=value, indirect=indirect)

    if key.startswith("imitation/"):
      return  # do not re-insert models from imitation

    @contextlib.contextmanager
    def reward_fn_loader(path: str,
                         venv: vec_env.VecEnv,
                        ) -> Iterator[serialize.RewardFn]:
      """Load a TensorFlow reward model, then convert it into a Callable."""
      reward_model_loader = self.get(key)
      with util.make_session() as (_, sess):
        reward_model = reward_model_loader(path, venv)

        def reward_fn(old_obs: np.ndarray,
                      actions: np.ndarray,
                      new_obs: np.ndarray,
                      steps: np.ndarray,
                     ) -> np.ndarray:
          """Helper method computing reward for registered model."""
          del steps
          batch = rewards.Batch(old_obs=old_obs,
                                actions=actions,
                                new_obs=new_obs)
          fd = rewards.make_feed_dict([reward_model], batch)
          return sess.run(reward_model.reward, feed_dict=fd)

        yield reward_fn

    serialize.reward_registry.register(key=key, value=reward_fn_loader)


reward_registry = RewardRegistry()


def _load_imitation(use_test: bool) -> RewardLoaderFn:
  """Higher-order function returning a reward loader function.

  Arguments:
    use_test: If True, unshaped reward; if False, shaped.

  Returns:
    A function that loads reward networks.
  """
  def f(path: str, venv: vec_env.VecEnv) -> rewards.RewardModel:
    """Loads a reward network saved to path, for environment venv.

    Arguments:
      path: The path to a serialized reward network.
      venv: The environment the reward network should operate in.

    Returns:
      A RewardModel representing the reward network.
    """
    random_id = uuid.uuid4().hex
    with tf.variable_scope(f"model_{random_id}"):
      logging.info(f"Loading imitation reward model from '{path}'")
      net = reward_net.RewardNet.load(path)
      assert venv.observation_space == net.observation_space
      assert venv.action_space == net.action_space
      return rewards.RewardNetToRewardModel(net, use_test=use_test)

  return f


def _load_native(path: str, venv: vec_env. VecEnv) -> rewards.RewardModel:
  """Load a RewardModel that implemented the Serializable interface."""

  random_id = uuid.uuid4().hex
  with tf.variable_scope(f"model_{random_id}"):
    logging.info(f"Loading native evaluating rewards model from '{path}'")
    model = util_serialize.Serializable.load(path)
    if not isinstance(model, rewards.RewardModel):
      raise TypeError(f"Serialized object from '{path}' is not a RewardModel")
    assert venv.observation_space == model.observation_space
    assert venv.action_space == model.action_space

  return model


reward_registry.register(key="imitation/RewardNet_unshaped-v0",
                         value=_load_imitation(True))
reward_registry.register(key="imitation/RewardNet_shaped-v0",
                         value=_load_imitation(False))
reward_registry.register(key="evaluating_rewards/RewardModel-v0",
                         value=_load_native)
reward_registry.register(key="evaluating_rewards/Zero-v0",
                         value=registry.build_loader_fn_require_space(
                             rewards.ZeroReward
                         ))


def load_reward(reward_type: str, reward_path: str, venv: vec_env.VecEnv,
               ) -> rewards.RewardModel:
  """Load serialized reward model.

  Args:
    reward_type: A key in `AGENT_LOADERS`, e.g. `ppo2`.
    reward_path: A path on disk where the policy is stored.
    venv: An environment that the policy is to be used with.

  Returns:
    The reward model loaded from reward_path.
  """
  agent_loader = reward_registry.get(reward_type)
  logging.debug(f"Loading {reward_type} from {reward_path}")
  return agent_loader(reward_path, venv)
