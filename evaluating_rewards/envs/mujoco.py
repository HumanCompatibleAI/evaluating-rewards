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

"""Reward functions for Gym environments."""

import abc

from evaluating_rewards import rewards
from evaluating_rewards import serialize as reward_serialize
import gym
from imitation.util import registry
from imitation.util import serialize
import numpy as np
from stable_baselines.common import vec_env
import tensorflow as tf


class MujocoHardcodedReward(rewards.BasicRewardModel,
                            serialize.LayersSerializable):
  """Hardcoded (non-trainable) reward model for a MuJoCo environment."""

  def __init__(self, observation_space: gym.Space, action_space: gym.Space,
               **kwargs):
    """Constructs the reward model.

    Args:
      observation_space: The observation space of the environment.
      action_space: The action space of the environment.
      **kwargs: Extra parameters to serialize and store in the instance,
          accessible as attributes.
    """
    rewards.BasicRewardModel.__init__(self, observation_space, action_space)
    serialize.LayersSerializable.__init__(self,
                                          layers={},
                                          observation_space=observation_space,
                                          action_space=action_space,
                                          **kwargs)
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


class HalfCheetahGroundTruthReward(MujocoHardcodedReward):
  """Reward for HalfCheetah-v2. Matches ground truth with default settings."""

  def __init__(self,
               observation_space: gym.Space,
               action_space: gym.Space,
               forward: bool = True,
               ctrl_coef: float = 0.1):
    """Constructs the reward model.

    Args:
      observation_space: The observation space of the environment.
      action_space: The action space of the environment.
      forward: whether to reward running forward (True) or backwards (False).
      ctrl_coef: Scale factor for control penalty.
    """
    super().__init__(observation_space, action_space,
                     forward=forward, ctrl_coef=ctrl_coef)

  def build_reward(self) -> tf.Tensor:
    """Intended to match the reward returned by gym.HalfCheetahEnv.

    Known differences: none.

    Returns:
      A Tensor containing predicted rewards.
    """
    # observations consist of concat(qpos, qvel)
    n = 9
    assert self.observation_space.shape == (2 * n,)
    # action = control, 6-dimensional (not all bodies actuated)
    assert self.action_space.shape == (6,)

    # Average velocity of C.O.M.
    # TODO(): would be more DRY to read dt from the environment
    # However, it should not change as Gym guarantees named environments
    # semantics should stay fixed. Extracting this from the environment is
    # non-trivial: it'd require taking a venv as input (which makes
    # serialization more challenging), and then calling env_method to access
    # the dt property.
    dt = 0.05  # model timestep 0.01, frameskip 5
    reward_run = (self._proc_next_obs[:, 0] - self._proc_obs[:, 0]) / dt
    # Control penalty
    reward_ctrl = tf.reduce_sum(tf.square(self._proc_act), axis=-1)
    forward_sign = 1.0 if self.forward else -1.0
    reward = forward_sign * reward_run - self.ctrl_coef * reward_ctrl

    return reward


class HopperGroundTruthReward(MujocoHardcodedReward):
  """Reward for Hopper-v2. Matches ground truth with default settings."""

  def __init__(self,
               observation_space: gym.Space,
               action_space: gym.Space,
               alive_bonus: float = 1.0,
               forward: bool = True,
               ctrl_coef: float = 1e-3):
    """Constructs the reward model.

    Args:
      observation_space: The observation space of the environment.
      action_space: The action space of the environment.
      alive_bonus: constant term added to each reward in non-terminal states.
      forward: Whether to reward running forward (True) or backwards (False).
      ctrl_coef: Scale factor for control penalty.
    """
    super().__init__(observation_space, action_space,
                     alive_bonus=alive_bonus, ctrl_coef=ctrl_coef,
                     forward=forward)

  def build_reward(self) -> tf.Tensor:
    """Intended to match the reward returned by gym.HopperEnv.

    Known differences:
      - If the starting observation is terminal (i.e. Gym would have returned
        done at the *previous* timestep), we return a zero reward.

        By contrast, Gym would compute the reward as usual, but these rewards
        would typically never be observed as the episode has ended, effectively
        corresponding to being in a zero-reward absorbing state.

        To match Gym behavior on trajectories, it is important to respect the
        `done` condition, since otherwise a transition from a terminal to a
        non-terminal state is possible (which would then get reward in
        subsequent steps). However, zeroing reward is sufficient to match the
        Gym behavior on individual transitions.

    Returns:
      A Tensor containing predicted rewards.
    """
    # Observation is concat(qpos, clipped(qvel)).
    n = 6
    assert self.observation_space.shape == (2 * n,)
    assert self.action_space.shape == (3,)

    forward_sign = 1.0 if self.forward else -1.0
    dt = 0.008  # model timestep 0.002, frameskip 4
    reward_vel = (self._proc_next_obs[:, 0] - self._proc_obs[:, 0]) / dt
    reward_ctrl = tf.reduce_sum(tf.square(self._proc_act), axis=-1)
    reward = (self.alive_bonus
              + forward_sign * reward_vel
              - self.ctrl_coef * reward_ctrl)

    height = self._proc_obs[:, 1]
    ang = self._proc_obs[:, 2]
    finite = tf.math.reduce_all(tf.math.is_finite(self._proc_obs), axis=-1)
    small_enough = tf.math.reduce_all(tf.abs(self._proc_obs[:, 2:]) < 100,
                                      axis=-1)
    alive_conditions = [finite, small_enough, height > 0.7, tf.abs(ang) < 0.2]
    alive = tf.math.reduce_all(alive_conditions, axis=0)
    # zero out rewards when starting observation was terminal
    reward = reward * tf.cast(alive, tf.float32)

    return reward


class HopperBackflipReward(MujocoHardcodedReward):
  """Reward for Hopper-v2 to make it do a backflip, rather than hop forward.

  Based on reward function in footnote of:
   https://openai.com/blog/deep-reinforcement-learning-from-human-preferences/
  """

  def __init__(self,
               observation_space: gym.Space,
               action_space: gym.Space,
               forward: bool = True,
               ctrl_coef: float = 0.1):
    """Constructs the reward model.

    Args:
      observation_space: The observation space of the environment.
      action_space: The action space of the environment.
      forward: whether to reward running forward (True) or backwards (False).
      ctrl_coef: Scale factor for control penalty.
    """
    super().__init__(observation_space, action_space,
                     forward=forward, ctrl_coef=ctrl_coef)

  def build_reward(self) -> tf.Tensor:
    """Intended to match the backflip reward described by Christiano et al.

    Known differences: we include a control cost by default.

    Returns:
      A tensor containing reward, shape (batch_size,).
    """
    # Observation is qpos[1:] + clipped(qvel).
    npos = 6
    nvel = 6
    nctrl = 3
    assert self.observation_space.shape == (npos + nvel,)
    assert self.action_space.shape == (nctrl,)

    forward_sign = 1.0 if self.forward else -1.0
    backroll = -forward_sign * self._proc_obs[:, npos+2]  # qvel[2]
    height = self._proc_obs[:, 1]
    # Control in the same direction as the velocity?
    nuncontrolled = 3  # first three bodies are unactuated.
    vel_act = [self._proc_act[:, i] * self._proc_obs[:, npos+nuncontrolled+i]
               for i in range(nctrl)]
    vel_act = sum(vel_act)
    backslide = -self._proc_obs[:, 6]
    reward_ctrl = tf.reduce_sum(tf.square(self._proc_act), axis=-1)

    reward = (backroll * (1.0 + .3 * height + .1 * vel_act + .05 * backslide)
              - self.ctrl_coef * reward_ctrl)
    return reward


class PointMazeReward(MujocoHardcodedReward):
  """Reward for imitation/PointMaze{Left,Right}-v0.

  This in turn is based on on Fu et al (2018)'s PointMaze environment:
  https://arxiv.org/pdf/1710.11248.pdf
  """

  def __init__(self,
               observation_space: gym.Space,
               action_space: gym.Space,
               target: np.ndarray,
               ctrl_coef: float = 1e-3):
    """Constructs the reward model.

    Args:
      observation_space: The observation space of the environment.
      action_space: The action space of the environment.
      target: The position of the target (goal state).
      ctrl_coef: Scale factor for control penalty.
    """
    super().__init__(observation_space, action_space,
                     target=target, ctrl_coef=ctrl_coef)
    self.target = target

  @classmethod
  def from_venv(cls, venv: vec_env.VecEnv, *args, **kwargs):
    """Factory constructor, extracting spaces and target from environment."""
    target = venv.env_method("get_body_com", "target")
    assert np.all(target[0] == target)
    return PointMazeReward(venv.observation_space, venv.action_space,
                           target[0], *args, **kwargs)

  def build_reward(self) -> tf.Tensor:
    """Matches the ground-truth reward, with default constructor arguments.

    Known differences: none.

    Returns:
      A tensor containing reward, shape (batch_size,).
    """
    assert self.observation_space.shape == (3,)
    particle = self._proc_obs[:, 0:3]
    reward_dist = tf.norm(particle - self.target, axis=-1)
    reward_ctrl = tf.reduce_sum(tf.square(self._proc_act), axis=-1)
    reward = -reward_dist - self.ctrl_coef * reward_ctrl
    return reward


# Register reward models
def _register_models(format_str, cls, forward=True):
  """Registers reward models of type cls under key formatted by format_str."""
  forwards = {
      "Forward": {"forward": forward},
      "Backward": {"forward": not forward}
  }
  control = {"WithCtrl": {}, "NoCtrl": {"ctrl_coef": 0.0}}

  res = {}
  for k1, cfg1 in forwards.items():
    for k2, cfg2 in control.items():
      fn = registry.build_loader_fn_require_space(cls, **cfg1, **cfg2)
      key = format_str.format(k1 + k2)
      reward_serialize.reward_registry.register(key=key, value=fn)
  return res


def _register_point_maze():
  control = {"WithCtrl": {}, "NoCtrl": {"ctrl_coef": 0.0}}
  for k, cfg in control.items():
    fn = registry.build_loader_fn_require_env(PointMazeReward.from_venv, **cfg)
    reward_serialize.reward_registry.register(
        key=f"imitation/PointMazeGroundTruth{k}-v0",
        value=fn
    )


_register_models("evaluating_rewards/HalfCheetahGroundTruth{}-v0",
                 HalfCheetahGroundTruthReward)
_register_models("evaluating_rewards/HopperGroundTruth{}-v0",
                 HopperGroundTruthReward)
_register_models("evaluating_rewards/HopperBackflip{}-v0",
                 HopperBackflipReward, forward=False)
_register_point_maze()
