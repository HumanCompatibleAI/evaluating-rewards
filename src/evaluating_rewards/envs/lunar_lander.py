# Copyright 2020 Adam Gleave
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

"""Reward function for Gym LunarLander-v2 environment."""

from gym import spaces
from gym.envs.box2d import lunar_lander
from imitation.util import registry
import numpy as np
import tensorflow as tf

from evaluating_rewards import serialize as reward_serialize
from evaluating_rewards.envs import core


def _potential(obs: tf.Tensor) -> tf.Tensor:
    """Potential function used to compute shaping.

    Based on `shaping` variable in `LunarLander.step()`.
    """
    leg_contact = obs[:, 6] + obs[:, 7]
    l2 = tf.sqrt(tf.math.square(obs[:, 0]) + tf.math.square(obs[:, 1]))
    l2 += tf.sqrt(tf.math.square(obs[:, 2]) + tf.math.square(obs[:, 3]))
    return 10 * leg_contact - 100 * l2 - 100 * tf.abs(obs[:, 4])


class LunarLanderContinuousObservable(lunar_lander.LunarLanderContinuous):
    """LunarLander environment lightly modified from Gym to make reward a function of observation.

    Adds `self.game_over` and `self.lander.awake` flags to state, which are used by Gym
    internally to compute the reward. They are computed by the Box2D simulator, and cannot easily
    be derived from the rest of the state.

    `game_over` is set based on contact forces on the lunar lander. The `lander.awake` flag is
    set when the body is not "asleep":
        "When Box2D determines that a body [...] has come to rest, the body enters a sleep state"
         (see https://box2d.org/documentation/md__d_1__git_hub_box2d_docs_dynamics.html).
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)

    def step(self, action):
        prev_shaping = self.prev_shaping
        obs, rew, done, info = super().step(action)
        shaping = self.prev_shaping
        extra_obs = [1.0 if self.game_over else 0.0, 1.0 if self.lander.awake else 0.0]
        obs = np.concatenate((obs, extra_obs))

        if done:
            # Gym does not apply shaping or control cost to final reward.
            # No control cost is weird but harmless. No shaping is problematic though so we fix
            # it to satisfy Ng et al (1999)'s conditions.
            rew += shaping - prev_shaping

        return obs, rew, done, info


class LunarLanderContinuousGroundTruthReward(core.HardcodedReward):
    """Reward for LunarLanderContinuousObservable. Matches ground truth with default settings."""

    def build_reward(self) -> tf.Tensor:
        """Intended to match the reward returned by gym.LunarLander.

        Known differences:
          - Will disagree on states *after* episode termination due to non-Markovian leg contact
            condition in Gym.

        Returns:
            A Tensor containing predicted rewards.
        """
        # Sparse reward
        game_over = (tf.abs(self._proc_next_obs[:, 0]) >= 1.0) | (self._proc_next_obs[:, 8] > 0)
        landed_safely = self._proc_next_obs[:, 9] == 0.0
        done = game_over | landed_safely
        sparse_reward = -100.0 * tf.cast(game_over, tf.float32)
        sparse_reward += 100.0 * tf.cast(landed_safely, tf.float32)

        # Control cost
        m_thrust = self._proc_act[:, 0] > 0
        m_power_when_act = 0.5 * (tf.clip_by_value(self._proc_act[:, 0], 0.0, 1.0) + 1.0)
        m_power = tf.where(m_thrust, m_power_when_act, 0.0 * m_power_when_act)
        abs_side_act = tf.abs(self._proc_act[:, 1])
        s_thrust = abs_side_act > 0.5
        s_power_when_act = tf.clip_by_value(abs_side_act, 0.5, 1.0)
        s_power = tf.where(s_thrust, s_power_when_act, 0.0 * s_power_when_act)
        ctrl_cost = -0.3 * m_power - 0.03 * s_power
        # Gym does not apply control cost to final step. (Seems weird, but OK.)
        ctrl_cost = tf.where(done, 0 * ctrl_cost, ctrl_cost)

        # Shaping
        # Note this assumes no discount (matching Gym implementation), which will make it
        # not *quite* potential shaping for any RL algorithm using discounting.
        shaping = _potential(self._proc_next_obs) - _potential(self._proc_obs)

        return sparse_reward + shaping + ctrl_cost


reward_serialize.reward_registry.register(
    key="evaluating_rewards/LunarLanderContinuousGroundTruth-v0",
    value=registry.build_loader_fn_require_space(LunarLanderContinuousGroundTruthReward),
)
