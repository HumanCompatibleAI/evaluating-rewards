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

"""Greedy Monte-Carlo optimization of a reward model with a true simulator."""

import warnings

from imitation.policies import serialize as policy_serialize
from imitation.util import registry
import numpy as np
from stable_baselines.common import policies, vec_env
import tensorflow as tf

from evaluating_rewards import rewards, serialize


class MonteCarloGreedyPolicy(policies.BasePolicy):
    """Policy choosing actions to greedily maximize reward."""

    def __init__(self, venv: vec_env.VecEnv, reward_model: rewards.RewardModel, n_samples=100):
        """Builds a MonteCarloGreedyPolicy. Note this is not trainable.

        At each timestep, for each observation it samples n_samples actions,
        and chooses the one with the highest *instantaneous* reward under
        reward_model.

        Reward models can depend on the previous observation, action and next
        observation. If env is a ModelBasedEnv, we compute the next observation
        by using the `transition` function. Note: `transition` can be stochastic.

        Otherwise, we assume the next observation is the same as the current
        observation. This is generally false, and may well cause problems: in
        particular, the contribution of potential shaping will always be zero.
        But there's not much else that can be done.

        Args:
            venv: A VecEnv environment.
            reward_model: A reward model to myopically maximize.
            n_samples: The number of actions to choose from.
        """
        super().__init__(
            sess=tf.get_default_session(),
            ob_space=venv.observation_space,
            ac_space=venv.action_space,
            # The following parameters don't matter as the placeholders are unused
            n_env=1,
            n_steps=1,
            n_batch=1,
        )

        self.venv = venv
        self.reward_model = reward_model
        self.n_samples = n_samples

    def seed(self, seed=None):
        self.ac_space.seed(seed)

    def step(self, obs, state=None, mask=None, deterministic=False):
        # actions: (n_samples, ) + ac_space, obs: (batch_size, ) + ob_space
        actions = np.array([self.ac_space.sample() for _ in range(self.n_samples)])
        # dup_actions: (1, n_samples) + ac_space
        # dup_obs: (batch_size, 1) + ob_space
        dup_actions = actions[np.newaxis, :]
        dup_obs = obs[:, np.newaxis]
        # dup_actions: (batch_size, n_samples) + ac_space
        # dup_obs: (batch_size, n_samples) + ob_space
        batch_size = obs.shape[0]
        dup_actions = dup_actions.repeat(batch_size, axis=0)
        dup_obs = dup_obs.repeat(self.n_samples, axis=1)
        # dup_actions: (batch_size * n_samples, ) + ac_space
        # dup_obs: (batch_size * n_samples, ) + ob_space
        dup_actions = dup_actions.reshape(batch_size * self.n_samples, -1)
        dup_obs = dup_obs.reshape(batch_size * self.n_samples, -1)

        try:
            # TODO(): vectorizing transition would improve performance
            next_obs = []
            for old_ob, act in zip(dup_obs, dup_actions):
                old_s = self.venv.env_method("state_from_obs", old_ob, indices=[0])[0]
                new_s = self.venv.env_method("transition", old_s, act, indices=[0])[0]
                next_ob = self.venv.env_method("obs_from_state", new_s, indices=[0])[0]
                next_obs.append(next_ob)
            next_obs = np.array(next_obs)
        except AttributeError:
            warnings.warn(
                "Environment is not model-based: will assume next "
                "observation is the same as current observation."
            )
            next_obs = dup_obs

        batch = rewards.Batch(obs=dup_obs, actions=dup_actions, next_obs=next_obs)
        feed_dict = rewards.make_feed_dict([self.reward_model], batch)
        # TODO(): add a function to RewardModel to compute this?
        reward = self.sess.run(self.reward_model.reward, feed_dict=feed_dict)

        reward = np.reshape(reward, (batch_size, self.n_samples))
        best_actions_idx = reward.argmax(axis=1)
        best_actions = actions[best_actions_idx]

        return best_actions, None, None, None

    def proba_step(self, obs, state=None, mask=None):
        raise NotImplementedError()


@registry.sess_context
def load_monte_carlo_greedy(path: str, env: vec_env.VecEnv) -> MonteCarloGreedyPolicy:
    reward_type, reward_path = path.split(":")
    reward_model = serialize.load_reward(reward_type, reward_path, env)
    return MonteCarloGreedyPolicy(env, reward_model=reward_model)


policy_serialize.policy_registry.register(
    key="evaluating_rewards/MCGreedy-v0", value=load_monte_carlo_greedy
)
