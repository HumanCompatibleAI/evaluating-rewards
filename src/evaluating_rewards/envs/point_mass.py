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

"""A simple point-mass environment in N-dimensions."""

from typing import Optional

import gym
from imitation.envs import resettable_env
from imitation.policies import serialize as policy_serialize
from imitation.util import registry, serialize
import numpy as np
from stable_baselines.common import policies
import tensorflow as tf

from evaluating_rewards import serialize as reward_serialize
from evaluating_rewards.rewards import base


class PointMassEnv(resettable_env.ResettableEnv):
    """A simple point-mass environment."""

    def __init__(
        self,
        ndim: int = 2,
        dt: float = 1e-1,
        ctrl_coef: float = 1.0,
        threshold: float = -1,
        var: float = 1.0,
    ):
        """Builds a PointMass environment.

        Args:
            ndim: Number of dimensions.
            dt: Size of timestep.
            ctrl_coef: Weight for control cost.
            threshold: Distance to goal within which episode terminates.
                    (Set negative to disable episode termination.)
            var: Standard deviation of components of initial state distribution.
        """
        super().__init__()

        self.ndim = ndim
        self.dt = dt
        self.ctrl_coef = ctrl_coef
        self.threshold = threshold
        self.var = var

        substate_space = gym.spaces.Box(-np.inf, np.inf, shape=(ndim,))
        subspaces = {k: substate_space for k in ["pos", "vel", "goal"]}
        self._state_space = gym.spaces.Dict(spaces=subspaces)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(3 * ndim,))
        self._action_space = gym.spaces.Box(-1, 1, shape=(ndim,))

        self.viewer = None
        self._agent_transform = None
        self._goal_transform = None

    def initial_state(self):
        """Choose initial state randomly from region at least 1-step from goal."""
        while True:
            pos = self.rand_state.randn(self.ndim) * np.sqrt(self.var)
            vel = self.rand_state.randn(self.ndim) * np.sqrt(self.var)
            goal = self.rand_state.randn(self.ndim) * np.sqrt(self.var)
            dist = np.linalg.norm(pos - goal)
            min_dist_next = dist - self.dt * np.linalg.norm(vel)
            if min_dist_next > self.threshold:
                break
        return {"pos": pos, "vel": vel, "goal": goal}

    def transition(self, old_state, action):
        action = np.array(action)
        action = action.clip(-1, 1)
        return {
            "pos": old_state["pos"] + self.dt * old_state["vel"],
            "vel": old_state["vel"] + self.dt * action,
            "goal": old_state["goal"],
        }

    def reward(self, old_state, action, new_state):
        del old_state
        dist = np.linalg.norm(new_state["pos"] - new_state["goal"])
        ctrl_penalty = np.dot(action, action)
        return -dist - self.ctrl_coef * ctrl_penalty

    def terminal(self, state, step: int) -> bool:
        """Terminate if agent within threshold of goal.

        Set threshold to be negative to disable early termination, making environment
        fixed horizon.
        """
        dist = np.linalg.norm(state["pos"] - state["goal"])
        return bool(dist < self.threshold)

    def obs_from_state(self, state):
        obs = np.concatenate([state["pos"], state["vel"], state["goal"]], axis=-1)
        return obs.astype(np.float32)

    def state_from_obs(self, obs):
        return {
            "pos": obs[..., 0 : self.ndim],
            "vel": obs[..., self.ndim : 2 * self.ndim],
            "goal": obs[..., 2 * self.ndim : 3 * self.ndim],
        }

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering  # pylint:disable=import-outside-toplevel

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-5, 5, -5, 5)

            def make_circle(**kwargs):
                obj = rendering.make_circle(**kwargs)
                transform = rendering.Transform()
                obj.add_attr(transform)
                self.viewer.add_geom(obj)
                return obj, transform

            goal, self._goal_transform = make_circle(radius=0.2)
            goal.set_color(1.0, 0.85, 0.0)  # golden
            _, self._agent_transform = make_circle(radius=0.1)

        def project(arr):
            if self.ndim == 1:
                assert len(arr) == 1
                return arr[0], 0
            elif self.ndim == 2:
                assert len(arr) == 2
                return tuple(arr)
            else:
                raise ValueError()

        self._goal_transform.set_translation(*project(self.cur_state["goal"]))
        self._agent_transform.set_translation(*project(self.cur_state["pos"]))

        return self.viewer.render(return_rgb_array=(mode == "rgb_array"))

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class PointMassGroundTruth(base.BasicRewardModel, serialize.LayersSerializable):
    """RewardModel representing the true (dense) reward in PointMass."""

    def __init__(
        self, observation_space: gym.Space, action_space: gym.Space, ctrl_coef: float = 1.0
    ):
        serialize.LayersSerializable.__init__(**locals(), layers={})

        self.ndim, remainder = divmod(observation_space.shape[0], 3)
        assert remainder == 0
        self.ctrl_coef = ctrl_coef

        base.BasicRewardModel.__init__(self, observation_space, action_space)
        self._reward = self.build_reward()

    def build_reward(self):
        """Computes reward from observation and action in PointMass environment."""
        pos = self._proc_next_obs[:, 0 : self.ndim]
        goal = self._proc_next_obs[:, 2 * self.ndim : 3 * self.ndim]
        dist = tf.norm(pos - goal, axis=-1)
        ctrl_cost = tf.reduce_sum(tf.square(self._proc_act), axis=-1)
        return -dist - self.ctrl_coef * ctrl_cost

    @property
    def reward(self):
        """Reward tensor."""
        return self._reward


class PointMassSparseReward(base.BasicRewardModel, serialize.LayersSerializable):
    """A sparse reward for the point mass being close to the goal.

    Should produce similar behavior to PointMassGroundTruth. However, it is not
    equivalent up to potential shaping.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        ctrl_coef: float = 1.0,
        threshold: float = 0.05,
        goal_offset: Optional[np.ndarray] = None,
    ):
        """Constructs a PointMassSparseReward instance.

        Args:
            observation_space: Observation space of environment.
            action_space: Action of environment.
            ctrl_coef: The multiplier for the quadratic control penalty.
            threshold: How near the point mass must be to the goal to receive reward.
            goal_offset: If specified, shifts the goal in the direction specified.
                    The larger this is, the more dissimilar the reward model and resulting
                    policy will be from PointMassGroundTruth.
        """
        serialize.LayersSerializable.__init__(**locals(), layers={})

        self.ndim, remainder = divmod(observation_space.shape[0], 3)
        assert remainder == 0
        self.ctrl_coef = ctrl_coef
        self.threshold = threshold
        self.goal_offset = goal_offset
        base.BasicRewardModel.__init__(self, observation_space, action_space)

        self._reward = self.build_reward()

    def build_reward(self):
        """Computes reward from observation and action in PointMass environment."""
        pos = self._proc_obs[:, 0 : self.ndim]
        goal = self._proc_obs[:, 2 * self.ndim : 3 * self.ndim]
        if self.goal_offset is not None:
            goal += self.goal_offset[np.newaxis, :]
        dist = tf.norm(pos - goal, axis=-1)
        goal_reward = tf.to_float(dist < self.threshold)
        ctrl_cost = tf.reduce_sum(tf.square(self._proc_act), axis=-1)
        return goal_reward - self.ctrl_coef * ctrl_cost

    @property
    def reward(self):
        """Reward tensor."""
        return self._reward


def _point_mass_dist(obs: tf.Tensor, ndim: int) -> tf.Tensor:
    pos = obs[:, 0:ndim]
    goal = obs[:, 2 * ndim : 3 * ndim]
    return tf.norm(pos - goal, axis=-1)


# pylint false positive: thinks `reward` is missing, but defined in `rewards.PotentialShaping`
class PointMassShaping(
    base.PotentialShaping, base.BasicRewardModel, serialize.LayersSerializable
):  # pylint:disable=abstract-method
    """Potential shaping term, based on distance to goal."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        discount: float = 1.0,
    ):
        """Builds PointMassShaping.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            discount: The initial discount rate to use.
        """
        params = dict(locals())

        base.BasicRewardModel.__init__(self, observation_space, action_space)
        self.ndim, remainder = divmod(observation_space.shape[0], 3)
        assert remainder == 0

        old_potential = -_point_mass_dist(self._proc_obs, self.ndim)
        new_potential = -_point_mass_dist(self._proc_next_obs, self.ndim)
        end_potential = tf.constant(0.0)
        base.PotentialShaping.__init__(
            self, old_potential, new_potential, end_potential, self._proc_dones, discount
        )

        self.set_discount(discount)  # set it so no need for TF initializer to be called
        serialize.LayersSerializable.__init__(**params, layers={"discount": self._discount})


class PointMassDenseReward(base.LinearCombinationModelWrapper):
    """Sparse reward plus potential shaping."""

    def __init__(
        self, observation_space: gym.Space, action_space: gym.Space, discount: float = 1.0, **kwargs
    ):
        sparse = PointMassSparseReward(observation_space, action_space, **kwargs)
        # pylint thinks PointMassShaping is abstract but it's concrete.
        shaping = PointMassShaping(  # pylint:disable=abstract-class-instantiated
            observation_space, action_space, discount
        )
        models = {"sparse": (sparse, tf.constant(1.0)), "shaping": (shaping, tf.constant(10.0))}
        super().__init__(models)


class PointMassPolicy(policies.BasePolicy):
    """Hard-coded policy that accelerates towards goal."""

    def __init__(
        self, observation_space: gym.Space, action_space: gym.Space, magnitude: float = 1.0
    ):
        super().__init__(
            sess=None,
            ob_space=observation_space,
            ac_space=action_space,
            n_env=1,
            n_steps=1,
            n_batch=1,
        )
        self.ndim, remainder = divmod(observation_space.shape[0], 3)
        assert remainder == 0
        self.magnitude = magnitude

    def step(self, obs, state=None, mask=None, deterministic=False):
        del deterministic
        pos = obs[:, 0 : self.ndim]
        vel = obs[:, self.ndim : 2 * self.ndim]
        goal = obs[:, 2 * self.ndim : 3 * self.ndim]
        target_vel = goal - pos
        target_vel = target_vel / np.linalg.norm(target_vel, axis=1).reshape(-1, 1)
        delta_vel = target_vel - vel
        delta_vel_norm = np.linalg.norm(delta_vel, ord=np.inf, axis=1).reshape(-1, 1)
        act = delta_vel / np.maximum(delta_vel_norm, 1e-4)
        act = act.clip(-1, 1)
        return act, None, None, None

    def proba_step(self, obs, state=None, mask=None):
        raise NotImplementedError()


# Register custom policies with imitation
policy_serialize.policy_registry.register(
    key="evaluating_rewards/PointMassHardcoded-v0",
    value=registry.build_loader_fn_require_space(registry.dummy_context(PointMassPolicy)),
)

# Register custom rewards with evaluating_rewards
reward_serialize.reward_registry.register(
    key="evaluating_rewards/PointMassGroundTruth-v0",
    value=registry.build_loader_fn_require_space(PointMassGroundTruth),
)
reward_serialize.reward_registry.register(
    key="evaluating_rewards/PointMassSparseWithCtrl-v0",
    value=registry.build_loader_fn_require_space(PointMassSparseReward),
)
reward_serialize.reward_registry.register(
    key="evaluating_rewards/PointMassSparseNoCtrl-v0",
    value=registry.build_loader_fn_require_space(PointMassSparseReward, ctrl_coef=0.0),
)
reward_serialize.reward_registry.register(
    key="evaluating_rewards/PointMassDenseWithCtrl-v0",
    value=registry.build_loader_fn_require_space(PointMassDenseReward),
)
reward_serialize.reward_registry.register(
    key="evaluating_rewards/PointMassDenseNoCtrl-v0",
    value=registry.build_loader_fn_require_space(PointMassDenseReward, ctrl_coef=0.0),
)
