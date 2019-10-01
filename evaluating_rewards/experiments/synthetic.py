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

"""Experiments with synthetic randomly generated reward models.

See Colab notebook for use cases.
"""

import functools
from typing import Callable, Iterable, Optional, Type

from absl import logging
from evaluating_rewards import comparisons
from evaluating_rewards import rewards
from evaluating_rewards.experiments import datasets
import gym
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf


TensorCallable = Callable[..., tf.Tensor]

log_normal = functools.partial(np.random.lognormal, mean=0.0, sigma=np.log(10))


def compare_synthetic(observation_space: gym.Space,
                      action_space: gym.Space,
                      dataset_generator: datasets.BatchCallable,
                      reward_noise: Optional[np.ndarray] = None,
                      potential_noise: Optional[np.ndarray] = None,
                      reward_hids: Optional[Iterable[int]] = None,
                      dataset_potential_hids: Optional[Iterable[int]] = None,
                      model_potential_hids: Optional[Iterable[int]] = None,
                      dataset_activation: Optional[TensorCallable] = tf.nn.tanh,
                      model_activation: Optional[TensorCallable] = tf.nn.tanh,
                      state_only: bool = True,
                      scale_fn: Callable[[], float] = lambda: 1,
                      constant_fn: Callable[[float], float] = lambda _: 0,
                      model_affine: bool = True,
                      model_potential: bool = True,
                      discount: float = 0.99,
                      optimizer: Type[tf.train.Optimizer] =
                      tf.train.AdamOptimizer,
                      total_timesteps: int = 2 ** 16,
                      batch_size: int = 128,
                      test_size: int = 4096,
                      pretrain: bool = True,
                      pretrain_size: int = 4096,
                      learning_rate: float = 1e-2,
                     ) -> pd.DataFrame:
  r"""Compares rewards with varying noise to a ground-truth reward.

  Randomly generates ground truth reward model $$r_g$$ and potential $$\phi_g$$.

  Creates an additive noise reward $$r_n$$, and potential $$\phi_n$$.
  For each noise magnitude $$\sigma_r$$ in reward_noise and $$\sigma_{\phi}$$ in
  potential_noise, creating an intermediate noised reward model:
    $$r_i(s,a,s') = r'_g(s,a,s') + \sigma_r\cdot r_n(s,a,s') +
                  + \sigma_{\phi}(\gamma \phi_n(s') - \phi_n(s).)$$

  It then calls `random_scale_fn` and `random_constant_fn` to generate
  scaling and shift parameters $\lambda$ and $c$ respectively, creating:
    $$r_o(s,a,s') = \lambda \cdot r_i(s,a,s') + c.$$

  It then uses `ClosestPotential` to fit a potential $\phi$ to minimize
  the mean-squared error between $r_o$ after shaping with $\phi$ and $r'_g$
  on the dataset.

  Args:
    observation_space: Observation space of reward model.
    action_space: Action space of reward model.
    dataset_generator: A callable generating a dataset.
        The callable takes an argument specifying the batch size, and should
        return an iterator of batches of old observation, action and new
        observations. The canonical example of this is rollouts of a policy
        in an environment.
    reward_noise: Magnitude of additive reward noise.
    potential_noise: Magnitude of additive potential noise.
    reward_hids: Number of hidden units at each layer in reward model.
    dataset_potential_hids: Number of hidden units at each layer in the
        random model used to generate the dataset.
    model_potential_hids: Number of hidden units at each layer in the model
        used to match the dataset.
    dataset_activation: Activation function for random model used to
        generate the dataset.
    model_activation: Activation function for model used to match the dataset.
    state_only: If True, uses state-only ground truth and additive noise.
        This helps reduce the amount of potential shaping implicit in the
        reward noise, and so may produce cleaner results (but be less general).
    scale_fn: Thunk that returns a (possibly random) scale factor.
    constant_fn: Function that returns a (possibly random) aditive constant.
        It is passed the scale returned by `scale_fn`.
    model_affine: Model includes a scaling factor and additive constant.
    model_potential: Model includes a shaping factor.
    discount: Discount rate, $\gamma$ (above).
    optimizer: The TensorFlow optimizer to use to fit the model.
    total_timesteps: Total number of timesteps to train on.
    batch_size: Size of training batch.
    test_size: Size of test dataset.
    pretrain: If True, pretrain affine parameters.
    pretrain_size: Size of dataset to pretrain affine parameters.
    learning_rate: Learning rate.

  Returns:
    A pandas DataFrame with intrinsic and shaping distance returned by
    `summary_comparison`, for each noised output reward model $r_o$ generated.
  """
  # Graph construction
  if reward_noise is None:
    reward_noise = np.arange(0.0, 1.0, 0.2)
  if potential_noise is None:
    potential_noise = np.arange(0.0, 10.0, 2.0)

  noise_kwargs = {}
  if state_only:
    noise_kwargs = {"use_act": False, "use_next_obs": False}

  with tf.variable_scope("ground_truth"):
    ground_truth = rewards.MLPRewardModel(observation_space, action_space,
                                          hid_sizes=reward_hids,
                                          **noise_kwargs)

  with tf.variable_scope("noise"):
    noise_reward = rewards.MLPRewardModel(observation_space, action_space,
                                          hid_sizes=reward_hids,
                                          **noise_kwargs)
    noise_potential = rewards.PotentialShaping(observation_space,
                                               action_space,
                                               hid_sizes=dataset_potential_hids,
                                               activation=dataset_activation,
                                               discount=discount)

    # Additive constant and scaling of ground truth
    gt_scale = scale_fn()
    initializer = tf.initializers.ones
    constant_one_model = rewards.ConstantReward(observation_space,
                                                action_space,
                                                initializer=initializer)
    gt_constant = constant_fn(gt_scale)

  originals = {}
  matchings = {}
  # TODO(): graph construction is quite slow -- investigate speed-ups?
  # (e.g. could re-use graph for different random seeds.)
  for rew_nm in reward_noise:
    for pot_nm in potential_noise:
      with tf.variable_scope(f"rew{rew_nm}_pot{pot_nm}"):
        models = {
            "ground_truth": (ground_truth, gt_scale),
            "noise_reward": (noise_reward, rew_nm * gt_scale),
            "noise_potential": (noise_potential, pot_nm * gt_scale),
            "constant": (constant_one_model, gt_constant),
        }

        noised_ground_shaped = rewards.LinearCombinationModelWrapper(models)
        originals[(rew_nm, pot_nm)] = noised_ground_shaped

        with tf.variable_scope("matching"):
          model_wrapper = functools.partial(
              comparisons.equivalence_model_wrapper,
              affine=model_affine,
              potential=model_potential,
              hid_sizes=model_potential_hids,
              activation=model_activation,
              discount=discount)
          matched = comparisons.RegressWrappedModel(noised_ground_shaped,
                                                    ground_truth,
                                                    model_wrapper=model_wrapper,
                                                    learning_rate=learning_rate,
                                                    optimizer=optimizer)
          matchings[(rew_nm, pot_nm)] = matched

  # Initialization
  sess = tf.get_default_session()
  sess.run(tf.global_variables_initializer())

  # Datasets
  training_generator = dataset_generator(total_timesteps, batch_size)
  test_set = next(dataset_generator(test_size, test_size))

  # Pre-train to initialize affine parameters
  initial_constants = {}
  initial_scales = {}
  pretrain_set = next(dataset_generator(pretrain_size, pretrain_size))
  for key, matched in matchings.items():
    if model_affine and pretrain:
      logging.info(f"Pretraining {key}")
      # Try to rescale the original model to match target.
      # This ignores the (randomly initialized) potential shaping,
      # which will make our estimated statistics less accurate.
      original = matched.model_extra["original"]
      initial = matched.model_extra["affine"].pretrain(pretrain_set,
                                                       target=ground_truth,
                                                       original=original)
      initial_constants[key] = initial.constant
      initial_scales[key] = initial.scale
    else:
      initial_constants[key] = 0
      initial_scales[key] = 1

  # Train potential shaping and other parameters
  metrics = comparisons.fit_models(matchings, training_generator)

  # Evaluation
  intrinsics = {}
  shapings = {}
  extrinsics = {}
  ub_intrinsic = rewards.evaluate_models({"n": noise_reward}, test_set)["n"]
  ub_intrinsic = np.linalg.norm(ub_intrinsic) / np.sqrt(len(ub_intrinsic))
  ub_intrinsics = {}
  final_constants = {}
  final_scales = {}
  # TODO(): this is a sequential bottleneck
  for rew_nm in reward_noise:
    for pot_nm in potential_noise:
      original = originals[(rew_nm, pot_nm)]
      matched = matchings[(rew_nm, pot_nm)]
      shaping_model = None
      if model_potential:
        shaping_model = matched.model_extra["shaping"].models["shaping"][0]

      res = comparisons.summary_comparison(original=original,
                                           matched=matched.model,
                                           target=ground_truth,
                                           shaping=shaping_model,
                                           test_set=test_set)
      intrinsic, shaping, extrinsic = res
      intrinsics[(rew_nm, pot_nm)] = intrinsic
      shapings[(rew_nm, pot_nm)] = shaping
      extrinsics[(rew_nm, pot_nm)] = extrinsic
      ub_intrinsics[(rew_nm, pot_nm)] = rew_nm * ub_intrinsic

      if model_affine:
        final = matched.model_extra["affine"].get_weights()
      else:
        final = rewards.AffineParameters(constant=0, scale=1.0)
      final_constants[(rew_nm, pot_nm)] = final.constant
      final_scales[(rew_nm, pot_nm)] = final.scale

  res = {
      "Intrinsic": intrinsics,
      "Intrinsic Upper Bound": ub_intrinsics,
      "Shaping": shapings,
      "Extrinsic": extrinsics,
      # Report scale from the perspective of the transformation needed to
      # map the generated reward model back to the target. So we need to
      # invert the gt_scale and gt_constant parameters, but can report the
      # parameters from the AffineTransform verbatim.
      "Real Scale": 1 / gt_scale,
      "Real Constant": -gt_constant / gt_scale,
      "Initial Scale": initial_scales,
      "Initial Constant": initial_constants,
      "Inferred Scale": final_scales,
      "Inferred Constant": final_constants,
  }
  df = pd.DataFrame(res)
  df.index.names = ["Reward Noise", "Potential Noise"]
  return df, metrics


def summary_stats(observation_space: gym.Space,
                  action_space: gym.Space,
                  dataset: rewards.Batch,
                  reward_hids: Optional[Iterable[int]] = None,
                  potential_hids: Optional[Iterable[int]] = None):
  """Compute summary statistics of a random reward and potential model."""
  # Construct randomly initialized reward and potential
  rew_model = rewards.MLPRewardModel(observation_space, action_space,
                                     reward_hids)
  pot_model = rewards.PotentialShaping(observation_space, action_space,
                                       potential_hids)
  tf.get_default_session().run(tf.global_variables_initializer())

  # Compute their predictions on dataset
  models = {"reward": rew_model, "shaping": pot_model}
  preds = rewards.evaluate_models(models, dataset)
  potentials = rewards.evaluate_potentials([pot_model], dataset)
  old_potential = potentials[0][0]
  new_potential = potentials[1][0]

  # Compute summary statistics
  res = dict(**preds,
             old_potential=old_potential,
             new_potential=new_potential)
  return {k: sp.stats.describe(v) for k, v in res.items()}
