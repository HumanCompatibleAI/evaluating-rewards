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

"""Experiments with synthetic randomly generated reward models.

See Colab notebook for use cases.
"""

import functools
import logging
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple, Type

import gym
from imitation.data import types
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import tensorflow as tf

from evaluating_rewards import datasets
from evaluating_rewards.distances import npec
from evaluating_rewards.rewards import base, comparisons

TensorCallable = Callable[..., tf.Tensor]

log_normal = functools.partial(np.random.lognormal, mean=0.0, sigma=np.log(10))


def _compare_synthetic_build_base_models(
    observation_space: gym.Space,
    action_space: gym.Space,
    reward_hids: Optional[Iterable[int]],
    dataset_potential_hids: Optional[Iterable[int]],
    dataset_activation: Optional[TensorCallable],
    state_only: bool,
    discount: float,
):
    # Graph construction
    noise_kwargs = {}
    if state_only:
        noise_kwargs = {"use_act": False, "use_next_obs": False}

    with tf.variable_scope("ground_truth"):
        ground_truth = base.MLPRewardModel(
            observation_space, action_space, hid_sizes=reward_hids, **noise_kwargs
        )

    with tf.variable_scope("noise"):
        noise_reward = base.MLPRewardModel(
            observation_space, action_space, hid_sizes=reward_hids, **noise_kwargs
        )
        noise_potential = base.MLPPotentialShaping(
            observation_space,
            action_space,
            hid_sizes=dataset_potential_hids,
            activation=dataset_activation,
            discount=discount,
        )

        # Additive constant and scaling of ground truth
        initializer = tf.initializers.ones
        constant_one_model = base.ConstantReward(
            observation_space, action_space, initializer=initializer
        )

    return ground_truth, noise_reward, noise_potential, constant_one_model


def _compare_synthetic_build_comparison_graph(
    ground_truth: base.RewardModel,
    noise_reward: base.RewardModel,
    noise_potential: base.RewardModel,
    constant_one_model: base.RewardModel,
    model_affine: bool,
    model_potential: bool,
    discount: float,
    reward_noise: np.ndarray,
    potential_noise: np.ndarray,
    gt_constant: float,
    gt_scale: float,
    model_potential_hids: Optional[Iterable[int]],
    model_activation: Optional[TensorCallable],
    optimizer: Type[tf.train.Optimizer],
    learning_rate: float,
):
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

                noised_ground_shaped = base.LinearCombinationModelWrapper(models)
                originals[(rew_nm, pot_nm)] = noised_ground_shaped

                with tf.variable_scope("matching"):
                    model_wrapper = functools.partial(
                        npec.equivalence_model_wrapper,
                        affine=model_affine,
                        potential=model_potential,
                        hid_sizes=model_potential_hids,
                        activation=model_activation,
                        discount=discount,
                    )
                    matched = npec.RegressWrappedModel(
                        noised_ground_shaped,
                        ground_truth,
                        model_wrapper=model_wrapper,
                        learning_rate=learning_rate,
                        optimizer=optimizer,
                    )
                    matchings[(rew_nm, pot_nm)] = matched

    return originals, matchings


def _compare_synthetic_eval(
    metrics: Mapping[str, List[Mapping[Tuple[float, float], Any]]],
    originals,
    matchings,
    test_set: types.Transitions,
    initial_constants: Mapping[Tuple[float, float], float],
    initial_scales: Mapping[Tuple[float, float], float],
    gt_constant: float,
    gt_scale: float,
    model_affine: bool,
    model_potential: bool,
    ground_truth: base.RewardModel,
    noise_reward: base.RewardModel,
    reward_noise: np.ndarray,
    potential_noise: np.ndarray,
):
    intrinsics = {}
    shapings = {}
    extrinsics = {}
    ub_intrinsic = base.evaluate_models({"n": noise_reward}, test_set)["n"]
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

            res = summary_comparison(
                original=original,
                matched=matched.model,
                target=ground_truth,
                shaping=shaping_model,
                test_set=test_set,
            )
            intrinsic, shaping, extrinsic = res
            intrinsics[(rew_nm, pot_nm)] = intrinsic
            shapings[(rew_nm, pot_nm)] = shaping
            extrinsics[(rew_nm, pot_nm)] = extrinsic
            ub_intrinsics[(rew_nm, pot_nm)] = rew_nm * ub_intrinsic

            if model_affine:
                final = matched.model_extra["affine"].get_weights()
            else:
                final = base.AffineParameters(shift=0, scale=1.0)
            final_constants[(rew_nm, pot_nm)] = final.shift
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


def compare_synthetic(
    observation_space: gym.Space,
    action_space: gym.Space,
    dataset_generator: datasets.TransitionsCallable,
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
    optimizer: Type[tf.train.Optimizer] = tf.train.AdamOptimizer,
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
    if reward_noise is None:
        reward_noise = np.arange(0.0, 1.0, 0.2)
    if potential_noise is None:
        potential_noise = np.arange(0.0, 10.0, 2.0)

    models = _compare_synthetic_build_base_models(
        observation_space,
        action_space,
        reward_hids=reward_hids,
        dataset_potential_hids=dataset_potential_hids,
        dataset_activation=dataset_activation,
        state_only=state_only,
        discount=discount,
    )
    ground_truth, noise_reward, noise_potential, constant_one_model = models

    gt_scale = scale_fn()
    gt_constant = constant_fn(gt_scale)
    originals, matchings = _compare_synthetic_build_comparison_graph(
        ground_truth=ground_truth,
        noise_reward=noise_reward,
        noise_potential=noise_potential,
        constant_one_model=constant_one_model,
        model_affine=model_affine,
        model_potential=model_potential,
        discount=discount,
        reward_noise=reward_noise,
        potential_noise=potential_noise,
        gt_constant=gt_constant,
        gt_scale=gt_scale,
        model_potential_hids=model_potential_hids,
        model_activation=model_activation,
        optimizer=optimizer,
        learning_rate=learning_rate,
    )

    # Initialization
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())

    # Datasets
    test_set = dataset_generator(test_size)

    # Pre-train to initialize affine parameters
    initial_constants = {}
    initial_scales = {}
    pretrain_set = dataset_generator(pretrain_size)
    for key, matched in matchings.items():
        if model_affine and pretrain:
            logging.info(f"Pretraining {key}")
            initial = matched.fit_affine(pretrain_set)
        else:
            initial = base.AffineParameters(shift=0, scale=1)
        initial_constants[key] = initial.shift
        initial_scales[key] = initial.scale

    # Train potential shaping and other parameters
    metrics = None
    if total_timesteps > 0:
        metrics = comparisons.fit_models(matchings, dataset_generator, total_timesteps, batch_size)

    return _compare_synthetic_eval(
        metrics=metrics,
        originals=originals,
        matchings=matchings,
        test_set=test_set,
        initial_constants=initial_constants,
        initial_scales=initial_scales,
        gt_constant=gt_constant,
        gt_scale=gt_scale,
        model_affine=model_affine,
        model_potential=model_potential,
        ground_truth=ground_truth,
        noise_reward=noise_reward,
        reward_noise=reward_noise,
        potential_noise=potential_noise,
    )


def summary_stats(
    observation_space: gym.Space,
    action_space: gym.Space,
    dataset: types.Transitions,
    reward_hids: Optional[Iterable[int]] = None,
    potential_hids: Optional[Iterable[int]] = None,
):
    """Compute summary statistics of a random reward and potential model."""
    # Construct randomly initialized reward and potential
    rew_model = base.MLPRewardModel(observation_space, action_space, reward_hids)
    pot_model = base.MLPPotentialShaping(observation_space, action_space, potential_hids)
    tf.get_default_session().run(tf.global_variables_initializer())

    # Compute their predictions on dataset
    models = {"reward": rew_model, "shaping": pot_model}
    preds = base.evaluate_models(models, dataset)
    potentials = base.evaluate_potentials([pot_model], dataset)
    old_potential = potentials[0][0]
    new_potential = potentials[1][0]

    # Compute summary statistics
    res = dict(**preds, old_potential=old_potential, new_potential=new_potential)
    return {k: sp.stats.describe(v) for k, v in res.items()}


def plot_shaping_comparison(
    df: pd.DataFrame, cols: Optional[Iterable[str]] = None, **kwargs
) -> pd.DataFrame:
    """Plots return value of experiments.compare_synthetic."""
    if cols is None:
        cols = ["Intrinsic", "Shaping"]
    df = df.loc[:, cols]
    longform = df.reset_index()
    longform = pd.melt(
        longform,
        id_vars=["Reward Noise", "Potential Noise"],
        var_name="Metric",
        value_name="Distance",
    )
    sns.lineplot(
        x="Reward Noise",
        y="Distance",
        hue="Potential Noise",
        style="Metric",
        data=longform,
        **kwargs,
    )
    return longform


def _scaled_norm(x):
    """l2 norm, normalized to be invariant to length of vectors."""
    return np.linalg.norm(x) / np.sqrt(len(x))


def summary_comparison(
    original: base.RewardModel,
    matched: base.RewardModel,
    target: base.RewardModel,
    test_set: types.Transitions,
    shaping: Optional[base.RewardModel] = None,
) -> Tuple[float, float, float]:
    """Compare rewards in terms of intrinsic and shaping difference.

    Args:
        original: The inferred reward model.
        matched: The reward model after trying to match target via shaping.
        target: The target reward model (e.g. ground truth, if available).
        test_set: A dataset to evaluate on.
        shaping: A reward model adding potential shaping to original.
                If unspecified, will return 0 for the shaping component.

    Returns:
        A tuple (intrinsic, shaping, extrinsic). The intrinsic difference is the
        approximation of the nearest point between the equivalence classes for
        original and target. Shaping is the magnitude of the potential shaping
        term we are adding. Extrinsic is the raw difference between original and
        target without any transformations.
    """
    models = {"original": original, "matched": matched, "target": target}

    if shaping is not None:
        models["shaping"] = shaping

    preds = base.evaluate_models(models, test_set)
    intrinsic_l2 = _scaled_norm(preds["matched"] - preds["target"])
    if "shaping" in preds:
        shaping_l2 = _scaled_norm(preds["shaping"])
    else:
        shaping_l2 = 0.0
    extrinsic_l2 = _scaled_norm(preds["original"] - preds["target"])

    return intrinsic_l2, shaping_l2, extrinsic_l2
