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

"""Unit tests for evaluating_rewards.synthetic.

Also indirectly tests evaluating_rewards.deep, evaluating_rewards.datasets and
evaluating_rewards.util.
"""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

from evaluating_rewards.experiments import datasets
from evaluating_rewards.experiments import synthetic
from tests import common
import numpy as np
import pandas as pd

ENVIRONMENTS = {
    "Uniform5D": datasets.dummy_env_and_dataset(dims=5),
    "PointLine": datasets.make_pm("evaluating_rewards/PointMassLine-v0"),
    "PointGrid": datasets.make_pm("evaluating_rewards/PointMassGrid-v0"),
}

ARCHITECTURES = {
    "Linear": {
        "reward_hids": [],
        "dataset_potential_hids": [],
        "model_potential_hids": [],
        "learning_rate": 1e-2,
        "total_timesteps": 2 ** 18,
        "batch_size": 256,
        "rel_upperbound": 0.1,
    },
    "OneLayer": {
        "reward_hids": [32],
        "dataset_potential_hids": [4],
        "model_potential_hids": [32],
        "learning_rate": 1e-2,
        "total_timesteps": 2 ** 18,
        "batch_size": 512,
        "rel_upperbound": 0.2,
    },
    "TwoLayer": {
        "reward_hids": [32, 32],
        "dataset_potential_hids": [4, 4],
        "model_potential_hids": [32, 32],
        "learning_rate": 1e-2,
        "total_timesteps": 2 ** 18,
        "batch_size": 512,
        "rel_upperbound": 0.2,
    }
}

EQUIV_SCALES = {
    "identity": {},  # reward functions are identical
    "random": {  # reward functions are affine transformations of each other
        "scale_fn": synthetic.log_normal,
        "constant_fn": lambda scale: scale * np.random.normal(),
    }
}

NOISY_AFFINE_ENVIRONMENTS = {
    # It generally does much better on Point{Line,Grid}, which has a much
    # smaller scale of potential noise than Uniform5D. So set higher upper bound
    # for Uniform5D than for Point*.
    "Uniform5D": dict(**ENVIRONMENTS["Uniform5D"],
                      upperbound=2.0),
    "PointLine": dict(**ENVIRONMENTS["PointLine"],
                      upperbound=0.025),
    "PointGrid": dict(**ENVIRONMENTS["PointGrid"],
                      upperbound=0.025),
}


def const_functor(x):
  def f(*args):
    del args
    return x
  return f


AFFINE_TRANSFORMS = {
    "random": {
        "scale_fn": synthetic.log_normal,
        "constant_fn": lambda scale: scale * np.random.normal(),
    },
    "identity": {}
}
for scale, sign in zip([1e-3, 1e-1, 1e1, 1e3], [1, -1, 1, -1]):
  AFFINE_TRANSFORMS[f"{scale}_{sign}"] = {
      "scale_fn": const_functor(scale),
      "constant_fn": const_functor(scale / 2 * sign),
  }


SYNTHETIC_TEST = {
    "same_scale": {
        "rescale": False,
        "fudge_factor": 1.0,  # follow upper bound in ARCHITECTURES
    },
    "random_scale": {
        "scale_fn": synthetic.log_normal,
        "constant_fn": lambda scale: scale * np.random.normal(),
        "rescale": True,
        "fudge_factor": 2.0,  # be twice as lenient
    },
}


class ExperimentTest(common.TensorFlowTestCase):
  """Sanity checks results from miscellaneous experiments."""

  @parameterized.named_parameters(common.combine_dicts_as_kwargs(
      ENVIRONMENTS, ARCHITECTURES, EQUIV_SCALES
  ))
  def test_identical(self, **kwargs):
    """Try to minimize the difference between two identical reward models."""
    del kwargs["rel_upperbound"]
    with self.graph.as_default():
      with self.sess.as_default():
        noise = np.array([0.0])
        _, metrics = synthetic.compare_synthetic(reward_noise=noise,
                                                 potential_noise=noise,
                                                 **kwargs)
        loss = pd.DataFrame(metrics["loss"])
        loss = loss[(0.0, 0.0)]
        initial_loss = loss.iloc[0]
        final_loss = loss.iloc[-1]
        self.assertLess(final_loss, 1e-4)
        self.assertGreater(initial_loss / final_loss, 1e2)

  def _test_affine(self, upperbound, **kwargs):
    """Do we recover affine parameters correctly?"""
    with self.graph.as_default():
      with self.sess.as_default():
        df, _ = synthetic.compare_synthetic(reward_noise=np.array([0.0]),
                                            model_affine=True,
                                            pretrain=True,
                                            pretrain_size=4096,
                                            **kwargs)
        rel_error_scale = ((df["Inferred Scale"] - df["Real Scale"]) /
                           df["Real Scale"])
        # The constant parameter is in the same scale as the target
        # (which should be consistent across test configurations),
        # so no need to normalize.
        abs_error_constant = df["Inferred Constant"] - df["Real Constant"]

        with pd.option_context("display.max_rows", None,
                               "display.max_columns", None):
          logging.info("Comparison: %s", df)
          logging.info("Relative error scale: %s", rel_error_scale)
          logging.info("Absolute error constant: %s", abs_error_constant)

        self.assertLess(rel_error_scale.abs().max(), upperbound)
        self.assertLess(abs_error_constant.abs().max(), upperbound)

  @parameterized.named_parameters(common.combine_dicts_as_kwargs(
      ENVIRONMENTS, AFFINE_TRANSFORMS,
  ))
  def test_clean_affine(self, **kwargs):
    """Can we get a good initialization when there is no noise?"""
    return self._test_affine(total_timesteps=0,
                             potential_noise=np.array([0.0]),
                             model_potential=False,
                             upperbound=1e-3,
                             **kwargs)

  @parameterized.named_parameters(common.combine_dicts_as_kwargs(
      NOISY_AFFINE_ENVIRONMENTS, AFFINE_TRANSFORMS,
  ))
  def test_pretrain_affine(self, **kwargs):
    """Can we recover good affine parameters in presence of potential noise?"""
    return self._test_affine(reward_hids=[32, 32],
                             dataset_potential_hids=[4, 4],
                             model_potential=True,
                             model_potential_hids=[32, 32],
                             total_timesteps=2**18,
                             learning_rate=1e-2,
                             potential_noise=np.array([0.0, 1.0]),
                             **kwargs)

  @parameterized.named_parameters(common.combine_dicts_as_kwargs(
      ENVIRONMENTS, ARCHITECTURES, SYNTHETIC_TEST,
  ))
  def test_compare_synthetic(self, rel_upperbound, fudge_factor,
                             rescale, **kwargs):
    """Try comparing randomly generated reward models, same scale."""
    with self.graph.as_default():
      with self.sess.as_default():
        noise = np.array([0.0, 0.5, 1.0])  # coarse-grained for speed
        df, _ = synthetic.compare_synthetic(reward_noise=noise,
                                            potential_noise=noise,
                                            model_affine=rescale,
                                            **kwargs)
        with pd.option_context("display.max_rows", None,
                               "display.max_columns", None):
          logging.info("Results: %s", df)

        for k in ["Intrinsic", "Shaping", "Extrinsic"]:
          self.assertTrue((df[k] >= 0).all(axis=None),
                          f"distances {k} should not be negative")

        # No reward noise, but potential noise
        no_rew_noise = df.loc[(0.0, slice(0.1, None)), :]
        rel = no_rew_noise["Intrinsic"] / no_rew_noise["Extrinsic"]
        self.assertLess(rel.max(axis=None),
                        rel_upperbound * fudge_factor)

        if not rescale:
          # When ground truth and noised reward are on the same scale,
          # shaping distance should increase proportionally with potential
          # magnitude. When reward-noise is non-zero there's a confounder as
          # the shaping noise we add can *cancel* with shaping in the reward
          # noise. So just consider zero reward noise.
          deltas = no_rew_noise["Shaping"].diff().dropna()  # first row is N/A
          self.assertGreater(deltas.min(axis=None), 0.0)
          mean_delta = deltas.mean()
          # Increment should be similar: allow it to vary by 2x up & down
          self.assertTrue((deltas < mean_delta * 2).all(axis=None))
          self.assertTrue((deltas > mean_delta * 0.5).all(axis=None))

        # We're no more than 10% of intrinsic upper bound at any point.
        # The upper bound is based on the magnitude of the reward noise
        # we added. It's an upper bound since it may include some potential
        # shaping, so we actually could find a shorter intrinsic distance.
        # Add 10% margin of error since we don't expect perfect optimization.
        some_noise = df.loc[df.index.get_level_values("Reward Noise") > 0.0]
        rel = some_noise["Intrinsic"] / some_noise["Intrinsic Upper Bound"]
        self.assertLess(rel.max(axis=None), 1 + 0.1 * fudge_factor)


if __name__ == "__main__":
  absltest.main()
