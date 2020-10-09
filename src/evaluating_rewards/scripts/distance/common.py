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

"""Configurations for dissimilarity_heatmaps heatmaps.

Shared between `evaluating_rewards.analysis.{plot_epic_heatmap,plot_canon_heatmap}`.
"""

import functools
import logging
import os
import pickle
from typing import Any, Callable, Iterable, Mapping, Sequence, Tuple

import gym
import numpy as np
import sacred
import scipy.stats
from stable_baselines.common import vec_env
import tensorflow as tf

from evaluating_rewards import serialize
from evaluating_rewards.analysis import util
from evaluating_rewards.rewards import base

AggregateFn = Callable[[Sequence[float]], Mapping[str, float]]
RewardCfg = Tuple[str, str]  # (type, path)
AggregatedDistanceReturn = Mapping[str, Mapping[Tuple[RewardCfg, RewardCfg], float]]

logger = logging.getLogger("evaluating_rewards.scripts.distance.common")


def _config_from_kinds(kinds: Iterable[str], **kwargs) -> Mapping[str, Any]:
    cfgs = [(kind, "dummy") for kind in kinds]
    res = dict(kwargs)
    res.update({"x_reward_cfgs": cfgs, "y_reward_cfgs": cfgs})
    return res


POINT_MASS_KINDS = [
    f"evaluating_rewards/PointMass{label}-v0"
    for label in ["SparseNoCtrl", "SparseWithCtrl", "DenseNoCtrl", "DenseWithCtrl", "GroundTruth"]
]
POINT_MAZE_KINDS = [
    "imitation/PointMazeGroundTruthWithCtrl-v0",
    "imitation/PointMazeGroundTruthNoCtrl-v0",
]
MUJOCO_STANDARD_ORDER = [
    "ForwardNoCtrl",
    "ForwardWithCtrl",
    "BackwardNoCtrl",
    "BackwardWithCtrl",
]

COMMON_CONFIGS = {
    # evaluating_rewards/PointMass* environments.
    "point_mass": _config_from_kinds(
        POINT_MASS_KINDS, env_name="evaluating_rewards/PointMassLine-v0"
    ),
    # imitation/PointMaze{Left,Right}-v0 environments
    "point_maze": _config_from_kinds(
        POINT_MAZE_KINDS,
        env_name="imitation/PointMazeLeft-v0",
    ),
    # Compare rewards learned in imitation/PointMaze* to the ground-truth reward
    "point_maze_learned": {
        "env_name": "imitation/PointMazeLeftVel-v0",
        "x_reward_cfgs": [("evaluating_rewards/PointMazeGroundTruthWithCtrl-v0", "dummy")],
        "y_reward_cfgs": [
            ("evaluating_rewards/RewardModel-v0", "transfer_point_maze/reward/regress/model"),
            ("evaluating_rewards/RewardModel-v0", "transfer_point_maze/reward/preferences/model"),
            (
                "imitation/RewardNet_unshaped-v0",
                "transfer_point_maze/reward/irl_state_only/checkpoints/final/discrim/reward_net",
            ),
            (
                "imitation/RewardNet_unshaped-v0",
                "transfer_point_maze/reward/irl_state_action/checkpoints/final/discrim/reward_net",
            ),
        ],
    },
    # seals version of the canonical MuJoCo tasks
    "half_cheetah": _config_from_kinds(
        [
            f"evaluating_rewards/HalfCheetahGroundTruth{suffix}-v0"
            for suffix in MUJOCO_STANDARD_ORDER
        ],
        env_name="seals/HalfCheetah-v0",
    ),
    "hopper": _config_from_kinds(
        kinds=[
            f"evaluating_rewards/Hopper{prefix}{suffix}-v0"
            for prefix in ["GroundTruth", "Backflip"]
            for suffix in MUJOCO_STANDARD_ORDER
        ],
        env_name="seals/Hopper-v0",
    ),
}


def canonicalize_reward_cfg(reward_cfg: RewardCfg, data_root: str) -> RewardCfg:
    """Canonicalize path in reward configuration.

    Specifically, join paths with the `data_root`, unless they are the special "dummy" path.
    Also ensure the return value is actually of type RewardCfg: it is forgiving and will accept
    any iterable pair as input `reward_cfg`. This is important since Sacred has the bad habit of
    converting tuples to lists in configurations.

    Args:
        reward_cfg: Iterable of configurations to canonicailze.
        data_root: The root to join paths to.

    Returns:
        Canonicalized RewardCfg.
    """
    kind, path = reward_cfg
    if path != "dummy":
        path = os.path.join(data_root, path)
    return (kind, path)


def load_models(
    env_name: str,
    reward_cfgs: Iterable[RewardCfg],
    discount: float,
) -> Mapping[RewardCfg, base.RewardModel]:
    """Load models specified by the `reward_cfgs`.

    Args:
        - env_name: The environment name in the Gym registry of the rewards to compare.
        - reward_cfgs: Iterable of reward configurations.
        - discount: Discount to use for reward models (mostly for shaping).

    Returns:
         A mapping from reward configurations to the loaded reward model.
    """
    venv = vec_env.DummyVecEnv([lambda: gym.make(env_name)])
    return {
        (kind, path): serialize.load_reward(kind, path, venv, discount)
        for kind, path in reward_cfgs
    }


def relative_error(
    lower: float,
    mean: float,
    upper: float,
) -> float:
    """Compute relative upper bound rel, which the mean is within +/- rel % of.

    Since this may be asymmetric between lower and upper, returns the maximum of
    upper / mean - 1 (relative upper bound) and 1 - lower / mean (relative lower bound).
    """
    return max(upper / mean - 1, 1 - lower / mean)


def bootstrap_ci(vals: Iterable[float], n_bootstrap: int, alpha: float) -> Mapping[str, float]:
    """Compute `alpha` %ile confidence interval of mean of `vals` from `n_bootstrap` samples."""
    bootstrapped = util.bootstrap(np.array(vals), stat_fn=np.mean, n_samples=n_bootstrap)
    lower, middle, upper = util.empirical_ci(bootstrapped, alpha)
    return {
        "lower": lower,
        "middle": middle,
        "upper": upper,
        "width": upper - lower,
        "relative": relative_error(lower, middle, upper),
    }


def studentt_ci(vals: Sequence[float], alpha: float) -> Mapping[str, float]:
    """Compute `alpha` %ile confidence interval of mean of `vals` using t-distribution."""
    assert len(vals) > 1
    df = len(vals) - 1
    mu = np.mean(vals)
    stderr = scipy.stats.sem(vals)
    lower, upper = scipy.stats.t.interval(alpha / 100, df, loc=mu, scale=stderr)
    return {
        "lower": lower,
        "middle": mu,
        "upper": upper,
        "width": upper - lower,
        "relative": relative_error(lower, mu, upper),
    }


def sample_mean_sd(vals: Sequence[float]) -> Mapping[str, float]:
    """Returns sample mean and (unbiased) standard deviation."""
    assert len(vals) > 1
    return {"mean": np.mean(vals), "sd": np.std(vals, ddof=1)}


def make_config(
    experiment: sacred.Experiment,
):  # pylint: disable=unused-variable,too-many-statements
    """Adds configs and named configs to `experiment`.

    The standard config parameters it defines are:
        - vals_path (Optional[str]): path to precomputed values to plot.
        - env_name (str): The environment name in the Gym registry of the rewards to compare.
        - x_reward_cfgs (Iterable[RewardCfg]): tuples of reward_type and reward_path for x-axis.
        - y_reward_cfgs (Iterable[RewardCfg]): tuples of reward_type and reward_path for y-axis.
        - log_root (str): the root directory to log; subdirectory path automatically constructed.
        - n_bootstrap (int): the number of bootstrap samples to take.
        - alpha (float): percentile confidence interval
        - aggregate_kinds (Iterable[str]): the type of aggregations to perform across seeds.
            Not used in `plot_return_heatmap` which only supports its own kind of bootstrapping.
        - heatmap_kwargs (dict): passed through to `analysis.compact_heatmaps`.
        - styles (Iterable[str]): styles to apply from `evaluating_rewards.analysis.stylesheets`.
        - save_kwargs (dict): passed through to `analysis.save_figs`.
    """

    @experiment.config
    def default_config():
        """Default configuration values."""
        env_name = "evaluating_rewards/PointMassLine-v0"
        data_root = serialize.get_output_dir()  # where models are read from
        log_root = serialize.get_output_dir()  # where results are written to
        n_bootstrap = 1000  # number of bootstrap samples
        alpha = 95  # percentile confidence interval
        aggregate_kinds = ("bootstrap", "studentt", "sample")
        vals_path = None

        # Reward configurations: models to compare
        x_reward_cfgs = None
        y_reward_cfgs = None

        _ = locals()
        del _

    @experiment.config
    def aggregate_fns(aggregate_kinds, n_bootstrap, alpha):
        """Make a mapping of aggregate functions of kinds `subset` with specified parameters.

        Used in plot_{canon,epic}_heatmap; currently ignored by plot_return_heatmap since
        it does not use multiple seeds and instead bootstraps over samples.
        """
        aggregate_fns = {}
        if "bootstrap" in aggregate_kinds:
            aggregate_fns["bootstrap"] = functools.partial(
                bootstrap_ci, n_bootstrap=n_bootstrap, alpha=alpha
            )
        if "studentt" in aggregate_kinds:
            aggregate_fns["studentt"] = functools.partial(studentt_ci, alpha=alpha)
        if "sample" in aggregate_kinds:
            aggregate_fns["sample"] = sample_mean_sd

    for name, cfg in COMMON_CONFIGS.items():
        experiment.add_named_config(name, cfg)


def make_main(
    experiment: sacred.Experiment, compute_vals: Callable[..., AggregatedDistanceReturn]
):  # pylint: disable=unused-variable
    """Helper to make main function for distance scripts.

    Specifically, register a main function with `experiment` that:
      - Creates log directory.
      - Canonicalizes reward configurations.
      - Loads reward models.
      - Calls `compute_vals`.
      - Saves the return value of `compute_vals` in the log directory.

    Args:
        experiment: the Sacred experiment to register a main function with.
        compute_vals: a function to call to compute the distances.
            It is always passed a graph `g`, session `sess`, loaded models `models`, and
            canonicalized reward configurations `x_reward_cfgs` and `y_reward_cfgs`.
            It is typically defined with `@experiment.capture` in which case Sacred will
            fill in other parameters defined in the configuration.
      -"""

    @experiment.main
    def main(
        env_name: str,
        discount: float,
        x_reward_cfgs: Iterable[RewardCfg],
        y_reward_cfgs: Iterable[RewardCfg],
        data_root: str,
        log_dir: str,
    ) -> AggregatedDistanceReturn:
        """Wrapper around `compute_vals` performing common setup and saving logic.

        Args:
            env_name: the name of the environment to compare rewards for.
            x_reward_cfgs: tuples of reward_type and reward_path for x-axis.
            y_reward_cfgs: tuples of reward_type and reward_path for y-axis.
            data_root: directory to load learned reward models from.
            log_dir: directory to save data to.

        Returns:
            The values returned by `compute_vals`.
        """
        # Sacred turns our tuples into lists :(, undo
        x_reward_cfgs = [canonicalize_reward_cfg(cfg, data_root) for cfg in x_reward_cfgs]
        y_reward_cfgs = [canonicalize_reward_cfg(cfg, data_root) for cfg in y_reward_cfgs]

        logger.info("Loading models")
        g = tf.Graph()
        with g.as_default():
            sess = tf.Session()
            with sess.as_default():
                reward_cfgs = x_reward_cfgs + y_reward_cfgs
                models = load_models(env_name, reward_cfgs, discount)

        # If `compute_vals` is a capture function, then Sacred will fill in other parameters
        aggregated = compute_vals(
            g=g, sess=sess, models=models, x_reward_cfgs=x_reward_cfgs, y_reward_cfgs=y_reward_cfgs
        )

        aggregated_path = os.path.join(log_dir, "aggregated.pkl")
        logger.info(f"Saving aggregated values to {aggregated_path}")
        with open(aggregated_path, "wb") as f:
            pickle.dump(aggregated, f)

        return aggregated
