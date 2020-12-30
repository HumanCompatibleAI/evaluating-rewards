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

"""Command-line script to train RL algorithms on (learned) rewards and compute mean return.

Baseline "distance" between reward functions.
"""

import logging
import os
import pickle
from typing import Any, Iterable, Mapping

from imitation.data import types
from imitation.util import util as imit_util
import numpy as np
import ray
import sacred

from evaluating_rewards.distances import common_config
from evaluating_rewards.rewards import base
from evaluating_rewards.scripts import rl_common, script_utils
from evaluating_rewards.scripts.distances import common

rollout_distance_ex = sacred.Experiment("rollout_distance")
logger = logging.getLogger("evaluating_rewards.scripts.distances.rollout_return")

common.make_config(rollout_distance_ex)
rl_common.make_config(rollout_distance_ex)


@rollout_distance_ex.config
def default_config():
    """Default configuration values."""
    discount = 0.99  # discount rate when calculating return
    _ = locals()
    del _


@rollout_distance_ex.config
def logging_config(env_name, discount, log_root):
    """Default logging configuration: hierarchical directory structure based on config."""
    log_dir = os.path.join(  # noqa: F841  pylint:disable=unused-variable
        log_root,
        "rollout_distance",
        script_utils.sanitize_path(env_name),
        f"discount{discount}",
        imit_util.make_unique_timestamp(),
    )


@rollout_distance_ex.named_config
def high_precision():
    n_bootstrap = 10000  # noqa: F841  pylint:disable=unused-variable


@rollout_distance_ex.named_config
def test():
    locals().update(**rl_common.FAST_CONFIG)


@rollout_distance_ex.named_config
def point_maze_learned_multi_seed_airl_sa():
    y_reward_cfgs = [  # noqa: F841  pylint:disable=unused-variable
        (
            "imitation/RewardNet_unshaped-v0",
            f"transfer_point_maze.multiseed/reward/irl_state_action.seed{seed}/"
            "checkpoints/final/discrim/reward_net",
        )
        # seed 3 does very poorly in monitor_return_mean
        # seed 2 we picked and already evaluated
        for seed in (0, 1, 4)
    ]


@rollout_distance_ex.capture
def do_training(
    ray_kwargs: Mapping[str, Any],
    num_cpus_fudge_factor: float,
    global_configs: Mapping[str, Any],
    log_dir: str,
    env_name: str,
    y_reward_cfgs: Iterable[common_config.RewardCfg],
) -> rl_common.Stats:
    """For each reward in `y_reward_cfgs`, train policies and collect rollouts.

    Args:
        ray_kwargs: Passed through to `ray.init`.
        num_cpus_fudge_factor: Factor by which to scale `num_vec` to compute CPU requirements.
        global_configs: Configuration to apply to all environment-reward pairs.
        log_dir: Directory to save data to.
        env_name: The name of the environment to compare rewards for.
        y_reward_cfgs: Tuples of reward_type and reward_path for y-axis.

    Returns:
        Training statistics, including log directories.
    """
    configs = {}
    for reward_type, reward_path in y_reward_cfgs:
        configs.setdefault(reward_type, {})[reward_path] = dict(rl_common.CONFIG_BY_ENV[env_name])
    configs = {env_name: configs}

    ray.init(**ray_kwargs)
    try:
        return rl_common.parallel_training(
            global_configs=global_configs,
            configs=configs,
            num_cpus_fudge_factor=num_cpus_fudge_factor,
            log_dir=log_dir,
        )
    finally:
        ray.shutdown()


@rollout_distance_ex.capture
def compute_returns(
    stats: rl_common.Stats,
    env_name: str,
    discount: float,
    x_reward_cfgs: Iterable[common_config.RewardCfg],
    y_reward_cfgs: Iterable[common_config.RewardCfg],
) -> common.Dissimilarities:
    """For each rollout, compute returns under each reward in x_reward_cfgs.

    Args:
        stats: Training statistics, including log directory containing rollouts.
        env_name: The name of the environment to compare rewards for.
        discount: Discount to use for reward models (mostly for shaping).
        x_reward_cfgs: Tuples of reward_type and reward_path for x-axis.
        y_reward_cfgs: Tuples of reward_type and reward_path for y-axis.

    Returns:
        Return of rollouts for each (x,y) reward configuration pair for each seed.
    """
    models, _g, sess = common.load_models_create_sess(env_name, discount, x_reward_cfgs)
    mean_returns = {}
    for y_cfg in y_reward_cfgs:
        for _metrics, run_dir in stats[(env_name, y_cfg)]:
            rollout_path = os.path.join(run_dir, "rollouts", "final.pkl")
            trajs = types.load(rollout_path)
            with sess.as_default():
                returns = base.compute_return_of_models(models, trajs, discount)
            for x_cfg, ret in returns.items():
                mean_ret = float(np.mean(ret))
                mean_returns.setdefault((x_cfg, y_cfg), []).append(mean_ret)
    return mean_returns


@rollout_distance_ex.capture
def compute_vals(
    x_reward_cfgs: Iterable[common_config.RewardCfg],
    y_reward_cfgs: Iterable[common_config.RewardCfg],
    aggregate_fns: Mapping[str, common.AggregateFn],
    log_dir: str,
) -> common_config.AggregatedDistanceReturn:
    """Computes mean returns for policies trained on `y_reward_cfgs` evaluated on `x_reward_cfgs`.

    Args:
        x_reward_cfgs: Tuples of reward_type and reward_path for x-axis.
        y_reward_cfgs: Tuples of reward_type and reward_path for y-axis.
        aggregate_fns: Mapping from strings to aggregators to be applied on sequences of floats.
        log_dir: Directory to save data to.

    Returns:
        Nested dictionary of aggregated return values.
    """
    stats = do_training(y_reward_cfgs=y_reward_cfgs)  # pylint:disable=no-value-for-parameter
    dissimilarities = compute_returns(  # pylint:disable=no-value-for-parameter
        stats, x_reward_cfgs=x_reward_cfgs, y_reward_cfgs=y_reward_cfgs
    )
    logger.info("Saving raw dissimilarities")
    with open(os.path.join(log_dir, "dissimilarities.pkl"), "wb") as f:
        pickle.dump(dissimilarities, f)

    # Lower return = higher distance, higher return = lower distance.
    # So flip the signs for CI calculation, then flip it back.
    dissimilarities = {k: [-x for x in v] for k, v in dissimilarities.items()}
    aggregated = common.aggregate_seeds(aggregate_fns, dissimilarities)
    aggregated = dict(aggregated)
    keys = aggregated.keys()
    for k in keys:
        if not (k.endswith("relative") or k.endswith("width")):
            aggregated[k] = {k2: -v2 for k2, v2 in aggregated[k].items()}
    return aggregated


common.make_main(rollout_distance_ex, compute_vals)

if __name__ == "__main__":
    script_utils.experiment_main(rollout_distance_ex, "rollout_distance")
