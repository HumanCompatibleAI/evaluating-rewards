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

"""Command-line script to train expert policies.

Picks best seed of train_rl for each (environment, reward) pair specified.
"""

import math
import os
from typing import Any, Mapping, Optional

from imitation.util import util
import numpy as np
import ray
import sacred
import tabulate

from evaluating_rewards import serialize
from evaluating_rewards.distances import common_config
from evaluating_rewards.experiments import env_rewards
from evaluating_rewards.scripts import script_utils
from evaluating_rewards.scripts.rl import rl_common

experts_ex = sacred.Experiment("train_experts")
rl_common.make_config(experts_ex)


@experts_ex.config
def default_config():
    """Default configuration."""
    log_root = serialize.get_output_dir()  # where results are written to
    configs = {}
    run_tag = "default"
    _ = locals()
    del _


@experts_ex.config
def default_env_rewards(configs):
    """Set default env-reward pair in `configs` entry if it is empty.

    This is needed since if we were to define it in `default_config` it would be impossible
    to delete it given how Sacred dictionary merging works.
    """
    if not configs:
        configs = {  # noqa: F401
            "evaluating_rewards/PointMassLine-v0": {
                "evaluating_rewards/PointMassGroundTruth-v0": {"dummy": {}}
            },
        }


@experts_ex.config
def logging_config(log_root, run_tag):
    """Logging configuration: timestamp plus unique UUID."""
    log_dir = os.path.join(log_root, "train_experts", run_tag, util.make_unique_timestamp())
    _ = locals()
    del _


def _make_ground_truth_configs():
    """Ground truth configs.

    Separate function to avoid polluting Sacred ConfigScope with local variables."""
    configs = {}
    for env, gt_reward in env_rewards.GROUND_TRUTH_REWARDS_BY_ENV.items():
        cfg = rl_common.CONFIG_BY_ENV.get(env, {})
        configs.setdefault(env, {}).setdefault(str(gt_reward), {})["dummy"] = cfg
    return configs


@experts_ex.named_config
def ground_truth():
    """Train RL expert on all configured environments with the ground-truth reward."""
    configs = _make_ground_truth_configs()
    run_tag = "ground_truth"
    _ = locals()
    del _


# TODO(adam): remove except for WrongTarget? this is redundant with table_combined?
@experts_ex.named_config
def point_maze_pathologicals():
    """Train RL policies on the "wrong" rewards in PointMaze."""
    configs = {
        env: {
            reward: {"dummy": dict(rl_common.CONFIG_BY_ENV[env])}
            for reward in (
                # Repellent and BetterGoal we just want to report the policy return
                "evaluating_rewards/PointMazeRepellentWithCtrl-v0",
                "evaluating_rewards/PointMazeBetterGoalWithCtrl-v0",
                # We use WrongTarget expert for a visitation distribution
                "evaluating_rewards/PointMazeWrongTargetWithCtrl-v0",
            )
        }
        for env in ("imitation/PointMazeLeftVel-v0", "imitation/PointMazeRightVel-v0")
    }
    run_tag = "point_maze_pathologicals"
    _ = locals()
    del _


# TODO(adam): remove? this is redundant with distances.rollout?
def _point_maze_learned(fast_config: bool):
    """Train RL policies on learned rewards in PointMaze."""
    suffix = "_fast" if fast_config else ""
    configs = {}
    for env in ("imitation/PointMazeLeftVel-v0", "imitation/PointMazeRightVel-v0"):
        configs[env] = {}
        for reward_type, reward_path in common_config.point_maze_learned_cfgs(
            f"transfer_point_maze{suffix}"
        ):
            configs[env].setdefault(reward_type, {})[reward_path] = dict(
                rl_common.CONFIG_BY_ENV[env]
            )

    return dict(
        configs=configs,
        # Increase from default number of evaluation episodes since we actually report statistics,
        # not just use them to pick the best seed. (Note there may be a slight optimizer's curse
        # here biasing these numbers upward, since we report numbers from the best seed and do not
        # re-evaluate, but it should be small given large number of episodes plus the fact we pick
        # seed with best learned reward but report the ground-truth reward.)
        global_configs={
            "config_updates": {
                "n_episodes_eval": 1000,
            }
        },
        run_tag=f"point_maze_learned{suffix}",
    )


experts_ex.add_named_config("point_maze_learned", _point_maze_learned(fast_config=False))
experts_ex.add_named_config("point_maze_learned_fast", _point_maze_learned(fast_config=True))


@experts_ex.named_config
def test():
    """Unit test config."""
    locals().update(**rl_common.FAST_CONFIG)
    configs = {
        "evaluating_rewards/PointMassLine-v0": {
            "evaluating_rewards/PointMassGroundTruth-v0": {"dummy": {}},
        }
    }
    run_tag = "test"
    _ = locals()
    del _


def _filter_key(k: str) -> Optional[str]:
    """Returns None if key k should be omitted; otherwise returns the (possibly modified) key."""
    if k.startswith("return_"):
        return None
    elif k.endswith("_max") or k.endswith("_min"):
        return None
    else:
        k = k.replace("monitor_return", "mr")
        k = k.replace("wrapped_return", "wr")
        return k


def tabulate_stats(stats: rl_common.Stats) -> str:
    """Pretty-prints the statistics in `stats` in a table."""
    res = []
    for (env_name, (reward_type, reward_path)), vs in stats.items():
        for seed, (x, _log_dir) in enumerate(vs):
            row = {
                "env_name": env_name,
                "reward_type": reward_type,
                "reward_path": reward_path,
                "seed": seed,
            }
            row.update(x)

            filtered_row = {}
            for k, v in row.items():
                if k.endswith("_std"):
                    k = k[:-4] + "_se"
                    v = v / math.sqrt(row["n_traj"])
                new_k = _filter_key(k)
                if new_k is not None:
                    filtered_row[new_k] = v
            res.append(filtered_row)

    return tabulate.tabulate(res, headers="keys")


def select_best(stats: rl_common.Stats, log_dir: str) -> None:
    """Pick the best seed for each environment-reward pair in `stats`.

    Concretely, chooses the seed with highest mean return, and:
      - Adds a symlink `best` in the same directory as the seeds;
      - Adds a key "best" that is `True` for the winning seed and `False` otherwise.
        Note this modifies `stats` in-place.

    For experiments where `reward_type` is not `None` (i.e. we are using a wrapped reward),
    uses `wrapped_return_mean` for selection. Otherwise, uses `monitor_return_mean` (the
    environment ground-truth return).

    Args:
        stats: The statistics to select the best seed from. Note this is modified in-place.
        log_dir: The log directory for this experiment.
    """
    for key, single_stats in stats.items():
        env_name, (reward_type, reward_path) = key
        return_key = "wrapped_return_mean" if reward_type else "monitor_return_mean"

        threshold = env_rewards.THRESHOLDS.get(key, -np.inf)

        returns = [x[return_key] for x, _log in single_stats]
        best_seed = np.argmax(returns)
        base_dir = os.path.join(
            log_dir,
            script_utils.sanitize_path(env_name),
            script_utils.sanitize_path(reward_type),
            script_utils.sanitize_path(reward_path),
        )
        # make symlink relative so it'll work even if directory structure is copied/moved
        os.symlink(str(best_seed), os.path.join(base_dir, "best"))

        for v, _log in single_stats:
            v["pass"] = v[return_key] > threshold
            v["best"] = False

        best_v, _best_log = single_stats[best_seed]

        best_v["best"] = True
        if not best_v["pass"]:
            print(
                f"WARNING: ({env_name}, {reward_type}, {reward_path}) did not meet threshold: "
                f"{best_v[return_key]} < {threshold}"
            )


@experts_ex.main
def train_experts(
    ray_kwargs: Mapping[str, Any],
    num_cpus_fudge_factor: float,
    global_configs: Mapping[str, Any],
    configs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    log_dir: str,
) -> rl_common.Stats:
    """Entry-point into script to train expert policies specified by config.

    Args:
        ray_kwargs: arguments passed to `ray.init`.
        num_cpus_fudge_factor: factor by which to scale `num_vec` to compute CPU requirements.
        global_configs: configuration to apply to all environment-reward pairs.
        configs: configuration for each environment-reward pair.
        log_dir: the root directory to log experiments to.

    Returns:
        Statistics `stats` for all policies, where
            `stats[(env_name, (reward_type, reward_path))][i]`
        are the statistics for seed `i` of the given environment and reward pair.
    """
    ray.init(**ray_kwargs)

    try:
        stats = rl_common.parallel_training(global_configs, configs, num_cpus_fudge_factor, log_dir)
        select_best(stats, log_dir)
    finally:
        ray.shutdown()

    print(tabulate_stats(stats))

    return stats


if __name__ == "__main__":
    script_utils.experiment_main(experts_ex, "train_experts")
