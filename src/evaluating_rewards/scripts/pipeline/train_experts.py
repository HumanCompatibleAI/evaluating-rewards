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

import copy
import math
import os
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence

from imitation.scripts import expert_demos
from imitation.util import util
import numpy as np
import ray
import sacred
from sacred import observers
import tabulate

from evaluating_rewards import serialize
from evaluating_rewards.distances import common_config
from evaluating_rewards.experiments import env_rewards
from evaluating_rewards.scripts import script_utils

Stats = Mapping[str, Sequence[MutableMapping[str, Any]]]

experts_ex = sacred.Experiment("train_experts")


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear schedule, e.g. for learning rate.

    Args:
        initial_value: the initial value.

    Returns:
        A function computing the current output.
    """

    def func(progress: float) -> float:
        """Computes current rate.

        Args:
            progress: between 1 (beginning) and 0 (end).

        Returns:
            The current rate.
        """
        return progress * initial_value

    return func


POINT_MASS_CONFIGS = {
    "config_updates": {
        "init_rl_kwargs": {
            "n_steps": 512,
        },
    },
}

# Optional parameters passed to sacred.Experiment.run, such as config_updates
# or named_configs, specified on a per-environment basis.
CONFIG_BY_ENV = {
    "evaluating_rewards/PointMassLine-v0": POINT_MASS_CONFIGS,
    "imitation/PointMazeLeftVel-v0": POINT_MASS_CONFIGS,
    "imitation/PointMazeRightVel-v0": POINT_MASS_CONFIGS,
    "seals/HalfCheetah-v0": {
        "config_updates": {
            # HalfCheetah does OK after 1e6 but keeps on improving
            "total_timesteps": int(5e6),
        }
    },
}


@experts_ex.config
def default_config():
    """Default configuration."""
    ray_kwargs = {}
    num_cpus_fudge_factor = 0.5  # we can usually run 2 environments per CPU for MuJoCo
    log_root = os.path.join(serialize.get_output_dir(), "train_experts")
    global_configs = {
        "n_seeds": 9,
        "config_updates": {
            # Increase from default since we need reliable evaluation to pick the best seed
            "n_episodes_eval": 200,
            # Save a very large rollout so IRL algorithms will have plenty of data.
            # (We can always truncate later if we want to consider data-limited setting.)
            "rollout_save_n_timesteps": 100000,
            # Set some reasonable default hyperparameters for continuous control tasks.
            "num_vec": 8,
            "init_rl_kwargs": {
                "n_steps": 256,  # batch size = n_steps * num_vec
                "learning_rate": linear_schedule(3e-4),
                "cliprange_vf": -1,  # matches original PPO algorithm
            },
        },
    }
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
    log_dir = os.path.join(log_root, run_tag, util.make_unique_timestamp())
    _ = locals()
    del _


def _make_ground_truth_configs():
    """Ground truth configs.

    Separate function to avoid polluting Sacred ConfigScope with local variables."""
    configs = {}
    for env, gt_reward in env_rewards.GROUND_TRUTH_REWARDS_BY_ENV.items():
        cfg = CONFIG_BY_ENV.get(env, {})
        configs.setdefault(env, {}).setdefault(str(gt_reward), {})["dummy"] = cfg
    return configs


@experts_ex.named_config
def ground_truth():
    """Train RL expert on all configured environments with the ground-truth reward."""
    configs = _make_ground_truth_configs()
    run_tag = "ground_truth"
    _ = locals()
    del _


@experts_ex.named_config
def point_maze_pathologicals():
    """Train RL policies on the "wrong" rewards in PointMaze."""
    configs = {
        env: {
            reward: {"dummy": dict(CONFIG_BY_ENV[env])}
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


def _point_maze_learned(fast_config: bool):
    """Train RL policies on learned rewards in PointMaze."""
    prefix = "point_maze_learned_fast" if fast_config else "point_maze_learned"
    configs = {}
    for env in ("imitation/PointMazeLeftVel-v0", "imitation/PointMazeRightVel-v0"):
        configs[env] = {}
        for reward_type, reward_path in common_config.point_maze_learned_cfgs(prefix):
            configs[env].setdefault(reward_type, {})[reward_path] = dict(CONFIG_BY_ENV[env])

    return dict(
        configs=configs,
        # Increase from default number of evaluation episodes since we actually report statistics,
        # not just use them to pick the best seed. (Note there may be a slight optimizer's curse
        # here biasing these numbers upward, since we report numbers from the best seed and do not
        # re-evaluate, but it should be small given large number of episodes plus the fact we pick
        # seed with best learned reward but report that of best ground-truth reward.)
        global_configs={
            "config_updates": {
                "n_episodes_eval": 1000,
            }
        },
        run_tag=prefix,
    )


experts_ex.add_named_config("point_maze_learned", _point_maze_learned(fast_config=False))
experts_ex.add_named_config("point_maze_learned_fast", _point_maze_learned(fast_config=True))


FAST_CONFIG = dict(
    ray_kwargs={
        # CI build only has 1 core per test
        "num_cpus": 1,
    },
    global_configs={
        "n_seeds": 2,
        "config_updates": {
            "num_vec": 1,  # avoid taking up too many resources on CI
            "n_episodes_eval": 1,
            "rollout_save_n_timesteps": 100,
        },
        "named_configs": ["fast"],
    },
)


@experts_ex.named_config
def fast():
    """Intended for debugging small # of seeds and CPU cores, single env-reward pair."""
    locals().update(**FAST_CONFIG)


@experts_ex.named_config
def test():
    """Unit test config."""
    locals().update(**FAST_CONFIG)
    configs = {
        "evaluating_rewards/PointMassLine-v0": {
            "evaluating_rewards/PointMassGroundTruth-v0": {"dummy": {}},
        }
    }
    run_tag = "test"
    _ = locals()
    del _


@ray.remote
def rl_worker(
    env_name: str,
    reward_type: Optional[str],
    reward_path: Optional[str],
    seed: int,
    log_root: str,
    updates: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Trains an RL policy.

    Args:
        env_name: the name of the environment to train a policy in.
        reward_type: the reward type to load and train a policy on;
                     if None, use original environment reward.
        reward_path: the path to load the reward from. (Ignored if
            `reward_type` is None.)
        seed: seed for RL algorithm.
        log_root: the root logging directory for this experiment.
            Each RL policy will have a subdirectory created:
            `{env_name}/{reward_type}/{seed}`.
        updates: Configuration updates to pass to `expert_demos_ex.run`.

    Returns:
        Training statistics returned by `expert_demos.rollouts_and_policy`.
    """
    updates = dict(updates)
    log_dir = os.path.join(
        log_root,
        script_utils.sanitize_path(env_name),
        script_utils.sanitize_path(reward_type),
        str(seed),
    )
    updates["config_updates"] = dict(updates.get("config_updates", {}))
    updates["config_updates"].update(
        {
            "env_name": env_name,
            "seed": seed,
            "reward_type": reward_type,
            "reward_path": reward_path,
            "log_dir": log_dir,
        }
    )
    observer = observers.FileStorageObserver(os.path.join(log_dir, "sacred_out"))
    expert_demos.expert_demos_ex.observers.append(observer)
    run = expert_demos.expert_demos_ex.run(**updates)
    assert run.status == "COMPLETED"
    return run.result


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


def tabulate_stats(stats: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    """Pretty-prints the statistics in `stats` in a table."""
    res = []
    for (env_name, reward_type), vs in stats.items():
        for seed, x in enumerate(vs):
            row = {"env_name": env_name, "reward_type": reward_type, "seed": seed}
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


def parallel_training(
    global_configs: Mapping[str, Any],
    configs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    num_cpus_fudge_factor: float,
    log_dir: str,
) -> Stats:
    """Train experts in parallel.

    Args:
        global_configs: configuration to apply to all environment-reward pairs.
        configs: configuration for each environment and reward type pair.
        num_cpus_fudge_factor: factor by which to scale `num_vec` to compute CPU requirements.
        log_dir: the root directory to log experiments to.

    Returns:
        Statistics `stats` for all policies, where `stats[(env_name, reward_type)][i]` are
        the statistics for seed `i` of the given environment and reward pair.
    """
    keys = []
    refs = []
    for env_name, inner_configs in configs.items():
        for reward_type, path_configs in inner_configs.items():
            if reward_type == "None":  # Sacred config doesn't support literal None
                reward_type = None
            for reward_path, cfg in path_configs.items():
                updates = copy.deepcopy(dict(global_configs))
                script_utils.recursive_dict_merge(updates, cfg, overwrite=True)
                n_seeds = updates.pop("n_seeds", 1)
                for seed in range(n_seeds):
                    # Infer the number of parallel environments being run and reserve that many CPUs
                    config_updates = updates.get("config_updates", {})
                    num_vec = config_updates.get("num_vec", 8)  # 8 is default in expert_demos
                    parallel = config_updates.get("parallel", True)
                    num_cpus = math.ceil(num_vec * num_cpus_fudge_factor) if parallel else 1
                    rl_worker_tagged = rl_worker.options(num_cpus=num_cpus)
                    # Now execute RL training
                    obj_ref = rl_worker_tagged.remote(
                        env_name=env_name,
                        reward_type=reward_type,
                        reward_path=reward_path,
                        seed=seed,
                        log_root=log_dir,
                        updates=updates,
                    )
                    keys.append((env_name, (reward_type, reward_path)))
                    refs.append(obj_ref)
    raw_values = ray.get(refs)

    stats = {}
    for k, v in zip(keys, raw_values):
        stats.setdefault(k, []).append(v)

    return stats


def select_best(stats: Stats, log_dir: str) -> None:
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

        returns = [x[return_key] for x in single_stats]
        best_seed = np.argmax(returns)
        base_dir = os.path.join(
            log_dir,
            script_utils.sanitize_path(env_name),
            script_utils.sanitize_path(reward_type),
            script_utils.sanitize_path(reward_path),
        )
        # make symlink relative so it'll work even if directory structure is copied/moved
        os.symlink(os.path.join(base_dir, "best"), str(best_seed))

        for v in single_stats:
            v["pass"] = v[return_key] > threshold
            v["best"] = False

        single_stats[best_seed]["best"] = True
        if not single_stats[best_seed]["pass"]:
            print(
                f"WARNING: ({env_name}, {reward_type}, {reward_path}) did not meet threshold: "
                f"{single_stats[best_seed][return_key]} < {threshold}"
            )


@experts_ex.main
def train_experts(
    ray_kwargs: Mapping[str, Any],
    num_cpus_fudge_factor: float,
    global_configs: Mapping[str, Any],
    configs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    log_dir: str,
) -> Stats:
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
        stats = parallel_training(global_configs, configs, num_cpus_fudge_factor, log_dir)
        select_best(stats, log_dir)
    finally:
        ray.shutdown()

    print(tabulate_stats(stats))

    return stats


if __name__ == "__main__":
    script_utils.experiment_main(experts_ex, "train_experts")
