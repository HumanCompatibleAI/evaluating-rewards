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

import os
from typing import Any, Mapping, Optional, Sequence

from imitation.scripts import expert_demos
from imitation.util import util
import numpy as np
import ray
import sacred
from sacred import observers
import tabulate

from evaluating_rewards import serialize
from evaluating_rewards.experiments import env_rewards
from evaluating_rewards.scripts import script_utils

experts_ex = sacred.Experiment("train_experts")

# Optional parameters passed to sacred.Experiment.run, such as config_updates
# or named_configs, specified on a per-environment basis.
CONFIG_BY_ENV = {}


@experts_ex.config
def default_config():
    """Default configuration."""
    ray_kwargs = {}
    log_root = os.path.join(serialize.get_output_dir(), "train_experts")
    n_seeds = 3
    global_updates = {
        "config_updates": {
            # Increase from default since we need reliable evaluation to pick the best seed
            "n_episodes_eval": 100,
            # Save a very large rollout so IRL algorithms will have plenty of data.
            # (We can always truncate later if we want to consider data-limited setting.)
            "rollout_save_n_timesteps": 100000,
        }
    }
    configs = {
        "evaluating_rewards/PointMassLine-v0": {"evaluating_rewards/PointMassGroundTruth-v0": {}},
    }
    run_tag = "default"
    _ = locals()
    del _


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
        configs.setdefault(env, {})[gt_reward] = CONFIG_BY_ENV.get(env, {})
    return configs


# TODO: set long enough eval_sample_until
@experts_ex.named_config
def ground_truth():
    """Train RL expert on all configured environments with the ground-truth reward."""
    configs = _make_ground_truth_configs()
    run_tag = "ground_truth"
    _ = locals()
    del _


@experts_ex.named_config
def test():
    """Intended for tests / debugging: small # of seeds and CPU cores, single env-reward pair."""
    ray_kwargs = {"num_cpus": 2}  # CI build only has 2 cores
    n_seeds = 2
    global_updates = {
        "config_updates": {
            "n_episodes_eval": 1,
            "rollout_save_n_timesteps": 100,
        }
    }
    configs = {
        "evaluating_rewards/PointMassLine-v0": {
            "evaluating_rewards/PointMassGroundTruth-v0": {"named_configs": ["fast"]},
        }
    }
    run_tag = "test"
    _ = locals()
    del _


@ray.remote
def rl_worker(
    env_name: str,
    reward_type: str,
    seed: int,
    log_root: str,
    updates: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Trains an RL policy.

    Args:
        env_name: the name of the environment to train a policy in.
        reward_type: the reward type to load and train a policy on.
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
    updates.setdefault("config_updates", {}).update(
        {
            "env_name": env_name,
            "seed": seed,
            "reward_type": reward_type,
            "reward_path": "dummy",
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
        return k.replace("monitor_return", "mr")


def tabulate_stats(stats: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    """Pretty-prints the statistics in `stats` in a table."""
    res = []
    for (env_name, reward_type), vs in stats.items():
        for seed, x in enumerate(vs):
            row = {"env_name": env_name, "reward_type": reward_type, "seed": seed}
            row.update(x)

            filtered_row = {}
            for k, v in row.items():
                new_k = _filter_key(k)
                if new_k is not None:
                    filtered_row[new_k] = v
            res.append(filtered_row)

    return tabulate.tabulate(res, headers="keys")


@experts_ex.main
def train_experts(
    ray_kwargs: Mapping[str, Any],
    n_seeds: int,
    log_dir: str,
    configs: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Mapping[str, Any]:
    """Entry-point into script to train expert policies specified by config."""
    ray.init(**ray_kwargs)

    try:
        # Train policies
        keys = []
        refs = []
        for env_name, inner_configs in configs.items():
            for reward_type, updates in inner_configs.items():
                for seed in range(n_seeds):
                    obj_ref = rl_worker.remote(
                        env_name=env_name,
                        reward_type=reward_type,
                        seed=seed,
                        log_root=log_dir,
                        updates=updates,
                    )
                    keys.append((env_name, reward_type))
                    refs.append(obj_ref)
        raw_values = ray.get(refs)

        stats = {}
        for k, v in zip(keys, raw_values):
            stats.setdefault(k, []).append(v)

        for key, single_stats in stats.items():
            env_name, reward_type = key
            returns = [x["return_mean"] for x in single_stats]
            best_seed = np.argmax(returns)
            base_dir = os.path.join(
                log_dir,
                script_utils.sanitize_path(env_name),
                script_utils.sanitize_path(reward_type),
            )
            # make symlink relative so it'll work even if directory structure is copied/moved
            os.symlink(str(best_seed), os.path.join(base_dir, "best"))

            for v in single_stats:
                v["best"] = False
            single_stats[best_seed]["best"] = True
    finally:
        ray.shutdown()

    print(tabulate_stats(stats))

    return stats


if __name__ == "__main__":
    script_utils.experiment_main(experts_ex, "train_experts")
