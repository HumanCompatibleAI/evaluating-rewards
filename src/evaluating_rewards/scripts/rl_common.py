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

"""Common config and helper methods for scripts training RL algorithms."""

import copy
import math
import os
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence, Tuple

from imitation.scripts import expert_demos
import ray
import sacred
from sacred import observers

from evaluating_rewards.distances import common_config
from evaluating_rewards.scripts import script_utils

Stats = Mapping[Tuple[str, common_config.RewardCfg], Sequence[Tuple[MutableMapping[str, Any], str]]]


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


def make_config(experiment: sacred.Experiment) -> None:
    """Adds configs and named configs to `experiment`.

    The standard config parameters it defines are:
        - ray_kwargs (dict): passed to `ray.init`.
        - num_cpus_fudge_factor (float): used by `parallel_training` to calculate CPU reservation.
        - global_configs (dict): used as a basis for `expert_demos_ex.run` keyword arguments.
    """

    # pylint: disable=unused-variable

    @experiment.config
    def default_config():
        """Default configuration."""
        ray_kwargs = {}
        num_cpus_fudge_factor = 0.5  # we can usually run 2 environments per CPU for MuJoCo
        global_configs = {
            "n_seeds": 9,
            "config_updates": {
                # Increase from default since we need reliable evaluation to pick the best seed
                "n_episodes_eval": 200,
                # Save a very large rollout so that:
                # (a) IRL algorithms will have plenty of data.
                # (We can always truncate later if we want to consider data-limited setting.)
                # (b) So that `evaluating_rewards.scripts.rl.rollout` will have sufficient
                # data for its evaluation.
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
        _ = locals()
        del _

    @experiment.named_config
    def fast():
        """Intended for debugging small # of seeds and CPU cores, single env-reward pair."""
        locals().update(**FAST_CONFIG)


@ray.remote
def rl_worker(
    env_name: str,
    reward_type: Optional[str],
    reward_path: Optional[str],
    seed: int,
    log_root: str,
    updates: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], str]:
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
    script_utils.configure_logging()

    updates = dict(updates)
    log_dir = os.path.join(
        log_root,
        script_utils.sanitize_path(env_name),
        script_utils.sanitize_path(reward_type),
        script_utils.sanitize_path(reward_path),
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
    return run.result, log_dir


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
