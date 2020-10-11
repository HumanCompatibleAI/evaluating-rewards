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

"""Sacred configs for distances: transitions factories and per-environment reward configurations.

Used in `plot_heatmap`, `epic`, `npec` and `erc`.
"""

import functools
import os
from typing import Any, Iterable, Mapping, Tuple

import sacred

from evaluating_rewards import datasets

RewardCfg = Tuple[str, str]  # (type, path)
AggregatedDistanceReturn = Mapping[str, Mapping[Tuple[RewardCfg, RewardCfg], float]]


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


def make_transitions_configs(
    experiment: sacred.Experiment,
):  # pylint: disable=unused-variable
    """Add configs to experiment `ex` related to visitations transition factory."""

    @experiment.named_config
    def sample_from_env_spaces(env_name):
        """Randomly sample from Gym spaces."""
        obs_sample_dist_factory = functools.partial(datasets.sample_dist_from_env_name, obs=True)
        act_sample_dist_factory = functools.partial(datasets.sample_dist_from_env_name, obs=False)
        sample_dist_factory_kwargs = {"env_name": env_name}
        obs_sample_dist_factory_kwargs = {}
        act_sample_dist_factory_kwargs = {}
        sample_dist_tag = "random_space"  # only used for logging
        _ = locals()
        del _

    @experiment.named_config
    def dataset_iid(
        env_name,
        obs_sample_dist_factory,
        act_sample_dist_factory,
        obs_sample_dist_factory_kwargs,
        act_sample_dist_factory_kwargs,
        sample_dist_factory_kwargs,
        sample_dist_tag,
    ):
        """Visitation distribution is i.i.d. samples from sample distributions.

        Set this to make `computation_kind` "sample" consistent with "mesh".

        WARNING: you *must* override the `sample_dist` *before* calling this,
        e.g. by using `sample_from_env_spaces`, since by default it is marginalized from
        `visitations_factory`, leading to an infinite recursion.
        """
        visitations_factory = datasets.transitions_factory_iid_from_sample_dist_factory
        visitations_factory_kwargs = {
            "obs_dist_factory": obs_sample_dist_factory,
            "act_dist_factory": act_sample_dist_factory,
            "obs_kwargs": obs_sample_dist_factory_kwargs,
            "act_kwargs": act_sample_dist_factory_kwargs,
            "env_name": env_name,
        }
        visitations_factory_kwargs.update(**sample_dist_factory_kwargs)
        dataset_tag = "iid_" + sample_dist_tag
        _ = locals()
        del _

    @experiment.named_config
    def dataset_from_random_transitions(env_name):
        visitations_factory = datasets.transitions_factory_from_random_model
        visitations_factory_kwargs = {"env_name": env_name}
        dataset_tag = "random_transitions"
        _ = locals()
        del _

    @experiment.named_config
    def dataset_permute(visitations_factory, visitations_factory_kwargs, dataset_tag):
        """Permute transitions of factory specified in *previous* named configs on the CLI."""
        visitations_factory_kwargs["factory"] = visitations_factory
        visitations_factory = datasets.transitions_factory_permute_wrapper
        dataset_tag = "permuted_" + dataset_tag
        _ = locals()
        del _

    @experiment.named_config
    def dataset_noise_rollouts(env_name):
        """Add noise to rollouts of serialized policy."""
        visitations_factory_kwargs = {
            "trajectory_factory": datasets.trajectory_factory_noise_wrapper,
            "factory": datasets.trajectory_factory_from_serialized_policy,
            "policy_type": "random",
            "policy_path": "dummy",
            "noise_env_name": env_name,
            "env_name": env_name,
        }
        visitations_factory = datasets.transitions_factory_from_trajectory_factory
        dataset_tag = "noised_random_policy"
        _ = locals()
        del _
