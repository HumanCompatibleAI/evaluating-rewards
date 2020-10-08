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

"""Sacred configs for transitions factories.

Used in `plot_epic_heatmap` and `npec_comparison`.
"""

import functools

import sacred

from evaluating_rewards import datasets


def make_config(
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
        visitation factory, leading to an infinite recursion.
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
