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

"""Thin wrapper around imitation.scripts.expert_demos."""

from imitation.scripts import expert_demos
import stable_baselines

from evaluating_rewards.scripts import script_utils


@expert_demos.expert_demos_ex.named_config
def lunar_lander():
    """PPO on LunarLander"""
    env_name = "evaluating_rewards/LunarLanderContinuous-v0"
    # Hyperparams from https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/ppo2.yml
    num_vec = 16
    init_rl_kwargs = dict(
        n_steps=1024, nminibatches=32, lam=0.98, gamma=0.999, noptepochs=4, ent_coef=0.01,
    )
    _ = locals()
    del _


@expert_demos.expert_demos_ex.named_config
def lunar_lander_sac():
    """SAC on LunarLander"""
    env_name = "evaluating_rewards/LunarLanderContinuous-v0"
    # Hyperparams from https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/sac.yml
    num_vec = 1
    total_timesteps = int(5e5)
    init_rl_kwargs = dict(
        model_class=stable_baselines.SAC,
        policy_class=stable_baselines.sac.policies.MlpPolicy,
        batch_size=256,
        learning_starts=1000,
    )
    log_interval = 10000
    policy_save_interval = 10000
    _ = locals()
    del _


if __name__ == "__main__":
    script_utils.add_logging_config(expert_demos.expert_demos_ex, "expert_demos")
    script_utils.experiment_main(expert_demos.expert_demos_ex, "expert_demos", sacred_symlink=False)
