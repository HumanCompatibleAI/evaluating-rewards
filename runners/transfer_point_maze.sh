#!/usr/bin/env bash
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

# End-to-end script to test transfer of reward models in imitation/PointMaze*-v0

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/common.sh

ENV_TRAIN="imitation/PointMazeLeftVel-v0"
ENV_TEST="imitation/PointMazeRightVel-v0"
ENVS="${ENV_TRAIN} ${ENV_TEST}"
ENVS_SANITIZED=$(echo ${ENVS} | sed -e 's/\//_/g')
TARGET_REWARD_TYPE="evaluating_rewards/PointMazeGroundTruthWithCtrl-v0"
N_STEPS=2048
NORMALIZE="normalize_kwargs.norm_obs=False"  # normalize reward still, but not observations
SEED=42
TRANSITION_P=0.05

TRAIN_ENV_RL=${EVAL_OUTPUT_ROOT}/train_experts/ground_truth/20201203_105631_297835/imitation_PointMazeLeftVel-v0/evaluating_rewards_PointMazeGroundTruthWithCtrl-v0/best/

if [[ ${fast} == "true" ]]; then
  # intended for debugging
  IRL_EPOCHS="n_epochs=5"
  NAMED_CONFIG="point_maze_learned_fast"
  TIMESTEPS_MODIFIER="fast"
  PM_OUTPUT=${EVAL_OUTPUT_ROOT}/transfer_point_maze_fast
else
  IRL_EPOCHS=""
  NAMED_CONFIG="point_maze_learned"
  TIMESTEPS_MODIFIER=""
  PM_OUTPUT=${EVAL_OUTPUT_ROOT}/transfer_point_maze
fi

# Step 1) Train Reward Models

MIXED_POLICY_PATH=${TRANSITION_P}:random:dummy:ppo2:${TRAIN_ENV_RL}/policies/final
$(call_script "rewards.train_preferences" "with") env_name=${ENV_TRAIN} seed=${SEED} \
    target_reward_type=${TARGET_REWARD_TYPE} log_dir=${PM_OUTPUT}/reward/preferences \
    ${TIMESTEPS_MODIFIER} policy_type=mixture policy_path=${MIXED_POLICY_PATH}&
$(call_script "rewards.train_regress" "with") env_name=${ENV_TRAIN} seed=${SEED} \
    target_reward_type=${TARGET_REWARD_TYPE} log_dir=${PM_OUTPUT}/reward/regress \
    ${TIMESTEPS_MODIFIER} dataset_factory_kwargs.policy_type=mixture \
    dataset_factory_kwargs.policy_path=${MIXED_POLICY_PATH}&

for state_only in True False; do
  if [[ ${state_only} == "True" ]]; then
    name="state_only"
  else
    name="state_action"
  fi
  $(call_script "rewards.train_adversarial" "with") env_name=${ENV_TRAIN} seed=${SEED} \
      init_trainer_kwargs.reward_kwargs.state_only=${state_only} \
      rollout_path=${TRAIN_ENV_RL}/rollouts/final.pkl \
      ${IRL_EPOCHS} log_dir=${PM_OUTPUT}/reward/irl_${name}&
done

wait

# Step 2) Evaluate Reward Models with Distance Metrics

for cmd in epic npec erc; do
  python -m evaluating_rewards.scripts.distances.${cmd} with \
      ${NAMED_CONFIG} ${TIMESTEPS_MODIFIER} log_dir=${PM_OUTPUT}/distance/${cmd}
done

# Step 3) Train Policies on Learnt Reward Models

python -m evaluating_rewards.scripts.pipeline.train_experts with \
    ${NAMED_CONFIG} ${TIMESTEPS_MODIFIER} log_dir=${PM_OUTPUT}/policy_learned

wait
