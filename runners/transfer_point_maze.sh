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

ENV_TRAIN="imitation/PointMazeLeft-v0"
ENV_TEST="imitation/PointMazeRight-v0"
ENVS="${ENV_TRAIN} ${ENV_TEST}"
ENVS_SANITIZED=$(echo ${ENVS} | sed -e 's/\//_/g')
TARGET_REWARD_TYPE="evaluating_rewards/PointMazeGroundTruthWithCtrl-v0"
RL_TIMESTEPS=200000
PM_OUTPUT=${OUTPUT_ROOT}/transfer_point_maze
SEED=42


# Step 0) Train Policies on Ground Truth
# This acts as a baseline, and the demonstrations are needed for IRL.

EXPERT_DEMO_CMD="$(call_script "expert_demos" "with") seed=${SEED} \
    total_timesteps=${RL_TIMESTEPS} reward_type=${TARGET_REWARD_TYPE}"
${EXPERT_DEMO_CMD} env_name=${ENV_TRAIN} log_dir=${PM_OUTPUT}/expert/train&
${EXPERT_DEMO_CMD} env_name=${ENV_TEST} log_dir=${PM_OUTPUT}/expert/test&

wait

# Step 1) Train Reward Models

# Direct Methods: Preference Comparison and Reward Regression
# These can learn directly from the reward model
$(call_script "train_preferences" "with") env_name=${ENV_TRAIN} seed=${SEED} \
    target_reward_type=${TARGET_REWARD_TYPE} log_dir=${PM_OUTPUT}/reward/preferences&
$(call_script "train_regress" "with") env_name=${ENV_TRAIN} seed=${SEED} \
    target_reward_type=${TARGET_REWARD_TYPE} log_dir=${PM_OUTPUT}/reward/regress&

# IRL: uses demonstrations from previous part
for state_only in True False; do
  if [[ ${state_only} == "True" ]]; then
    name="state_only"
  else
    name="state_action"
  fi
  $(call_script "train_adversarial" "with") env_name=${ENV_TRAIN} seed=${SEED} \
      init_trainer_kwargs.reward_kwargs.state_only=${state_only} \
      rollout_path=${PM_OUTPUT}/expert/train/rollouts/final.pkl \
      log_dir=${PM_OUTPUT}/reward/irl_${name}&
done

wait

# Step 2) Compare Reward Models

parallel --header : --results ${PM_OUTPUT}/parallel/comparison \
  $(call_script "model_comparison" "with") env_name=${ENV_TRAIN} seed={seed} \
  source_reward_type={source_reward_type} \
  source_reward_path=${PM_OUTPUT}/reward/{source_reward_path}/{source_reward_suffix} \
  target_reward_type=${TARGET_REWARD_TYPE} \
  log_dir=${PM_OUTPUT}/comparison/{source_reward_path}/{seed} \
  ::: source_reward_type evaluating_rewards/RewardModel-v0 evaluating_rewards/RewardModel-v0 \
                         imitation/RewardNet_unshaped-v0 imitation/RewardNet_unshaped-v0 \
  :::+ source_reward_path preferences regress irl_state_only irl_state_action \
  :::+ source_reward_suffix model model checkpoints/final/discrim/reward_net \
                            checkpoints/final/discrim/reward_net \
  ::: seed 0 1 2

# Step 3) Train Policies on Learnt Reward Models

parallel --header : --results ${PM_OUTPUT}/parallel/comparison \
  $(call_script "expert_demos" "with") total_timesteps=${RL_TIMESTEPS} \
  env_name={env} seed={seed} reward_type={reward_type} \
  reward_path=${PM_OUTPUT}/reward/{reward_path}/{reward_suffix} \
  log_dir=${PM_OUTPUT}/policy/{env_sanitized}/{reward_path}/{seed} \
  ::: env ${ENVS} :::+ env_sanitized ${ENVS_SANITIZED} \
  ::: reward_type evaluating_rewards/RewardModel-v0 evaluating_rewards/RewardModel-v0 \
                  imitation/RewardNet_unshaped-v0 imitation/RewardNet_unshaped-v0 \
  :::+ reward_path preferences regress irl_state_only irl_state_action \
  :::+ reward_suffix model model checkpoints/final/discrim/reward_net \
                     checkpoints/final/discrim/reward_net \
  ::: seed 0 1 2

# Step 4) Evaluate Policies

for env in ${ENVS}; do
  env_sanitized=$(echo ${env} | sed -e 's/\//_/g')
  parallel --header : --results $HOME/output/parallel/learnt \
           $(call_script "eval_policy" "with") rendder=False num_vec=8 \
           eval_n_timesteps=100000 policy_type=ppo2 env_name=${env} \
           reward_type=${TARGET_REWARD_TYPE} \
           policy_path={policy_path}/policies/final \
           log_dir=${PM_OUTPUT}/policy_eval/{policy_path} \
           ::: reward_type ${types} :::+ reward_type_sanitized ${types_sanitized} \
           ::: policy_path ${PM_OUTPUT}/policy/${env_sanitized}/*/*
done
