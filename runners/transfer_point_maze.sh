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

if [[ ${fast} == "true" ]]; then
  # intended for debugging
  RL_TIMESTEPS="total_timesteps=16384"
  IRL_EPOCHS="n_epochs=5"
  PREFERENCES_TIMESTEPS="fast"
  REGRESS_TIMESTEPS="fast"
  COMPARISON_TIMESTEPS="fast"
  EVAL_TIMESTEPS=4096
  PM_OUTPUT=${OUTPUT_ROOT}/transfer_point_maze_fast
else
  RL_TIMESTEPS=""
  IRL_EPOCHS=""
  PREFERENCES_TIMESTEPS=""
  REGRESS_TIMESTEPS=""
  COMPARISON_TIMESTEPS=""
  EVAL_TIMESTEPS=100000
  PM_OUTPUT=${OUTPUT_ROOT}/transfer_point_maze
fi


# Step 0) Train Policies on Ground Truth
# This acts as a baseline, and the demonstrations are needed for IRL.

EXPERT_DEMO_CMD="$(call_script "expert_demos" "with") seed=${SEED} \
    ${NORMALIZE} init_rl_kwargs.n_steps=${N_STEPS} \
    ${RL_TIMESTEPS} reward_type=${TARGET_REWARD_TYPE}"
${EXPERT_DEMO_CMD} env_name=${ENV_TRAIN} log_dir=${PM_OUTPUT}/expert/train \
    rollout_save_n_episodes=1000&
${EXPERT_DEMO_CMD} env_name=${ENV_TEST} log_dir=${PM_OUTPUT}/expert/test&

wait

# Step 1) Train Reward Models

# Direct Methods: Preference Comparison and Reward Regression
# These can learn directly from the reward model

MIXED_POLICY_PATH=${TRANSITION_P}:random:dummy:ppo2:${PM_OUTPUT}/expert/train/policies/final
$(call_script "train_preferences" "with") env_name=${ENV_TRAIN} seed=${SEED} \
    target_reward_type=${TARGET_REWARD_TYPE} log_dir=${PM_OUTPUT}/reward/preferences \
    ${PREFERENCES_TIMESTEPS} policy_type=mixture policy_path=${MIXED_POLICY_PATH}&
$(call_script "train_regress" "with") env_name=${ENV_TRAIN} seed=${SEED} \
    target_reward_type=${TARGET_REWARD_TYPE} log_dir=${PM_OUTPUT}/reward/regress \
    ${REGRESS_TIMESTEPS} dataset_factory_kwargs.policy_type=mixture \
    dataset_factory_kwargs.policy_path=${MIXED_POLICY_PATH}&

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
      ${IRL_EPOCHS} log_dir=${PM_OUTPUT}/reward/irl_${name}&
done

wait

# Step 2a) Compare Reward Models

for name in comparison_expert comparison_mixture comparison_random; do
  if [[ ${name} == "comparison_expert" ]]; then
    extra_flags="dataset_factory_kwargs.policy_type=ppo2 \
                 dataset_factory_kwargs.policy_path=${PM_OUTPUT}/expert/train/policies/final"
  elif [[ ${name} == "comparison_mixture" ]]; then
    extra_flags="dataset_factory_kwargs.policy_type=mixture \
                 dataset_factory_kwargs.policy_path=${MIXED_POLICY_PATH}"
  elif [[ ${name} == "comparison_random" ]]; then
    extra_flags=""
  else
    echo "BUG: unknown name ${name}"
    exit 1
  fi
  parallel --header : --results ${PM_OUTPUT}/parallel/${name} \
    $(call_script "model_comparison" "with") \
    env_name=${ENV_TRAIN} ${extra_flags} \
    seed={seed} source_reward_type={source_reward_type} \
    source_reward_path=${PM_OUTPUT}/reward/{source_reward_path}/{source_reward_suffix} \
    target_reward_type=${TARGET_REWARD_TYPE} \
    ${COMPARISON_TIMESTEPS} log_dir=${PM_OUTPUT}/${name}/{source_reward_path}/{seed} \
    ::: source_reward_type evaluating_rewards/Zero-v0 \
        evaluating_rewards/RewardModel-v0 evaluating_rewards/RewardModel-v0 \
        imitation/RewardNet_unshaped-v0 imitation/RewardNet_unshaped-v0 \
    :::+ source_reward_path zero preferences regress irl_state_only irl_state_action \
    :::+ source_reward_suffix dummy model model checkpoints/final/discrim/reward_net \
                              checkpoints/final/discrim/reward_net \
    ::: seed 0 1 2&
done

# Step 2b) Train Policies on Learnt Reward Models

parallel --header : --results ${PM_OUTPUT}/parallel/transfer \
  $(call_script "expert_demos" "with") ${RL_TIMESTEPS} \
  ${NORMALIZE} init_rl_kwargs.n_steps=${N_STEPS} \
  env_name={env} seed={seed} reward_type={reward_type} \
  reward_path=${PM_OUTPUT}/reward/{reward_path}/{reward_suffix} \
  log_dir=${PM_OUTPUT}/policy/{env_sanitized}/{reward_path}/{seed} \
  ::: env ${ENVS} :::+ env_sanitized ${ENVS_SANITIZED} \
  ::: reward_type evaluating_rewards/RewardModel-v0 evaluating_rewards/RewardModel-v0 \
                  imitation/RewardNet_unshaped-v0 imitation/RewardNet_unshaped-v0 \
  :::+ reward_path preferences regress irl_state_only irl_state_action \
  :::+ reward_suffix model model checkpoints/final/discrim/reward_net \
                     checkpoints/final/discrim/reward_net \
  ::: seed 0 1 2&

wait

# Step 3) Evaluate Policies

for env in ${ENVS}; do
  env_sanitized=$(echo ${env} | sed -e 's/\//_/g')
  parallel --header : --results $HOME/output/parallel/learnt \
           $(call_script "eval_policy" "with") render=False num_vec=8 \
           eval_n_timesteps=${EVAL_TIMESTEPS} policy_type=ppo2 env_name=${env} \
           reward_type=${TARGET_REWARD_TYPE} \
           policy_path={policy_path}/policies/final \
           log_dir=${PM_OUTPUT}/policy_eval/{policy_path} \
           ::: policy_path ${PM_OUTPUT}/policy/${env_sanitized}/*/*
done
