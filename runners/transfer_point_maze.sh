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
  TRAIN_TIMESTEPS_MODIFIER="fast"
  COMPARISON_TIMESTEPS_MODIFIER="fast"
  PM_OUTPUT=${EVAL_OUTPUT_ROOT}/transfer_point_maze_fast
else
  IRL_EPOCHS=""
  NAMED_CONFIG="point_maze_learned"
  TRAIN_TIMESTEPS_MODIFIER=""
  COMPARISON_TIMESTEPS_MODIFIER="high_precision"
  PM_OUTPUT=${EVAL_OUTPUT_ROOT}/transfer_point_maze
fi

# Step 1) Train Reward Models

MIXED_POLICY_PATH=${TRANSITION_P}:random:dummy:ppo2:${TRAIN_ENV_RL}/policies/final
$(call_script "rewards.train_preferences" "with") env_name=${ENV_TRAIN} seed=${SEED} \
    target_reward_type=${TARGET_REWARD_TYPE} log_dir=${PM_OUTPUT}/reward/preferences \
    ${TRAIN_TIMESTEPS_MODIFIER} policy_type=mixture policy_path=${MIXED_POLICY_PATH}&
$(call_script "rewards.train_regress" "with") env_name=${ENV_TRAIN} seed=${SEED} \
    target_reward_type=${TARGET_REWARD_TYPE} log_dir=${PM_OUTPUT}/reward/regress \
    ${TRAIN_TIMESTEPS_MODIFIER} dataset_factory_kwargs.policy_type=mixture \
    dataset_factory_kwargs.policy_path=${MIXED_POLICY_PATH}&

for use_action in True False; do
  if [[ ${use_action} == "True" ]]; then
    name="state_action"
  else
    name="state_only"
  fi
  for seed in {0..4}; do
    $(call_script "rewards.train_adversarial" "with") airl point_maze checkpoint_interval=1 \
        seed=${seed} algorithm_kwargs.airl.reward_net_kwargs.use_action=${use_action} \
        ${TRAIN_TIMESTEPS_MODIFIER} ${IRL_EPOCHS} log_dir=${PM_OUTPUT}/reward/irl_${name}&
    done
done

wait

# Step 2) Evaluate IRL reward models to pick the best one.
# This is necessary since IRL is very high-variance, and sometimes fails entirely.

IRL_RETURNS_LOG_DIR=${PM_OUTPUT}/irl_returns/
python -m evaluating_rewards.scripts.distances.rollout_return with point_maze_learned point_maze_learned_multi_seed \
  log_dir=${IRL_RETURNS_LOG_DIR}

# Step 3) Instruct user on next steps
echo "Reward models have finished training."
echo "Look at ${IRL_RETURNS_LOG_DIR}/sacred/cout.txt to identify which IRL model to use."
echo "In the paper, we chose the one with the highest return."
echo "Note this introduces a bias in favour of AIRL; see appendix A.2.2 in the paper for why this is tolerable."
echo "Symlink `irl_state_{only,action}` to the relevant seeds."
echo "Then to produce table of results, run:"
echo "python -m evaluating_rewards.scripts.pipeline.combined_distances with point_maze_learned_good high_precision \
      log_dir=${PM_OUTPUT}/distances/"
echo "python -m evaluating_rewards.scripts.pipeline.combined_distances with point_maze_learned_pathological high_precision \
      log_dir=${PM_OUTPUT}/distances_pathological/"
echo "Or to produce the checkpoint figure, run:"
echo ${DIR}/transfer_point_maze_checkpoints.sh
echo "WARNING: The checkpoint experiment is slow -- it took around 10 days on a 64 vCPU machine."
