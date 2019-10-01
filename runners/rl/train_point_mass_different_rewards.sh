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

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/../common.sh

EXPERT_DEMOS_CMD=$(call_script "expert_demos" "with")
EVAL_POLICY_CMD=$(call_script "eval_policy" "with")

REWARD_TYPES="evaluating_rewards/PointMassGroundTruth-v0 "
REWARD_TYPES+="evaluating_rewards/PointMassSparse-v0 "
REWARD_TYPES+="evaluating_rewards/PointMassDense-v0"
SANITIZED_REWARD_TYPES=""
for reward_type in ${REWARD_TYPES}; do
    reward_sanitized=$(echo ${reward_type} | sed -e 's/\//_/g')
    SANITIZED_REWARD_TYPES+="${reward_sanitized} "
done

parallel --header : --results $HOME/output/parallel/point_mass_different_rewards/expert \
         ${EXPERT_DEMOS_CMD} \
         env_name=evaluating_rewards/PointMassLine-v0 \
         reward_type={reward_type} \
         seed={seed} \
         log_dir=$HOME/output/point_mass_different_rewards/expert/{sanitized_reward_type}/{seed} \
         ::: reward_type ${REWARD_TYPES} \
         :::+ sanitized_reward_type ${SANITIZED_REWARD_TYPES} \
         ::: seed 0 1 2

for expert_sanitized_type in ${SANITIZED_REWARD_TYPES}; do
  parallel --header : --results $HOME/output/parallel/point_mass_different_rewards/eval/${expert_sanitized_type} \
           ${EVAL_POLICY_CMD} \
           render=False \
           timesteps=10000 \
           env_name=evaluating_rewards/PointMassLine-v0 \
           policy_type=ppo2 \
           policy_path=$HOME/output/point_mass_different_rewards/expert/${expert_sanitized_type}/{seed}/policies/final \
           reward_type={reward_type} \
           log_dir=$HOME/output/point_mass_different_rewards/eval/${expert_sanitized_type}/{sanitized_reward_type}/{seed} \
           ::: reward_type ${REWARD_TYPES} \
           :::+ sanitized_reward_type ${SANITIZED_REWARD_TYPES} \
           ::: seed 0 1 2
done
