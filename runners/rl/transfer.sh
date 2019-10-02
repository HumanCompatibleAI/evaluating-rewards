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

# Train RL policy on learnt rewards

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/../common.sh

EXPERT_DEMOS_CMD=$(call_script "expert_demos" "with")

learnt_model $*  # sets learnt_model_dir, source_reward_type and model_name

declare -A TRANSFER_ENVS=(
  ["imitation/PointMazeLeft-v0"]="imitation/PointMazeRight-v0"
)

echo "Starting model comparison"
for env_name in "${!TRANSFER_ENVS[@]}"; do
  echo "Training policies for for ${env_name}"
  env_name_sanitized=$(echo ${env_name} | sed -e 's/\//_/g')
  MODELS=$(find ${learnt_model_dir}/${env_name_sanitized} -path "*/${model_name}" -printf "%P\n" | sed -e "s@/${model_name}\$@@")

  echo "Training policy on learnt reward"
  echo "Models: ${MODELS}"

  transfer_envs="${env_name} ${TRANSFER_ENVS[$env_name]}"
  transfer_envs_sanitized=$(echo ${transfer_envs} | sed -e 's/\//_/g')

  parallel --header : --results $HOME/output/parallel/expert_demos \
           ${EXPERT_DEMOS_CMD} env_name={env_name} \
           reward_type=${source_reward_type} seed={seed} \
           reward_path=${learnt_model_dir}/${env_name_sanitized}/{reward_path}/${model_name} \
           log_dir=${OUTPUT_ROOT}/expert_transfer/${model_prefix}/${env_name_sanitized}/{env_name_sanitized}/{reward_path}/{seed} \
           ::: env_name ${transfer_envs} \
           :::+ env_name_sanitized ${transfer_envs_sanitized} \
           ::: reward_path ${MODELS} \
           ::: seed 0 1 2
done
