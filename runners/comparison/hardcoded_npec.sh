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

# Compare hardcoded rewards in PointMass to each other

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/../common.sh

TRAIN_CMD=$(call_script "model_comparison" "with")

echo "Starting model comparison"
for env_name in "${!REWARDS_BY_ENV[@]}"; do
  echo "Model comparison for ${env_name}"
  types=${REWARDS_BY_ENV[$env_name]}
  env_name_sanitized=$(echo ${env_name} | sed -e 's/\//_/g')
  types_sanitized=$(echo ${types} | sed -e 's/\//_/g')

  named_configs=""
  if [[ ${env_name} == "evaluating_rewards/PointMassLine-v0" ]]; then
    named_configs="dataset_random_transition"
  fi

  parallel --header : --results ${EVAL_OUTPUT_ROOT}/parallel/comparison/hardcoded_mujoco \
           ${TRAIN_CMD} env_name=${env_name} ${named_configs} \
           seed={seed} \
           source_reward_type={source_reward_type} \
           target_reward_type={target_reward_type} \
           log_dir=${EVAL_OUTPUT_ROOT}/comparison/hardcoded/${env_name_sanitized}/{source_reward_type_sanitized}_vs_{target_reward_type_sanitized}_seed{seed} \
           ::: source_reward_type ${types} \
           :::+ source_reward_type_sanitized ${types_sanitized} \
           ::: target_reward_type ${types} \
           :::+ target_reward_type_sanitized ${types_sanitized} \
           ::: seed 0 1 2
done
