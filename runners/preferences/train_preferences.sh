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

TRAIN_CMD=$(call_script "train_preferences" "with")

echo "Starting preference comparison"
for env_name in "${!REWARDS_BY_ENV[@]}"; do
  echo "Preference comparison for ${env_name}"
  types=${REWARDS_BY_ENV[$env_name]}
  env_name_sanitized=$(echo ${env_name} | sed -e 's/\//_/g')
  types_sanitized=$(echo ${types} | sed -e 's/\//_/g')

  parallel --header : --results ${EVAL_OUTPUT_ROOT}/parallel/train_preferences/${env_name} \
           ${TRAIN_CMD} env_name=${env_name} \
           seed={seed} target_reward_type={target_reward} \
           log_dir=${HOME}/output/train_preferences/${env_name_sanitized}/{target_reward_sanitized}/{seed} \
           ::: target_reward ${types} \
           :::+ target_reward_sanitized ${types_sanitized} \
           ::: seed 0 1 2
done
