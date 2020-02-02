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
. ${DIR}/common.sh

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <policy prefix>"
  echo "policy prefix must be relative to ${EVAL_OUTPUT_ROOT}"
  exit 1
fi

policy_prefix=$1
policy_dir=${EVAL_OUTPUT_ROOT}/${policy_prefix}
model_name="policies/final"

for env_name in ${ENVS}; do
  echo "Policy evaluation for ${env_name}"
  env_name_sanitized=$(echo ${env_name} | sed -e 's/\//_/g')
  policies=$(find ${policy_dir}/${env_name_sanitized} -path "*/${model_name}" -printf "%P\n" | sed -e "s@/${model_name}\$@@")

  types=${REWARDS_BY_ENV[$env_name]}
  types_sanitized=$(echo ${types} | sed -e 's/\//_/g')

  echo "Evaluating policies under hardcoded rewards"
  echo "Policies: ${policies}"
  echo "Hardcoded rewards: ${types}"

  parallel --header : --results ${EVAL_OUTPUT_ROOT}/parallel/learnt \
           ${EVAL_POLICY_CMD} env_name=${env_name} policy_type=ppo2 \
           reward_type={reward_type} \
           policy_path=${policy_dir}/${env_name_sanitized}/{policy_path}/${model_name} \
           log_dir=${EVAL_OUTPUT_ROOT}/eval/${policy_prefix}/${env_name_sanitized}/{policy_path}/eval_under_{reward_type_sanitized} \
           ::: reward_type ${types} \
           :::+ reward_type_sanitized ${types_sanitized} \
           ::: policy_path ${policies}
done
