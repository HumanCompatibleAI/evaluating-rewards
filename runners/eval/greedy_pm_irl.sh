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

for env in ${ENVS}; do
  env_sanitized=$(echo ${env} | sed -e 's/\//_/g')
  reward_paths=${EVAL_OUTPUT_ROOT}/train_adversarial/${env_sanitized}/*/final/discrim/reward_net
  policy_paths=""
  for rew_path in ${reward_paths}; do
    policy_paths="${policy_paths} BasicShapedRewardNet_shaped:${rew_path}"
    policy_paths="${policy_paths} BasicShapedRewardNet_unshaped:${rew_path}"
  done
  parallel --header : --results ${EVAL_OUTPUT_ROOT}/parallel/greedy_pm_irl \
           ${EVAL_POLICY_CMD} env_name=${env} policy_type=evaluating_rewards/MCGreedy-v0 \
           policy_path={policy_path} \
           ::: policy_path ${policy_paths}
done