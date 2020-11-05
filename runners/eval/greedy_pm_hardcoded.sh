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

GREEDY_REWARD_MODELS="evaluating_rewards/PointMassGroundTruth-v0:None:0.99 \
                      evaluating_rewards/PointMassSparseWithCtrl-v0:None:0.99 \
                      evaluating_rewards/PointMassDenseWithCtrl-v0:None:0.99"

parallel --header : --results ${EVAL_OUTPUT_ROOT}/parallel/greedy_pm_hardcoded \
         ${EVAL_POLICY_CMD} policy_type=evaluating_rewards/MCGreedy-v0 \
         env_name={env}  policy_path={policy_path} \
         ::: env ${PM_ENVS} \
         ::: policy_path ${GREEDY_REWARD_MODELS}
