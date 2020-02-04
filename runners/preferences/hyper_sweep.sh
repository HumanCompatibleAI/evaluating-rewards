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

TARGET_REWARDS="
PointMassDense-v0
PointMassSparse-v0
"

parallel --header : --results ${EVAL_OUTPUT_ROOT}/parallel/train_preferences_hyper \
         ${TRAIN_CMD} env_name=evaluating_rewards/PointMassLine-v0 \
         seed={seed} target_reward_type=evaluating_rewards/{target_reward} \
         batch_timesteps={batch_timesteps} trajectory_length={trajectory_length} \
         learning_rate={lr} total_timesteps=5e6 \
         log_dir=${EVAL_OUTPUT_ROOT}/train_preferences_hyper/{target_reward}/batch{batch_timesteps}_of_{trajectory_length}_lr{lr}/{seed} \
         ::: target_reward ${TARGET_REWARDS} \
         ::: batch_timesteps 500 2500 10000 50000 250000 \
         ::: trajectory_length 1 5 25 100 \
         ::: lr 1e-4 1e-3 1e-2 1e-1 \
         ::: seed 0 1 2
