#!/usr/bin/env bash
# Copyright 2020 DeepMind Technologies Limited, Adam Gleave
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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

ENV_NAME="evaluating_rewards/PointMassLine-v0"
OUT_DIR="PointMassLine-v0"
TYPES="evaluating_rewards/PointMassSparseWithCtrl-v0 evaluating_rewards/PointMassGroundTruth-v0"
TYPES_SANITIZED=$(echo ${TYPES} | sed -e 's/\//_/g')

cd "${SCRIPT_DIR}"

echo "Saving to '${SCRIPT_DIR}/${OUT_DIR}'"
mkdir -p "${OUT_DIR}"

parallel --header : \
         python -m evaluating_rewards.scripts.model_comparison with \
         total_timesteps=8192 env_name=${ENV_NAME} seed=0 \
         source_reward_type={source_reward_type} \
         target_reward_type={target_reward_type} \
         log_dir=${OUT_DIR}/{source_reward_type_sanitized}_vs_{target_reward_type_sanitized} \
         ::: source_reward_type ${TYPES} \
         :::+ source_reward_type_sanitized ${TYPES_SANITIZED} \
         ::: target_reward_type ${TYPES} \
         :::+ target_reward_type_sanitized ${TYPES_SANITIZED} \

# Copy symlinked Sacred directories

tar cfh ${OUT_DIR}.tar ${OUT_DIR}
rm -rf ${OUT_DIR}
tar xf ${OUT_DIR}.tar
rm ${OUT_DIR}.tar

# Remove unnecessary Sacred and model files to save space
find ${OUT_DIR} | grep -E 'cout.txt|metrics.json|run.json|model' | xargs rm -rf
