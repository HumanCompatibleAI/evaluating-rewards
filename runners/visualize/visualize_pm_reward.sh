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

VISUALIZE_CMD=$(call_script "visualize_pm_reward" "with")


if [[ $# -ne 1 ]]; then
  echo "usage: $0 <model prefix>"
  echo "model prefix must be relative to ${OUTPUT_ROOT}"
  exit 1
fi

MODEL_PREFIX=$1
LEARNT_MODEL_DIR=${OUTPUT_ROOT}/${MODEL_PREFIX}

MODELS=$(find ${LEARNT_MODEL_DIR} -name model -printf "%P\n" | xargs dirname)

echo "Visualizing models:"
echo ${MODELS}

parallel --header : --results ${OUTPUT_ROOT}/parallel/visualize_pm_reward/ \
    ${VISUALIZE_CMD} env_name=evaluating_rewards/PointMassLine-v0 \
    reward_type=evaluating_rewards/RewardModel-v0 \
    reward_path=${LEARNT_MODEL_DIR}/{reward_path}/model \
    save_path=${LEARNT_MODEL_DIR}/{reward_path}/visualize.pdf \
    ::: reward_path ${MODELS} \
    ::: seed 0 1 2
