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


function call_script {
  script_name=$1
  shift
  echo "python -m evaluating_rewards.scripts.${script_name} $@"
}

function learnt_model {
  if [[ $# -ne 1 ]]; then
    echo "usage: $0 <model prefix>"
    echo "model prefix must be relative to ${OUTPUT_ROOT}"
    exit 1
  fi

  model_prefix=$1
  learnt_model_dir=${OUTPUT_ROOT}/${model_prefix}

  case ${model_prefix} in
  train_adversarial)
    source_reward_type="imitation/RewardNet_unshaped-v0"
    model_name="checkpoints/final/discrim/reward_net"
    ;;
  *)
    source_reward_type="evaluating_rewards/RewardModel-v0"
    model_name="model"
    ;;
  esac
}

PM_ENVS="evaluating_rewards/PointMassLine-v0 \
         evaluating_rewards/PointMassLineVariableHorizon-v0 \
         evaluating_rewards/PointMassGrid-v0"

ENV_REWARD_CMD=$(call_script "env_rewards")
# This declares REWARDS_BY_ENV
echo "Loading environment to reward mapping"
eval "$(${ENV_REWARD_CMD} 2>/dev/null)"
ENVS="${!REWARDS_BY_ENV[@]}"
echo "Loaded mappings for environments ${ENVS}"

OUTPUT_ROOT=/mnt/eval_reward/data