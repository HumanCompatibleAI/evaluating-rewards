#!/usr/bin/env bash

# Compare hardcoded rewards in PointMass to each other

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/../common.sh

TRAIN_CMD=$(call_script "model_comparison" "with")

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <model prefix>"
  echo "model prefix must be relative to ${OUTPUT_ROOT}"
  exit 1
fi

MODEL_PREFIX=$1
LEARNT_MODEL_DIR=${OUTPUT_ROOT}/${MODEL_PREFIX}

echo "Starting model comparison"
for env_name in "${!REWARDS_BY_ENV[@]}"; do
  echo "Model comparison for ${env_name}"
  env_name_sanitized=$(echo ${env_name} | sed -e 's/\//_/g')
  MODELS=$(find ${LEARNT_MODEL_DIR}/${env_name_sanitized} -name model -printf "%P\n" | xargs dirname)

  types=${REWARDS_BY_ENV[$env_name]}
  types_sanitized=$(echo ${types} | sed -e 's/\//_/g')

  echo "Comparing models to hardcoded rewards"
  echo "Models: ${MODELS}"
  echo "Hardcoded rewards: ${types}"

  parallel --header : --results ${OUTPUT_ROOT}/parallel/comparison/learnt/${env_name_sanitized} \
    ${TRAIN_CMD} env_name=${env_name} \
    source_reward_type=evaluating_rewards/RewardModel-v0 \
    source_reward_path=${LEARNT_MODEL_DIR}/${env_name_sanitized}/{source_reward}/model seed={seed} \
    target_reward_type={target_reward} {named_config} \
    log_dir=${OUTPUT_ROOT}/comparison/${MODEL_PREFIX}/${env_name_sanitized}/{source_reward}/match_{named_config}_to_{target_reward_sanitized}_seed{seed} \
    ::: source_reward ${MODELS} \
    ::: target_reward ${types} \
    :::+ target_reward_sanitized ${types_sanitized} \
    ::: named_config "" affine_only \
    ::: seed 0 1 2
done
