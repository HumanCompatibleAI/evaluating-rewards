#!/usr/bin/env bash

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
  parallel --header : --results $HOME/output/parallel/comparison/hardcoded_mujoco \
           ${TRAIN_CMD} env_name=${env_name} \
           seed={seed} \
           source_reward_type={source_reward_type} \
           target_reward_type={target_reward_type} \
           log_dir=${HOME}/output/comparison/hardcoded/${env_name_sanitized}/{source_reward_type_sanitized}_vs_{target_reward_type_sanitized}_seed{seed} \
           ::: source_reward_type ${types} \
           :::+ source_reward_type_sanitized ${types_sanitized} \
           ::: target_reward_type ${types} \
           :::+ target_reward_type_sanitized ${types_sanitized} \
           ::: seed 0 1 2
done
