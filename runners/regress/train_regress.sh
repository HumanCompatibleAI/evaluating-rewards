#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/../common.sh

TRAIN_CMD=$(call_script "train_regress" "with")

echo "Starting model regression"
for env_name in "${!REWARDS_BY_ENV[@]}"; do
  echo "Model regression for ${env_name}"
  types=${REWARDS_BY_ENV[$env_name]}
  env_name_sanitized=$(echo ${env_name} | sed -e 's/\//_/g')
  types_sanitized=$(echo ${types} | sed -e 's/\//_/g')

  parallel --header : --results $HOME/output/parallel/train_regress/${env_name_sanitized} \
         ${TRAIN_CMD} env_name=${env_name} \
         seed={seed} target_reward_type={target_reward} \
         log_dir=${HOME}/output/train_regress/${env_name_sanitized}/{target_reward_sanitized}/{seed} \
         ::: target_reward ${types} \
         :::+ target_reward_sanitized ${types_sanitized} \
         ::: seed 0 1 2
done
