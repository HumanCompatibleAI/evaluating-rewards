#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/../common.sh

TRAIN_CMD=$(call_script "train_preferences" "with")

echo "Starting preference comparison"
for env_name in "${!REWARDS_BY_ENV[@]}"; do
  echo "Preference comparison for ${env_name}"
  types=${REWARDS_BY_ENV[$env_name]}
  env_name_sanitized=$(echo ${env_name} | sed -e 's/\//_/g')
  types_sanitized=$(echo ${types} | sed -e 's/\//_/g')

  parallel --header : --results $HOME/output/parallel/train_preferences/${env_name} \
           ${TRAIN_CMD} env_name=${env_name} \
           seed={seed} target_reward_type={target_reward} \
           log_dir=${HOME}/output/train_preferences/${env_name_sanitized}/{target_reward_sanitized}/{seed} \
           ::: target_reward ${types} \
           :::+ target_reward_sanitized ${types_sanitized} \
           ::: seed 0 1 2
done
