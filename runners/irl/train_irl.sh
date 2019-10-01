#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/../common.sh

TRAIN_CMD=$(call_script "train_adversarial" "with")

for env in ${ENVS}; do
  env_sanitized=$(echo ${env} | sed -e 's/\//_/g')
  parallel --header : --results $HOME/output/parallel/train_irl \
           ${TRAIN_CMD} env_name=${env} seed={seed} \
           rollout_path={data_path}/rollouts/final.pkl \
           ::: data_path $HOME/output/expert_demos/${env_sanitized}/* \
           ::: seed 0 1 2
done