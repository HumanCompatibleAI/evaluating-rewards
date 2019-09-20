#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/common.sh

for env in ${ENVS}; do
  env_sanitized=$(echo ${env} | sed -e 's/\//_/g')
  reward_paths=$HOME/output/train_adversarial/${env_sanitized}/*/final/discrim/reward_net
  policy_paths=""
  for rew_path in ${reward_paths}; do
    policy_paths="${policy_paths} BasicShapedRewardNet_shaped:${rew_path}"
    policy_paths="${policy_paths} BasicShapedRewardNet_unshaped:${rew_path}"
  done
  parallel --header : --results $HOME/output/parallel/greedy_pm_irl \
           ${EVAL_POLICY_CMD} env_name=${env} policy_type=evaluating_rewards/MCGreedy-v0 \
           policy_path={policy_path} \
           ::: policy_path ${policy_paths}
done