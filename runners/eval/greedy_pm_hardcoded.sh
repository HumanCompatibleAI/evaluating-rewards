#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/common.sh

GREEDY_REWARD_MODELS="PointMassGroundTruth-v0:None \
                      PointMassSparse-v0:None \
                      PointMassDense-v0:None"

parallel --header : --results $HOME/output/parallel/greedy_pm_hardcoded \
         ${EVAL_POLICY_CMD} policy_type=evaluating_rewards/MCGreedy-v0 \
         env_name={env}  policy_path={policy_path} \
         ::: env ${PM_ENVS} \
         ::: policy_path ${GREEDY_REWARD_MODELS}
