#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/../common.sh

TRAIN_CMD=$(call_script "train_regress" "with")

TARGET_REWARDS="
PointMassSparse-v0
PointMassDense-v0
PointMassSparseNoCtrl-v0
PointMassDenseNoCtrl-v0
PointMassGroundTruth-v0
"

parallel --header : --results $HOME/output/parallel/train_regress/hardcoded_pm \
         ${TRAIN_CMD} env_name=evaluating_rewards/PointMassLineFixedHorizon-v0 \
         seed={seed} target_reward_type=evaluating_rewards/{target_reward} \
         log_dir=${HOME}/output/train_regress/hardcoded_pm/{target_reward}/{seed} \
         ::: target_reward ${TARGET_REWARDS} \
         ::: seed 0 1 2
