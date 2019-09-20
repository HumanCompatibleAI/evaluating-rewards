#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/../common.sh

TRAIN_CMD=$(call_script "train_preferences" "with")

parallel --header : -j 50% --results $HOME/output/parallel/train_preferences_hardcoded \
         ${TRAIN_CMD} env_name=evaluating_rewards/PointMassLineFixedHorizon-v0 \
         seed={seed} target_reward_type=evaluating_rewards/{target_reward} \
         log_dir=${HOME}/output/train_preferences_hardcoded/{target_reward}/{seed} \
         ::: target_reward ${PM_REWARDS} \
         ::: seed 0 1 2
