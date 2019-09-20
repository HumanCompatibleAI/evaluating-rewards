#!/usr/bin/env bash

# Compare hardcoded rewards in PointMass to each other

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/../common.sh

TRAIN_CMD=$(call_script "model_comparison" "with")

parallel --header : --results $HOME/output/parallel/comparison/hardcoded \
         ${TRAIN_CMD} env_name=evaluating_rewards/PointMassLineFixedHorizon-v0 \
         seed={seed} source_reward_type=evaluating_rewards/{source_reward} \
         target_reward_type=evaluating_rewards/{target_reward} \
         log_dir=${HOME}/output/comparison/hardcoded/{source_reward}_vs_{target_reward}_seed{seed} \
         ::: source_reward ${PM_REWARDS} \
         ::: target_reward ${PM_REWARDS} \
         ::: seed 0 1 2
