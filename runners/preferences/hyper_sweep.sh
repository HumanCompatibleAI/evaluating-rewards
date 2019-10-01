#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/../common.sh

TRAIN_CMD=$(call_script "train_preferences" "with")

TARGET_REWARDS="
PointMassDense-v0
PointMassSparse-v0
"

parallel --header : --results $HOME/output/parallel/train_preferences_hyper \
         ${TRAIN_CMD} env_name=evaluating_rewards/PointMassLine-v0 \
         seed={seed} target_reward_type=evaluating_rewards/{target_reward} \
         batch_timesteps={batch_timesteps} trajectory_length={trajectory_length} \
         learning_rate={lr} total_timesteps=5e6 \
         log_dir=${HOME}/output/train_preferences_hyper/{target_reward}/batch{batch_timesteps}_of_{trajectory_length}_lr{lr}/{seed} \
         ::: target_reward ${TARGET_REWARDS} \
         ::: batch_timesteps 500 2500 10000 50000 250000 \
         ::: trajectory_length 1 5 25 100 \
         ::: lr 1e-4 1e-3 1e-2 1e-1 \
         ::: seed 0 1 2
