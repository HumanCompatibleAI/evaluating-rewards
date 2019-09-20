#!/usr/bin/env bash


function call_script {
  script_name=$1
  shift
  echo "python -m evaluating_rewards.scripts.${script_name} $@"
}

PM_ENVS="evaluating_rewards/PointMassLine-v0 \
         evaluating_rewards/PointMassLineFixedHorizon-v0 \
         evaluating_rewards/PointMassGrid-v0"
ENVS=${PM_ENVS}
PM_REWARDS="PointMassSparse-v0 \
            PointMassDense-v0 \
            PointMassSparseNoCtrl-v0 \
            PointMassDenseNoCtrl-v0 \
            PointMassGroundTruth-v0 \
            Zero-v0"
OUTPUT_ROOT=$HOME/output