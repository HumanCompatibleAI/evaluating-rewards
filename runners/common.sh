#!/usr/bin/env bash


function call_script {
  script_name=$1
  shift
  echo "python -m evaluating_rewards.scripts.${script_name} $@"
}

PM_ENVS="evaluating_rewards/PointMassLine-v0 \
         evaluating_rewards/PointMassLineVariableHorizon-v0 \
         evaluating_rewards/PointMassGrid-v0"

ENV_REWARD_CMD=$(call_script "env_rewards")
# This declares REWARDS_BY_ENV
echo "Loading environment to reward mapping"
eval "$(${ENV_REWARD_CMD} 2>/dev/null)"
ENVS="${!REWARDS_BY_ENV[@]}"
echo "Loaded mappings for environments ${ENVS}"

OUTPUT_ROOT=$HOME/output