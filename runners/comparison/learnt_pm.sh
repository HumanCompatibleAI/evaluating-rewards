#!/usr/bin/env bash

# Compare hardcoded rewards in PointMass to each other

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/../common.sh

TRAIN_CMD=$(call_script "model_comparison" "with")


if [[ $# -ne 1 ]]; then
  echo "usage: $0 <model prefix>"
  echo "model prefix must be relative to ${OUTPUT_ROOT}"
  exit 1
fi

MODEL_PREFIX=$1
LEARNT_MODEL_DIR=${OUTPUT_ROOT}/${MODEL_PREFIX}

MODELS=$(find ${LEARNT_MODEL_DIR} -name model -printf "%P\n" | xargs dirname)

echo "Visualizing models:"
echo ${MODELS}

parallel --header : --results ${OUTPUT_ROOT}/parallel/comparison/learnt \
    ${TRAIN_CMD} env_name=evaluating_rewards/PointMassLineFixedHorizon-v0 \
    source_reward_type=evaluating_rewards/RewardModel-v0 \
    seed={seed} source_reward_path=${LEARNT_MODEL_DIR}/{source_reward}/model \
    target_reward_type=evaluating_rewards/{target_reward} {named_config} \
    log_dir=${OUTPUT_ROOT}/comparison/${MODEL_PREFIX}/{source_reward}/match_{named_config}_to_{target_reward}_seed{seed} \
    ::: source_reward ${MODELS} \
    ::: target_reward ${PM_REWARDS} \
    ::: named_config "" affine_only \
    ::: seed 0 1 2
