#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/../common.sh

EXPERT_DEMOS_CMD=$(call_script "expert_demos" "with")

parallel --header : --results $HOME/output/parallel/train_experts \
         ${EXPERT_DEMOS_CMD} env_name={env} seed={seed} \
         ::: env ${ENVS} \
         ::: seed 0 1 2
