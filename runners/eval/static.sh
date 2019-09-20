#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/common.sh

POLICY_TYPES="random zero"

parallel --header : --results $HOME/output/parallel/static \
         ${EVAL_POLICY_CMD} env_name={env} policy_type={policy_type} \
         ::: env ${ENVS} \
         ::: policy_type ${POLICY_TYPES}
