#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/../common.sh

EVAL_POLICY_CMD=$(call_script "eval_policy" "with render=False num_vec=8")
