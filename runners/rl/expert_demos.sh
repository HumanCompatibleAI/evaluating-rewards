#!/usr/bin/env bash
# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. ${DIR}/../common.sh

EXPERT_DEMOS_CMD=$(call_script "expert_demos" "with")

for env_name in ${ENVS}; do
  types=${REWARDS_BY_ENV[$env_name]}
  parallel --header : --results $HOME/output/parallel/expert_demos \
           ${EXPERT_DEMOS_CMD} env_name=${env_name} \
           reward_type={type} seed={seed} \
           ::: type ${types} \
           ::: seed 0 1 2
done
