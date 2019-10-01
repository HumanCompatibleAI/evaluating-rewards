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

TRAIN_CMD=$(call_script "train_adversarial" "with")

for env in ${ENVS}; do
  env_sanitized=$(echo ${env} | sed -e 's/\//_/g')
  parallel --header : --results $HOME/output/parallel/train_irl \
           ${TRAIN_CMD} env_name=${env} seed={seed} \
           rollout_path={data_path}/rollouts/final.pkl \
           ::: data_path $HOME/output/expert_demos/${env_sanitized}/* \
           ::: seed 0 1 2
done