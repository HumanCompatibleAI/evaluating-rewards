#!/usr/bin/env bash
# Copyright 2020 Adam Gleave
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

# Plots heatmap of reward distances for all hard-coded rewards

ENVS="point_mass half_cheetah hopper"
DISCOUNT="0.99"

EPIC_CMD="python -m evaluating_rewards.analysis.dissimilarity_heatmaps.plot_epic_heatmap \
                    with discount=${DISCOUNT}"
NPEC_CMD="python -m evaluating_rewards.analysis.dissimilarity_heatmaps.plot_npec_heatmap \
          with normalize_distance"
# Override env_name since we change environment name since running experiment
# TODO(adam): rerun experiments and remove this backward compatibility code
NPEC_EXTRA_FLAGS=(
  ""
  "env_name=evaluating_rewards/HalfCheetah-v3"
  "env_name=evaluating_rewards/Hopper-v3"
)
ERC_CMD="python -m evaluating_rewards.analysis.dissimilarity_heatmaps.plot_erc_heatmap \
                    with discount=${DISCOUNT}"
COMMON_FLAGS="high_precision paper"

if [[ "${EVAL_OUTPUT_ROOT}" == "" ]]; then
  EVAL_OUTPUT_ROOT=$HOME/output
fi
LOG_ROOT=${EVAL_OUTPUT_ROOT}/hardcoded_figs

i=0
for env_name in ${ENVS}; do
  ${EPIC_CMD} ${COMMON_FLAGS} ${env_name} log_dir=${LOG_ROOT}/epic/${env_name}/&
  ${NPEC_CMD} ${COMMON_FLAGS} ${env_name} log_dir=${LOG_ROOT}/npec/${env_name}/ ${NPEC_EXTRA_FLAGS[$i]}&
  ${ERC_CMD} ${COMMON_FLAGS} ${env_name} log_dir=${LOG_ROOT}/erc/${env_name}/&
  i=$((i + 1))
done

wait
echo "Figures generated to ${LOG_ROOT}."
echo "There are lots of different types of figures, you may want to extract a subset, e.g:"
echo -n 'rsync -rvm --include="*/" --include="bootstrap_middle_all.pdf" --include="middle_all.pdf" --include="bootstrap_width_all.pdf" --include="width_all.pdf" --exclude="*"'
echo "${LOG_ROOT}/ <dest_directory>"
