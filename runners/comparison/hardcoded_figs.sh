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

ENVS="point_mass hopper half_cheetah"
DISCOUNT="0.99"

EPIC_CMD="python -m evaluating_rewards.analysis.dissimilarity_heatmaps.plot_epic_heatmap \
                    with discount=${DISCOUNT}"
NPEC_CMD="python -m evaluating_rewards.analysis.dissimilarity_heatmaps.plot_npec_heatmap with normalize"
ERC_CMD="python -m evaluating_rewards.analysis.dissimilarity_heatmaps.plot_erc_heatmap \
                    with discount=${DISCOUNT}"
COMMON_FLAGS="high_precision paper"

if [[ "${EVAL_OUTPUT_ROOT}" == "" ]]; then
  EVAL_OUTPUT_ROOT=$HOME/output
fi
LOG_ROOT=${EVAL_OUTPUT_ROOT}/hardcoded_figs

for env_name in ${ENVS}; do
  ${EPIC_CMD} ${COMMON_FLAGS} ${env_name} log_dir=${LOG_ROOT}/epic/${env_name}/&
  ${NPEC_CMD} ${COMMON_FLAGS} ${env_name} log_dir=${LOG_ROOT}/npec/${env_name}/&
  ${ERC_CMD} ${COMMON_FLAGS} ${env_name} log_dir=${LOG_ROOT}/erc/${env_name}/&
done

wait
echo "Figures generated to ${LOG_ROOT}."
echo "There are lots of different types of figures, you may want to extract a subset, e.g:"
echo 'rsync -rv --include="bootstrap_middle_all.pdf" --include="bootstrap_width_all.pdf" ${LOG_ROOT} <dest directory>'
