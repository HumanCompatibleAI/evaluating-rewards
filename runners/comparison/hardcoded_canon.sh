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

# Compares hardcoded rewards to each other using CANON

ENVS="point_mass hopper half_cheetah"
DISCOUNTS="0.9 0.99 1.0"

# Datasets?
# Default: Sample from Gym, IID
# Random policy, IID
# Random policy, random batch
# PointMass only: random transition, IID; random transition, random transition batch
for env in ${ENVS}; do
  for discount in ${DISCOUNTS}; do
    for distance_kind in direct pearson; do
      for computation_kind in sample mesh; do

         BASE_CMD="python -m evaluating_rewards.analysis.dissimilarity_heatmaps.plot_canon_heatmap \
                   with ${env} discount=${discount} distance_kind=${distance_kind} \
                   computation_kind=${computation_kind}"

         ${BASE_CMD}
         ${BASE_CMD} sample_from_serialized_policy
         ${BASE_CMD} sample_from_serialized_policy dataset_from_serialized_policy

         if [[ ${env} == "point_mass" ]]; then
           ${BASE_CMD} sample_from_random_transitions
           ${BASE_CMD} sample_from_random_transitions dataset_from_random_transitions
         fi
      done;
    done;
  done;
done
