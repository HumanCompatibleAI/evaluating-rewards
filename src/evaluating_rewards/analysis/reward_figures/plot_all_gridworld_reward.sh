#!/bin/bash
# Copyright 2020 Adam Gleave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CONFIGS="sparse_goal sparse_goal_shift sparse_goal_scale \
         dense_goal antidense_goal transformed_goal center_goal \
         dirt_path cliff_walk sparse_penalty all_zero"

parallel --header : python -m evaluating_rewards.analysis.plot_gridworld_reward \
          with {config} \
          ::: config ${CONFIGS}
