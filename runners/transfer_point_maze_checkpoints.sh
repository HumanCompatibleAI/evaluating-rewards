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

# Run `runners/transfer_point_maze.sh` before this script to create the reward models.

for shard_num in {0..9}; do
  echo "Running shard ${shard_num} of 10"
  python -m evaluating_rewards.scripts.pipeline.combined_distances with point_maze_checkpoints high_precision \
      "named_configs.point_maze_learned.global=('point_maze_checkpoints_100_${shard_num}of10',)" \
      log_dir=${PM_OUTPUT}/distances_checkpoints/${i}
done
