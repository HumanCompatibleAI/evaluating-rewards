#!/bin/bash

CONFIGS="sparse_goal sparse_goal_shift sparse_goal_scale \
         dense_goal antidense_goal transformed_goal \
         dirt_path cliff_walk sparse_penalty all_zero"

parallel --header : python -m evaluating_rewards.analysis.plot_gridworld_reward \
          with {config} \
          ::: config ${CONFIGS}
