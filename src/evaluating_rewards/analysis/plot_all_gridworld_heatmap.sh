#!/bin/bash

CONFIGS="sparse_goal sparse_goal_shift sparse_goal_scale \
         dense_goal antidense_goal transformed_goal \
         obstacle_course cliff_walk sparse_anti_goal all_zero"

parallel --header : python -m evaluating_rewards.analysis.plot_gridworld_heatmap \
          with {config} \
          ::: config ${CONFIGS}
