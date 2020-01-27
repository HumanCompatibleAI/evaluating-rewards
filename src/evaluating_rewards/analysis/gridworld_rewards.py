"""Illustrative rewards for gridworlds."""

import numpy as np

SPARSE_GOAL = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])

OBSTACLE_COURSE = np.array([[0, -1, -1], [0, 0, 0], [-1, -1, 5]])

CLIFF_WALK = np.array([[0, -1, -1], [0, 0, 0], [-9, -9, 5]])

MANHATTAN_FROM_GOAL = np.array([[4, 3, 2], [3, 2, 1], [2, 1, 0]])

ZERO = np.zeros((3, 3))

REWARDS = {
    # Equivalent rewards
    "sparse_goal": {"state_reward": SPARSE_GOAL, "potential": ZERO},
    "sparse_goal_shift": {"state_reward": SPARSE_GOAL + 1, "potential": ZERO},
    "sparse_goal_scale": {"state_reward": SPARSE_GOAL * 10, "potential": ZERO},
    "dense_goal": {"state_reward": SPARSE_GOAL, "potential": -MANHATTAN_FROM_GOAL},
    "antidense_goal": {"state_reward": SPARSE_GOAL, "potential": MANHATTAN_FROM_GOAL},
    # Non-equivalent rewards
    "transformed_goal": {
        # Shifted, rescaled and reshaped sparse goal.
        "state_reward": SPARSE_GOAL * 10 - 1,
        "potential": -MANHATTAN_FROM_GOAL * 10,
    },
    "dirt_path": {
        # Some minor penalties to avoid to reach goal.
        #
        # Optimal policy for this is optimal in `SPARSE_GOAL`, but not equivalent.
        # Think may come apart in some dynamics but not particularly intuitively.
        "state_reward": OBSTACLE_COURSE,
        "potential": ZERO,
    },
    "cliff_walk": {
        # Avoid cliff to reach goal. Same set of optimal policies as `obstacle_course` in
        # deterministic dynamics, but not equivalent.
        #
        # Optimal policy differs in sufficiently slippery gridworlds as want to stay on top line
        # to avoid chance of falling off cliff.
        "state_reward": CLIFF_WALK,
        "potential": ZERO,
    },
    "sparse_penalty": {
        # Negative of `sparse_goal`.
        "state_reward": -SPARSE_GOAL,
        "potential": ZERO,
    },
    "all_zero": {
        # All zero reward function
        "state_reward": ZERO,
        "potential": ZERO,
    },
}
