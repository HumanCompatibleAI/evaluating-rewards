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

"""CLI script to visualize reward models for deterministic gridworlds."""

import os
from typing import Iterable

from imitation.util import util
import matplotlib.pyplot as plt
import numpy as np
import sacred

from evaluating_rewards import serialize
from evaluating_rewards.analysis import gridworld_heatmap, stylesheets, visualize
from evaluating_rewards.scripts import script_utils

plot_gridworld_heatmap_ex = sacred.Experiment("plot_gridworld_heatmap")


@plot_gridworld_heatmap_ex.config
def default_config():
    """Default configuration values."""
    # Reward parameters
    exp_name = "default"
    state_reward = np.zeros((3, 3))
    potential = np.zeros((3, 3))

    # Figure parameters
    log_root = os.path.join(serialize.get_output_dir(), "plot_gridworld_heatmap")
    styles = ["paper", "gridworld-heatmap", "gridworld-heatmap-1col-narrow"]
    fmt = "pdf"  # file type
    _ = locals()  # quieten flake8 unused variable warning
    del _


@plot_gridworld_heatmap_ex.config
def logging_config(log_root, exp_name):
    log_dir = os.path.join(  # noqa: F841  pylint:disable=unused-variable
        log_root, exp_name, util.make_unique_timestamp(),
    )


@plot_gridworld_heatmap_ex.named_config
def test():
    """Small config, intended for tests / debugging."""
    # Don't use TeX for tests
    styles = ["paper", "gridworld-heatmap", "gridworld-heatmap-1col"]
    _ = locals()
    del _


SPARSE_GOAL = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])

OBSTACLE_COURSE = np.array([[0, -1, -1], [0, 0, 0], [-1, -1, 10]])

CLIFF_WALK = np.array([[0, -1, -1], [0, 0, 0], [-9, -9, 1]])

MANHATTAN_FROM_GOAL = np.array([[4, 3, 2], [3, 2, 1], [2, 1, 0]])


# Goal oriented and *equivalent* rewards
@plot_gridworld_heatmap_ex.named_config
def sparse_goal():
    exp_name = "sparse_goal"
    state_reward = SPARSE_GOAL
    _ = locals()
    del _


@plot_gridworld_heatmap_ex.named_config
def sparse_goal_shift():
    exp_name = "sparse_goal_shift"
    state_reward = SPARSE_GOAL + 1
    _ = locals()
    del _


@plot_gridworld_heatmap_ex.named_config
def sparse_goal_scale():
    exp_name = "sparse_goal_scale"
    state_reward = SPARSE_GOAL * 10
    _ = locals()
    del _


@plot_gridworld_heatmap_ex.named_config
def dense_goal():
    exp_name = "dense_goal"
    state_reward = SPARSE_GOAL
    potential = -MANHATTAN_FROM_GOAL
    _ = locals()
    del _


@plot_gridworld_heatmap_ex.named_config
def antidense_goal():
    exp_name = "antidense_goal"
    state_reward = SPARSE_GOAL
    potential = MANHATTAN_FROM_GOAL
    _ = locals()
    del _


@plot_gridworld_heatmap_ex.named_config
def transformed_goal():
    """Shifted, rescaled and reshaped sparse goal."""
    exp_name = "transformed_goal"
    state_reward = SPARSE_GOAL * 10 - 1
    potential = -MANHATTAN_FROM_GOAL * 10
    _ = locals()
    del _


# Non-equivalent rewards
@plot_gridworld_heatmap_ex.named_config
def obstacle_course():
    """Some minor penalties to avoid to reach goal.

    Optimal policy for this is optimal in `SPARSE_GOAL`, but not equivalent.
    Think may come apart in some dynamics but not particularly intuitively.
    """
    exp_name = "obstacle_course"
    state_reward = OBSTACLE_COURSE
    _ = locals()
    del _


@plot_gridworld_heatmap_ex.named_config
def cliff_walk():
    """Avoid cliff to reach goal.

    Same set of optimal policies as `obstacle_course` in deterministic dynamics, but not equivalent.

    Optimal policy differs in sufficiently slippery gridworlds as want to stay on top line
    to avoid chance of falling off cliff.
    """
    exp_name = "cliff_walk"
    state_reward = CLIFF_WALK
    _ = locals()
    del _


@plot_gridworld_heatmap_ex.named_config
def sparse_anti_goal():
    """Negative of `sparse_goal`."""
    exp_name = "sparse_anti_goal"
    state_reward = -SPARSE_GOAL
    _ = locals()
    del _


@plot_gridworld_heatmap_ex.named_config
def all_zero():
    """All zero reward function."""
    # default state_reward and potential_reward are zero, nothing more to do
    exp_name = "all_zero"  # noqa: F841  pylint:disable=unused-variable


def _normalize(state_array: np.ndarray) -> np.ndarray:
    """Tranposes and flips array.

    Matrix indexing convention (i,j) corresponds to row i, column j from top-left.
    By contrast, function plotting convention is (x,y) is column x, row y from bottom-left.
    We follow function plotting convention for visualizing gridworlds.
    Transform matrix (i,j)->(j,m-i) where m is the number of rows.
    """
    return state_array.T[:, ::-1]


@plot_gridworld_heatmap_ex.main
def plot_gridworld_heatmap(
    state_reward: np.ndarray, potential: np.ndarray, styles: Iterable[str], log_dir: str, fmt: str,
) -> plt.Figure:
    """Plots a heatmap of a reward for the gridworld.

    Args:
        - state_reward: a dict containing the name of the reward and a 2D array.
        - potential: a dict containing the name of the potential and a 2D array.
        - styles: styles defined in `stylesheets` to apply.
        - log_dir: the directory to save the figure in.
        - fmt: the format to save the figure in.

    Returns:
        The generated figure.
    """
    state_action_reward = gridworld_heatmap.shape(_normalize(state_reward), _normalize(potential))
    with stylesheets.setup_styles(styles):
        fig = gridworld_heatmap.plot_gridworld_reward(state_action_reward)
        visualize.save_fig(os.path.join(log_dir, "fig"), fig, fmt, transparent=False)

    return fig


if __name__ == "__main__":
    script_utils.experiment_main(plot_gridworld_heatmap_ex, "plot_gridworld_heatmap")
