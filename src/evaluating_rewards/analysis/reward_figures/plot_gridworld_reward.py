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
from typing import Iterable, Optional

from imitation.util import util
import matplotlib.pyplot as plt
import numpy as np
import sacred

from evaluating_rewards import serialize
from evaluating_rewards.analysis import gridworld_rewards, stylesheets, visualize
from evaluating_rewards.analysis.reward_figures import gridworld_reward_heatmap
from evaluating_rewards.scripts import script_utils

plot_gridworld_reward_ex = sacred.Experiment("plot_gridworld_reward")


@plot_gridworld_reward_ex.config
def default_config():
    """Default configuration values."""
    # Reward parameters
    exp_name = "default"
    state_reward = np.zeros((3, 3))
    potential = np.zeros((3, 3))
    discount = 0.99

    # Figure parameters
    vmin = -5.0
    vmax = 5.0
    log_root = os.path.join(serialize.get_output_dir(), "plot_gridworld_reward")
    styles = ["paper", "gridworld-heatmap", "gridworld-heatmap-1col-narrow"]
    fmt = "pdf"  # file type
    _ = locals()  # quieten flake8 unused variable warning
    del _


@plot_gridworld_reward_ex.config
def logging_config(log_root, exp_name):
    log_dir = os.path.join(  # noqa: F841  pylint:disable=unused-variable
        log_root, exp_name, util.make_unique_timestamp(),
    )


@plot_gridworld_reward_ex.named_config
def test():
    """Small config, intended for tests / debugging."""
    # No changes from default, but we need this present for unit tests.
    pass  # pylint:disable=unnecessary-pass


def _normalize(state_array: np.ndarray) -> np.ndarray:
    """Tranposes and flips array.

    Matrix indexing convention (i,j) corresponds to row i, column j from top-left.
    By contrast, function plotting convention is (x,y) is column x, row y from bottom-left.
    We follow function plotting convention for visualizing gridworlds.
    Transform matrix (i,j)->(j,m-i) where m is the number of rows.
    """
    return state_array.T[:, ::-1]


def _add_rewards() -> None:
    for exp_name, cfg in gridworld_rewards.REWARDS.items():
        cfg = cfg.copy()
        cfg["exp_name"] = exp_name
        plot_gridworld_reward_ex.add_named_config(exp_name, cfg)


_add_rewards()


@plot_gridworld_reward_ex.main
def plot_gridworld_reward(
    state_reward: np.ndarray,
    potential: np.ndarray,
    discount: float,
    styles: Iterable[str],
    log_dir: str,
    fmt: str,
    vmin: Optional[float],
    vmax: Optional[float],
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
    state_action_reward = gridworld_reward_heatmap.shape(
        _normalize(state_reward), _normalize(potential), discount
    )
    with stylesheets.setup_styles(styles):
        fig = gridworld_reward_heatmap.plot_gridworld_reward(
            state_action_reward, discount=discount, vmin=vmin, vmax=vmax
        )
        visualize.save_fig(os.path.join(log_dir, "fig"), fig, fmt, transparent=False)

    return fig


if __name__ == "__main__":
    script_utils.experiment_main(plot_gridworld_reward_ex, "plot_gridworld_reward")
