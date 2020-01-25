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
from typing import Any, Dict, Iterable

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
    state_reward = {
        "name": "zero",
        "array": np.zeros((3, 3)),
    }
    potential = {
        "name": "zero",
        "array": np.zeros((3, 3)),
    }

    # Figure parameters
    log_root = os.path.join(serialize.get_output_dir(), "plot_gridworld_heatmap")
    styles = ["paper", "gridworld-heatmap-1col", "tex"]
    fmt = "pdf"  # file type
    _ = locals()  # quieten flake8 unused variable warning
    del _


@plot_gridworld_heatmap_ex.config
def logging_config(log_root, state_reward, potential):
    log_dir = os.path.join(  # noqa: F841  pylint:disable=unused-variable
        log_root, f"{state_reward['name']}_{potential['name']}", util.make_unique_timestamp(),
    )


@plot_gridworld_heatmap_ex.named_config
def test():
    """Small config, intended for tests / debugging."""
    # Don't use TeX for tests
    styles = ["paper", "gridworld-heatmap-1col"]  # noqa: F841  pylint:disable=unused-variable


@plot_gridworld_heatmap_ex.main
def plot_gridworld_heatmap(
    state_reward: Dict[str, Any],
    potential: Dict[str, Any],
    styles: Iterable[str],
    log_dir: str,
    fmt: str,
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
    state_action_reward = gridworld_heatmap.shape(state_reward["array"], potential["array"])
    with stylesheets.setup_styles(styles):
        fig = gridworld_heatmap.plot_gridworld_reward(state_action_reward)
        visualize.save_fig(os.path.join(log_dir, "fig"), fig, fmt)

    return fig


if __name__ == "__main__":
    script_utils.experiment_main(plot_gridworld_heatmap_ex, "plot_gridworld_heatmap")
