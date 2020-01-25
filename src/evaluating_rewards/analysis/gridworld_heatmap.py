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

"""Heatmaps for rewards in gridworld environments.

This is currently only used for illustrative examples in the paper;
none of the actual experiments are gridworlds."""

import math
from typing import Tuple

import matplotlib
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# (x,y) offset caused by taking an action
ACTION_DELTA = [
    (0, 0),  # no-op
    (-1, 0),  # left
    (0, 1),  # up
    (1, 0),  # right
    (0, -1),  # down
]

# Counter-clockwise, corners of a unit square, centred at (0.5, 0.5).
CORNERS = [(0, 0), (0, 1), (1, 1), (1, 0)]
# Vertices subdividing the unit square for each action
OFFSETS = {
    # Triangles, cutting unit-square into quarters
    direction: np.array([CORNERS[i], [0.5, 0.5], CORNERS[(i + 1) % len(CORNERS)]])
    for i, direction in enumerate(ACTION_DELTA[1:])
}
# Circle at the center
OFFSETS[(0, 0)] = np.array([0.5, 0.5])


def shape(state_reward: np.ndarray, state_potential: np.ndarray) -> np.ndarray:
    """Shape `state_reward` with `state_potential`.

    Args:
        state_reward: a two-dimensional array, indexed by `(i,j)`.
        state_potential: a two-dimensional array of the same shape as `state_reward`.

    Returns:
        A state-action reward `sa_reward`. This is three-dimensional array,
        indexed by `(i,j,a)`, where `a` is an action indexing into `DIRECTIONS`.
        `sa_reward[i,j,a] = state_reward[i, j] + state_potential[i', j'] - state_potential[i,j]`,
        where `i', j'` is the successor state after taking action `a`.
    """
    assert state_reward.ndim == 2
    assert state_reward.shape == state_potential.shape

    padded_reward = np.pad(
        state_reward.astype(np.float32), pad_width=[(1, 1), (1, 1)], constant_values=np.nan
    )
    padded_potential = np.pad(
        state_potential.astype(np.float32), pad_width=[(1, 1), (1, 1)], constant_values=np.nan
    )

    res = []
    for x_delta, y_delta in ACTION_DELTA:
        axis = 0 if x_delta else 1
        delta = x_delta + y_delta
        new_potential = np.roll(padded_potential, -delta, axis=axis)
        shaped = padded_reward + new_potential - padded_potential
        res.append(shaped[1:-1, 1:-1])

    return np.array(res).transpose((1, 2, 0))


def _set_ticks(n: int, subaxis: matplotlib.axis.Axis) -> None:
    subaxis.set_ticks(np.arange(0, n + 1), minor=True)
    subaxis.set_ticks(np.arange(n) + 0.5)
    subaxis.set_ticklabels(np.arange(n))


def _reward_make_fig(xlen: int, ylen: int) -> Tuple[plt.Figure, plt.Axes]:
    """Construct figure and set sensible defaults."""
    fig, ax = plt.subplots(1, 1)
    # Axes limits
    ax.set_xlim(0, xlen)
    ax.set_ylim(ylen, 0)
    # Make ticks centred in each cell
    _set_ticks(xlen, ax.xaxis)
    _set_ticks(ylen, ax.yaxis)
    # Draw grid along minor ticks, then remove those ticks so they don't protrude
    ax.grid(which="minor", color="k")
    ax.tick_params(which="minor", length=0, width=0)

    return fig, ax


def _reward_make_color_map(state_action_reward: np.ndarray) -> matplotlib.cm.ScalarMappable:
    norm = mcolors.Normalize(
        vmin=np.nanmin(state_action_reward), vmax=np.nanmax(state_action_reward)
    )
    return matplotlib.cm.ScalarMappable(norm=norm)


def _reward_draw_spline(
    x: int,
    y: int,
    action: int,
    reward: float,
    from_dest: bool,
    mappable: matplotlib.cm.ScalarMappable,
    annot_padding: float,
    ax: plt.Axes,
) -> Tuple[np.ndarray, Tuple[float, ...]]:
    # Compute shape position and color
    pos = np.array([x, y])
    direction = np.array(ACTION_DELTA[action])
    if from_dest:
        pos = pos + direction
        direction = -direction
    vert = pos + OFFSETS[tuple(direction)]
    color = mappable.to_rgba(reward)

    # Add annotation
    text = f"{reward:.0f}"
    lum = sns.utils.relative_luminance(color)
    text_color = ".15" if lum > 0.408 else "w"
    xy = pos + 0.5

    if tuple(direction) != (0, 0):
        xy = xy + annot_padding * direction
    ax.annotate(
        text, xy=xy, ha="center", va="center", color=text_color,
    )

    return vert, color


def _reward_draw(
    state_action_reward: np.ndarray,
    fig: plt.Figure,
    ax: plt.Axes,
    mappable: matplotlib.cm.ScalarMappable,
    from_dest: bool,
    annot_padding: float,
    edgecolors: str = "gray",
) -> None:
    triangle_verts = []
    triangle_facecolors = []
    circle_offsets = []
    circle_facecolors = []

    circle_area_pt = 200
    circle_radius_pt = math.sqrt(circle_area_pt / math.pi)
    circle_radius_in = circle_radius_pt / 72
    corner_display = ax.transData.transform([0.0, 0.0])
    circle_radius_display = fig.dpi_scale_trans.transform([circle_radius_in, 0])
    circle_radius_data = ax.transData.inverted().transform(corner_display + circle_radius_display)
    annot_padding = 0.25 + 0.5 * circle_radius_data[0]

    it = np.nditer(state_action_reward, flags=["multi_index"])
    while not it.finished:
        reward = it[0]
        x, y, action = it.multi_index
        it.iternext()

        if not np.isfinite(reward):
            assert action != 0
            continue

        vert, color = _reward_draw_spline(
            x, y, action, reward, from_dest, mappable, annot_padding, ax
        )

        if action == 0:  # no-op: circle at center
            circle_offsets.append(vert)
            circle_facecolors.append(color)
        else:  # action moved: triangle at edge
            triangle_verts.append(vert)
            triangle_facecolors.append(color)

    legal_poly = mcollections.PolyCollection(
        verts=triangle_verts, facecolors=triangle_facecolors, edgecolors=edgecolors
    )
    circles = mcollections.CircleCollection(
        sizes=[circle_area_pt] * len(circle_offsets),
        facecolors=circle_facecolors,
        edgecolors=edgecolors,
        offsets=circle_offsets,
        transOffset=ax.transData,
    )

    ax.add_collection(legal_poly)
    ax.add_collection(circles)


def plot_gridworld_reward(
    state_action_reward: np.ndarray,
    from_dest: bool = False,
    annot_padding: float = 0.33,
    cbar_format: str = "%.0f",
    cbar_fraction: float = 0.05,
) -> plt.Figure:
    """
    Plots a heatmap of reward for the gridworld.

    Args:
      - state_action_reward: a three-dimensional array specifying the gridworld reward.
      - from_dest: if True, the triangular wedges represent reward when arriving into this
        cell from the adjacent cell; if False, represent reward when leaving this cell into
        the adjacent cell.
      - annot_padding: a fraction of a supercell to offset the annotation from the centre.
      - cbar_fraction: the fraction of the axes the colorbar takes up.

    Returns:
        A heatmap consisting of a "supercell" for each state `(i,j)` in the original gridworld.
        This supercell contains a central circle, representing the no-op action reward and four
        triangular wedges, representing the left, up, right and down action rewards.
    """
    xlen, ylen, num_actions = state_action_reward.shape
    assert num_actions == len(ACTION_DELTA)
    fig, ax = _reward_make_fig(xlen, ylen)
    mappable = _reward_make_color_map(state_action_reward)
    _reward_draw(state_action_reward, fig, ax, mappable, from_dest, annot_padding)
    fig.colorbar(mappable, format=cbar_format, fraction=cbar_fraction)
    return fig
