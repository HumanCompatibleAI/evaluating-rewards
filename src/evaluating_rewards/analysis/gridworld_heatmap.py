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

import collections
import enum
import math
from typing import Tuple
from unittest import mock

import matplotlib
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mdptoolbox
import numpy as np
import seaborn as sns


class Actions(enum.IntEnum):
    STAY = 0
    LEFT = 1
    UP = 2
    RIGHT = 3
    DOWN = 4


# (x,y) offset caused by taking an action
ACTION_DELTA = {
    Actions.STAY: (0, 0),
    Actions.LEFT: (-1, 0),
    Actions.UP: (0, 1),
    Actions.RIGHT: (1, 0),
    Actions.DOWN: (0, -1),
}

# Counter-clockwise, corners of a unit square, centred at (0.5, 0.5).
CORNERS = [(0, 0), (0, 1), (1, 1), (1, 0)]
# Vertices subdividing the unit square for each action
OFFSETS = {
    # Triangles, cutting unit-square into quarters
    direction: np.array(
        [CORNERS[action.value - 1], [0.5, 0.5], CORNERS[action.value % len(CORNERS)]]
    )
    for action, direction in ACTION_DELTA.items()
}
# Circle at the center
OFFSETS[(0, 0)] = np.array([0.5, 0.5])


def shape(
    state_reward: np.ndarray, state_potential: np.ndarray, discount: float = 0.99
) -> np.ndarray:
    """Shape `state_reward` with `state_potential`.

    Args:
        state_reward: a two-dimensional array, indexed by `(i,j)`.
        state_potential: a two-dimensional array of the same shape as `state_reward`.
        discount: discount rate of MDP.

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
    for x_delta, y_delta in ACTION_DELTA.values():
        axis = 0 if x_delta else 1
        delta = x_delta + y_delta
        new_potential = discount * np.roll(padded_potential, -delta, axis=axis)
        shaped = padded_reward + new_potential - padded_potential
        res.append(shaped[1:-1, 1:-1])

    return np.array(res).transpose((1, 2, 0))


def _make_transitions(
    transitions: np.ndarray,
    low_action: int,
    high_action: int,
    states: np.ndarray,
    idx: np.ndarray,
    n: int,
) -> None:
    transitions[low_action, states[idx == 0], states[idx == 0]] = 1
    transitions[low_action, states[idx > 0], states[idx < n - 1]] = 1
    transitions[high_action, states[idx == n - 1], states[idx == n - 1]] = 1
    transitions[high_action, states[idx < n - 1], states[idx > 0]] = 1


def build_mdp(state_action_reward: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create transition matrix for deterministic gridworld and reshape reward."""
    xlen, ylen, na = state_action_reward.shape
    ns = xlen * ylen

    transitions = np.zeros((na, ns, ns))
    transitions[Actions.STAY.value, :, :] = np.eye(ns, ns)
    states = np.arange(ns)
    xs = states % xlen
    ys = states // ylen
    _make_transitions(transitions, Actions.LEFT.value, Actions.RIGHT.value, states, ys, ylen)
    _make_transitions(transitions, Actions.DOWN.value, Actions.UP.value, states, xs, xlen)

    reward = state_action_reward.copy()
    reward = reward.reshape(ns, na)
    # We use NaN for transitions that would go outside the gridworld.
    # But in above transition dynamics these are equivalent to stay, so rewrite.
    mask = np.isnan(reward)
    stay_reward = reward[:, Actions.STAY.value]
    stay_tiled = np.tile(stay_reward, (na, 1)).T
    reward[mask] = stay_tiled[mask]
    assert np.isfinite(reward).all()

    return transitions, reward


def _no_op_iter(*args, **kwargs):
    """Does nothing, workaround for bug in pymdptoolbox GH#32."""
    del args, kwargs


def compute_qvalues(state_action_reward: np.ndarray, discount: float) -> np.ndarray:
    """Computes the Q-values of `state_action_reward` under deterministic dynamics."""
    transitions, reward = build_mdp(state_action_reward)

    # TODO(adam): remove this workaround once GH pymdptoolbox #32 merged.
    with mock.patch("mdptoolbox.mdp.ValueIteration._boundIter", new=_no_op_iter):
        vi = mdptoolbox.mdp.ValueIteration(
            transitions=transitions, reward=reward, discount=discount
        )
        vi.run()
    q_values = reward + discount * (transitions * vi.V).sum(2).T
    return q_values


def optimal_mask(state_action_reward: np.ndarray, discount: float = 0.99) -> np.ndarray:
    """Computes the optimal actions for each state in `state_action_reward`."""
    q_values = compute_qvalues(state_action_reward, discount)
    best_q = q_values.max(axis=1)[:, np.newaxis]
    optimal_action = np.isclose(q_values, best_q)
    return optimal_action.reshape(state_action_reward.shape)


def _set_ticks(n: int, subaxis: matplotlib.axis.Axis) -> None:
    subaxis.set_ticks(np.arange(0, n + 1), minor=True)
    subaxis.set_ticks(np.arange(n) + 0.5)
    subaxis.set_ticklabels(np.arange(n))


def _reward_make_fig(xlen: int, ylen: int) -> Tuple[plt.Figure, plt.Axes]:
    """Construct figure and set sensible defaults."""
    fig, ax = plt.subplots(1, 1)
    # Axes limits
    ax.set_xlim(0, xlen)
    ax.set_ylim(0, ylen)
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
    optimal: bool,
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
    fontweight = "bold" if optimal else None
    ax.annotate(
        text, xy=xy, ha="center", va="center", color=text_color, fontweight=fontweight,
    )

    return vert, color


def _reward_draw(
    state_action_reward: np.ndarray,
    discount: float,
    fig: plt.Figure,
    ax: plt.Axes,
    mappable: matplotlib.cm.ScalarMappable,
    from_dest: bool,
    edgecolor: str = "gray",
    hatchcolor: str = "white",
) -> None:
    optimal_actions = optimal_mask(state_action_reward, discount)

    circle_area_pt = 200
    circle_radius_pt = math.sqrt(circle_area_pt / math.pi)
    circle_radius_in = circle_radius_pt / 72
    corner_display = ax.transData.transform([0.0, 0.0])
    circle_radius_display = fig.dpi_scale_trans.transform([circle_radius_in, 0])
    circle_radius_data = ax.transData.inverted().transform(corner_display + circle_radius_display)
    annot_padding = 0.25 + 0.5 * circle_radius_data[0]

    verts = collections.defaultdict(lambda: collections.defaultdict(list))
    colors = collections.defaultdict(lambda: collections.defaultdict(list))

    it = np.nditer(state_action_reward, flags=["multi_index"])
    while not it.finished:
        reward = it[0]
        x, y, action = it.multi_index
        optimal = optimal_actions[it.multi_index]
        it.iternext()

        if not np.isfinite(reward):
            assert action != 0
            continue

        vert, color = _reward_draw_spline(
            x, y, action, optimal, reward, from_dest, mappable, annot_padding, ax
        )

        geom = "circle" if action == 0 else "triangle"
        verts[geom][optimal].append(vert)
        colors[geom][optimal].append(color)

    circle_collections = []
    triangle_collections = []

    def _make_triangle(optimal, **kwargs):
        return mcollections.PolyCollection(
            verts=verts["triangle"][optimal], facecolors=colors["triangle"][optimal], **kwargs,
        )

    def _make_circle(optimal, **kwargs):
        circle_offsets = verts["circle"][optimal]
        return mcollections.CircleCollection(
            sizes=[circle_area_pt] * len(circle_offsets),
            facecolors=colors["circle"][optimal],
            offsets=circle_offsets,
            transOffset=ax.transData,
            **kwargs,
        )

    maker_collection_dict = {
        _make_triangle: triangle_collections,
        _make_circle: circle_collections,
    }

    for optimal in [False, True]:
        hatch = "xx" if optimal else None

        for maker_fn, cols in maker_collection_dict.items():
            cols.append(maker_fn(optimal, edgecolors=edgecolor))
            if hatch:  # draw the hatch using a different color
                cols.append(maker_fn(optimal, edgecolors=hatchcolor, linewidth=0, hatch=hatch))

    for cols in triangle_collections + circle_collections:
        ax.add_collection(cols)


def plot_gridworld_reward(
    state_action_reward: np.ndarray,
    discount: float = 0.99,
    from_dest: bool = False,
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
    _reward_draw(state_action_reward, discount, fig, ax, mappable, from_dest)
    fig.colorbar(mappable, format=cbar_format, fraction=cbar_fraction)
    return fig
