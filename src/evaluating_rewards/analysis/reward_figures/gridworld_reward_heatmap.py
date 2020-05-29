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

import enum
import functools
from typing import Iterable, Mapping, Optional, Tuple
from unittest import mock

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
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


def build_transitions(xlen: int, ylen: int, na: int) -> np.ndarray:
    """Create transition matrix for deterministic gridworld."""
    ns = xlen * ylen
    transitions = np.zeros((na, ns, ns))
    transitions[Actions.STAY.value, :, :] = np.eye(ns, ns)
    states = np.arange(ns)
    xs = states % xlen
    ys = states // ylen
    _make_transitions(transitions, Actions.LEFT.value, Actions.RIGHT.value, states, ys, ylen)
    _make_transitions(transitions, Actions.DOWN.value, Actions.UP.value, states, xs, xlen)

    return transitions


def build_reward(state_action_reward: np.ndarray) -> np.ndarray:
    """Reshape reward and fill in NaNs."""
    xlen, ylen, na = state_action_reward.shape
    ns = xlen * ylen
    reward = state_action_reward.copy()
    reward = reward.reshape(ns, na)

    # We use NaN for transitions that would go outside the gridworld.
    # But in above transition dynamics these are equivalent to stay, so rewrite.
    mask = np.isnan(reward)
    stay_reward = reward[:, Actions.STAY.value]
    stay_tiled = np.tile(stay_reward, (na, 1)).T
    reward[mask] = stay_tiled[mask]
    assert np.isfinite(reward).all()

    return reward


def _no_op_iter(*args, **kwargs):
    """Does nothing, workaround for bug in pymdptoolbox GH#32."""
    del args, kwargs


def compute_qvalues(state_action_reward: np.ndarray, discount: float) -> np.ndarray:
    """Computes the Q-values of `state_action_reward` under deterministic dynamics."""
    transitions = build_transitions(*state_action_reward.shape)
    reward = build_reward(state_action_reward)

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


def _axis_formatting(ax: plt.Axes, xlen: int, ylen: int) -> None:
    """Construct figure and set sensible defaults."""
    # Axes limits
    ax.set_xlim(0, xlen)
    ax.set_ylim(0, ylen)
    # Make ticks centred in each cell
    _set_ticks(xlen, ax.xaxis)
    _set_ticks(ylen, ax.yaxis)
    # Draw grid along minor ticks, then remove those ticks so they don't protrude
    ax.grid(which="minor", color="k")
    ax.tick_params(which="minor", length=0, width=0)


def _reward_make_color_map(
    state_action_reward: Iterable[np.ndarray], vmin: Optional[float], vmax: Optional[float]
) -> matplotlib.cm.ScalarMappable:
    if vmin is None:
        vmin = min(np.nanmin(arr) for arr in state_action_reward)
    if vmax is None:
        vmax = max(np.nanmax(arr) for arr in state_action_reward)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
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
) -> Tuple[np.ndarray, Tuple[float, ...], str]:
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
    hatch_color = ".5" if lum > 0.408 else "w"
    xy = pos + 0.5

    if tuple(direction) != (0, 0):
        xy = xy + annot_padding * direction
    fontweight = "bold" if optimal else None
    ax.annotate(
        text, xy=xy, ha="center", va="center_baseline", color=text_color, fontweight=fontweight,
    )

    return vert, color, hatch_color


def _make_triangle(vert, color, **kwargs):
    return mpatches.Polygon(xy=vert, facecolor=color, **kwargs)


def _make_circle(vert, color, radius, **kwargs):
    return mpatches.Circle(xy=vert, radius=radius, facecolor=color, **kwargs)


def _reward_draw(
    state_action_reward: np.ndarray,
    discount: float,
    fig: plt.Figure,
    ax: plt.Axes,
    mappable: matplotlib.cm.ScalarMappable,
    from_dest: bool = False,
    edgecolor: str = "gray",
) -> None:
    """
    Args:
        state_action_reward: a three-dimensional array specifying the gridworld rewards.
        discount: MDP discount rate.
        fig: figure to plot on.
        ax: the axis on the figure to plot on.
        mappable: color map for heatmap.
        from_dest: if True, the triangular wedges represent reward when arriving into this
        cell from the adjacent cell; if False, represent reward when leaving this cell into
        the adjacent cell.
        edgecolor: color of edges.
    """
    optimal_actions = optimal_mask(state_action_reward, discount)

    circle_radius_pt = matplotlib.rcParams.get("font.size") * 0.7
    circle_radius_in = circle_radius_pt / 72
    corner_display = ax.transData.transform([0.0, 0.0])
    circle_radius_display = fig.dpi_scale_trans.transform([circle_radius_in, 0])
    circle_radius_data = ax.transData.inverted().transform(corner_display + circle_radius_display)
    annot_padding = 0.25 + 0.5 * circle_radius_data[0]

    triangle_patches = []
    circle_patches = []

    it = np.nditer(state_action_reward, flags=["multi_index"])
    while not it.finished:
        reward = it[0]
        x, y, action = it.multi_index
        optimal = optimal_actions[it.multi_index]
        it.iternext()

        if not np.isfinite(reward):
            assert action != 0
            continue

        vert, color, hatch_color = _reward_draw_spline(
            x, y, action, optimal, reward, from_dest, mappable, annot_padding, ax
        )

        hatch = "xx" if optimal else None
        if action == 0:
            fn = functools.partial(_make_circle, radius=circle_radius_data[0])
        else:
            fn = _make_triangle
        patches = circle_patches if action == 0 else triangle_patches
        if hatch:  # draw the hatch using a different color
            patches.append(fn(vert, tuple(color), linewidth=1, edgecolor=hatch_color, hatch=hatch))
            patches.append(fn(vert, tuple(color), linewidth=1, edgecolor=edgecolor, fill=False))
        else:
            patches.append(fn(vert, tuple(color), linewidth=1, edgecolor=edgecolor))

    for p in triangle_patches + circle_patches:
        # need to draw circles on top of triangles
        ax.add_patch(p)


def plot_gridworld_rewards(
    reward_arrays: Mapping[str, np.ndarray],
    ncols: int,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cbar_format: str = "%.0f",
    cbar_fraction: float = 0.05,
    **kwargs,
) -> plt.Figure:
    """
    Plots heatmaps of reward for the gridworld.

    Args:
      - reward_arrays: a mapping to three-dimensional arrays specifying the gridworld rewards.
      - **kwargs: passed through to `_reward_draw`.

    Returns:
        A heatmap consisting of a "supercell" for each state `(i,j)` in the original gridworld.
        This supercell contains a central circle, representing the no-op action reward and four
        triangular wedges, representing the left, up, right and down action rewards.
    """
    shapes = set((v.shape for v in reward_arrays.values()))
    assert len(shapes) == 1, "different shaped gridworlds cannot be in same plot"
    xlen, ylen, num_actions = next(iter(shapes))
    assert num_actions == len(ACTION_DELTA)

    nplots = len(reward_arrays)
    nrows = (nplots - 1) // ncols + 1
    width, height = matplotlib.rcParams.get("figure.figsize")
    fig = plt.figure(figsize=(width, height * nrows))
    width_ratios = [1] * ncols + [cbar_fraction]
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols + 1, width_ratios=width_ratios)

    mappable = _reward_make_color_map(reward_arrays.values(), vmin, vmax)
    base_ax = fig.add_subplot(gs[0, 0])
    for idx, (pretty_name, reward) in enumerate(reward_arrays.items()):
        i = idx // ncols
        j = idx % ncols
        if i == 0 and j == 0:
            ax = base_ax
        else:
            ax = fig.add_subplot(gs[i, j], sharex=base_ax, sharey=base_ax)

        _axis_formatting(ax, xlen, ylen)
        if not ax.is_last_row():
            ax.tick_params(axis="x", labelbottom=False)
        if not ax.is_first_col():
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(pretty_name)

        _reward_draw(reward, fig=fig, ax=ax, mappable=mappable, **kwargs)

    for i in range(nrows):
        cax = fig.add_subplot(gs[i, -1])
        fig.colorbar(mappable=mappable, cax=cax, format=cbar_format)

    return fig
