"""Heatmaps for rewards in gridworld environments.

This is currently only used for illustrative examples in the paper;
none of the actual experiments are gridworlds."""

from typing import Iterable

import matplotlib
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

ACTION_DELTA = [
    (0, 0),  # no-op
    (-1, 0),  # left
    (0, 1),  # up
    (1, 0),  # right
    (0, -1),  # down
]


def clipped_range(start: int, end: int, delta: int) -> Iterable[int]:
    """Range between `start` and `end`, updated away from `delta`.

    Has the invariant that for each i in the returned range, `i + delta` will lie in [start, end].
    Clips the range as little as possible to achieve this.

    Args:
        start: start index.
        end: end index.
        delta: direction: -1, 0 or 1.

    Returns:
        A range with the invariant that for each `i`, `i + delta` will lie in [start, end].
        Clips the range as little as possible to achieve this.
    """
    assert start < end
    assert delta in [-1, 0, 1]
    start = max(start, start - delta)
    end = min(end, end - delta)
    return range(start, end)


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


def offset(margin, arrow_length, parallel_delta, orthogonal_delta):
    offset_margin = margin * orthogonal_delta
    offset_align = (1 - arrow_length) * parallel_delta / 2
    return offset_margin + offset_align


def plot_gridworld_reward(state_reward, state_potential):  # pylint:disable=too-many-statements
    """Plots a heatmap of reward for the gridworld.

    Args:
        - state_reward: a two-dimensional array specifying a reward for each state.
        - state_potential: a two-dimensional array specifying a potential for each state.

    Returns:
        A heatmap of state_reward, with labeled arrows between cells showing the shaping
        induced by state_potential.
    """
    # TODO: is it OK to ignore actions?
    # TODO: is this too constrained a reward class?
    fig, ax = plt.subplots(1, 1)
    sns.heatmap(
        state_reward, annot=True, cmap="RdBu", linecolor="k", linewidths=1, ax=ax  # center=0,
    )

    assert state_reward.ndim == 2
    xlen, ylen = state_reward.shape
    for x_delta, y_delta in ACTION_DELTA[1:]:
        for x in clipped_range(0, xlen, x_delta):
            for y in clipped_range(0, ylen, y_delta):
                old_pot = state_potential[x, y]
                new_pot = state_potential[x + x_delta, y + y_delta]
                shaping = new_pot - old_pot
                arrow_length = 0.45
                margin = 0.15
                width = 0.02

                x_pos = x + 0.5 + offset(margin, arrow_length, x_delta, y_delta)
                y_pos = y + 0.5 + offset(margin, arrow_length, y_delta, x_delta)
                x_len = x_delta * arrow_length
                y_len = y_delta * arrow_length
                ax.arrow(
                    x_pos, y_pos, x_len, y_len, width=width, length_includes_head=True, color="k",
                )

                text = "{:.0f}".format(shaping)
                rotation = "horizontal" if x_delta else "vertical"
                horizontalalignment = "center" if x_delta else "left"
                verticalalignment = "center" if y_delta else "baseline"
                ax.text(
                    x_pos + x_len / 2 - 0.15 * abs(y_delta),
                    y_pos + y_len / 2 - 0.1 * abs(x_delta),
                    text,
                    horizontalalignment=horizontalalignment,
                    verticalalignment=verticalalignment,
                    rotation=rotation,
                    color="w",
                )

    return fig


# TODO(adam): is this too constrained a reward class?
def plot_gridworld_reward_nested(  # pylint:disable=too-many-statements
    state_action_reward: np.ndarray,
    from_dest: bool = True,
    cmap: str = "RdBu",
    annot_padding: float = 0.33,
    cbar_fraction: float = 0.05,
) -> plt.Figure:
    """
    Plots a heatmap of reward for the gridworld.

    Args:
      - state_action_reward: a three-dimensional array specifying the gridworld reward.
      - from_dest: if True, the triangular wedges represent reward when arriving into this
        cell from the adjacent cell; if False, represent reward when leaving this cell into
        the adjacent cell.
      - cmap: a matplotlib color map identifier.
      - annot_padding: a fraction of a supercell to offset the annotation from the centre.
      - cbar_fraction: the fraction of the axes the colorbar takes up.

    Returns:
        A heatmap consisting of a "supercell" for each state `(i,j)` in the original gridworld.
        This supercell contains a central circle, representing the no-op action reward and four
        triangular wedges, representing the left, up, right and down action rewards.
    """
    xlen, ylen, num_actions = state_action_reward.shape
    assert num_actions == len(ACTION_DELTA)

    # Construct figure
    fig, ax = plt.subplots(1, 1)
    # Axis limits
    ax.set_xlim(0, xlen)
    ax.set_ylim(ylen, 0)
    # Make grid between gridworld cells
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.grid(which="major", color="k")
    # Background
    ax.set_facecolor("lightgray")

    # Compute offsets of the vertices of the triangles to draw.
    # All offsets are relative to bottom-left
    corners = [(0, 0), (0, 1), (1, 1), (1, 0)]  # counter-clockwise, corners of unit square
    offsets = {  # triangles, cutting unit-square into quarters
        direction: np.array([corners[i], [0.5, 0.5], corners[(i + 1) % len(corners)]])
        for i, direction in enumerate(ACTION_DELTA[1:])
    }
    offsets[(0, 0)] = np.array([0.5, 0.5])

    # Color mapping
    cmap = matplotlib.cm.get_cmap(cmap)
    norm = mcolors.Normalize(
        vmin=np.nanmin(state_action_reward), vmax=np.nanmax(state_action_reward)
    )
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Construct patches
    legal_verts = []
    legal_facecolors = []
    illegal_verts = []
    circle_offsets = []
    circle_facecolors = []

    it = np.nditer(state_action_reward, flags=["multi_index"])
    while not it.finished:
        reward = it[0]
        x, y, action = it.multi_index

        pos = np.array([x, y])
        direction = np.array(ACTION_DELTA[action])
        if from_dest:
            pos = pos + direction
            direction = -direction
        vert = pos + offsets[tuple(direction)]

        if np.isfinite(reward):
            color = mappable.to_rgba(reward)
            text = f"{reward:.0f}"

            lum = sns.utils.relative_luminance(color)
            text_color = ".15" if lum > 0.408 else "w"

            xy = pos + 0.5
            if tuple(direction) != (0, 0):
                xy = xy + annot_padding * direction

            ax.annotate(
                text, xy=xy, ha="center", va="center", color=text_color,
            )

            if tuple(direction) == (0, 0):  # no-op: circle at center
                circle_offsets.append(vert)
                circle_facecolors.append(color)
            else:  # action moved: triangle at edge
                legal_verts.append(vert)
                legal_facecolors.append(color)
        else:
            assert tuple(direction) != (0, 0)
            illegal_verts.append(vert)

        it.iternext()

    # TODO(adam): can remove if don't do anything more interesting than solid background
    illegal_poly = mcollections.PolyCollection(
        verts=illegal_verts, facecolors="lightgray", edgecolors="lightgray"
    )
    legal_poly = mcollections.PolyCollection(
        verts=legal_verts, facecolors=legal_facecolors, edgecolors="gray"
    )
    circles = mcollections.CircleCollection(
        sizes=[200] * len(circle_offsets),
        facecolors=circle_facecolors,
        edgecolors="gray",
        offsets=circle_offsets,
        transOffset=ax.transData,
    )

    ax.add_collection(illegal_poly)
    ax.add_collection(legal_poly)
    ax.add_collection(circles)

    # Add color bar
    fig.colorbar(mappable, fraction=cbar_fraction)

    return fig
