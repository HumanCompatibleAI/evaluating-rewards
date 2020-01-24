"""Generate examples of different figure types."""

import numpy as np

from evaluating_rewards.analysis import gridworld_heatmap, stylesheets


def main():
    """Entry-point into CLI script."""
    base_reward = np.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, 100]])

    base_potential = np.zeros((3, 3))
    shaped_potential = -np.array([[400, 300, 200], [300, 200, 100], [200, 100, 0]])

    potentials = {
        "unshaped": base_potential,
        "shaped": shaped_potential,
    }

    with stylesheets.setup_styles(["paper", "gridworld-heatmap-1col"]):
        for rew_kind, potential in potentials.items():
            fig = gridworld_heatmap.plot_gridworld_reward(base_reward, potential)
            fig.savefig(f"{rew_kind}_arrow.pdf")

            sa_reward = gridworld_heatmap.shape(base_reward, potential)
            for dest in [False, True]:
                fig = gridworld_heatmap.plot_gridworld_reward_nested(sa_reward, from_dest=dest)
                fig.savefig(f"{rew_kind}_{dest}.pdf")


if __name__ == "__main__":
    main()
