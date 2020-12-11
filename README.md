[![CircleCI](https://circleci.com/gh/HumanCompatibleAI/evaluating-rewards.svg?style=svg)](https://circleci.com/gh/HumanCompatibleAI/evaluating_rewards)
[![codecov](https://codecov.io/gh/HumanCompatibleAI/evaluating-rewards/branch/master/graph/badge.svg)](https://codecov.io/gh/HumanCompatibleAI/evaluating_rewards)

# Evaluating Rewards

`evaluating_rewards` is a library to compare and evaluate reward functions. The accompanying paper,
[Quantifying Differences in Reward Functions](https://arxiv.org/abs/2006.13900), describes the
methods implemented in this repository.

## Getting Started

### Installation

To install `evaluating_rewards`, clone the repository and run:

```
pip install evaluating_rewards/
```

To install in developer mode so that edits will be immediately available:

```
pip install -e evaluating_rewards/
```

The package is compatible with Python 3.6 and upwards. There is no support for
Python 2.

### Computing EPIC Distance

The `evaluating_rewards.analysis.dissimilarity_heatmaps.plot_epic_heatmap` script provides
a convenient front-end to generate heatmaps of EPIC distance between reward models.
For example, to replicate Figure 2(a) from the paper, simply run:

```bash
python -m evaluating_rewards.analysis.dissimilarity_heatmaps.plot_epic_heatmap with point_mass paper
# If you want higher precision results like the paper (warning: will take several hours), use:
# python -m evaluating_rewards.analysis.dissimilarity_heatmaps.plot_epic_heatmap with point_mass paper high_precision 
```

`plot_epic_heatmap` uses the `evaluating_rewards.epic_sample` module to compute the EPIC distance.
You may wish to use this module directly, for example when integrating EPIC into an existing
evaluation suite. Check out this [notebook](examples/epic_demo.ipynb) for an example of how to use
`epic_sample`.

## Technical Structure

`evaluating_rewards` consists of:

-   the main package, containing some reward learning algorithms and distance methods. In particular:
    + `epic_sample.py` contains the implementation of EPIC.
    + `rewards.py` and `comparisons.py` define deep reward models and
        associated comparison methods, and `tabular.py` the equivalent in a
        tabular (finite-state) setting.
    + `serialize.py` loads reward models, both from this package and other projects.
-   `envs`, a sub-package defining some simple environments and associated
    hard-coded reward models and (in some cases) policies.
-   `experiments`, a sub-package defining helper methods supporting experiments
    we have performed.

## Obtaining Reward Models

To use `evaluating_rewards`, you will need to have some reward models to
compare. We provide native support to load reward models output by
[imitation](https://github.com/humancompatibleai/imitation), an open-source
implementation of AIRL and GAIL. It is also simple to add new formats to
`serialize.py`.

## Reproducing the Paper

Note in addition to the Python requirements, you will need to install
[GNU Parallel](https://www.gnu.org/software/parallel/) to run these bash scripts, for example with
`sudo apt-get parallel`.

By default results will be saved to `$HOME/output`; set the `EVAL_OUTPUT_ROOT` environment variable
to override this.

### Distance Between Hand-Designed Reward Functions

To replicate the results in section 6.1 "Comparing hand-designed reward functions", run:

```bash
./runners/comparison/hardcoded_npec.sh
./runners/comparison/hardcoded_figs.sh
```

The first script, `hardcoded_npec.sh`, computes the NPEC distance between the hand-designed
reward functions. This is relatively slow since it requires training a deep network.

The second script, `hardcoded_figs`, plots the EPIC, NPEC and ERC heatmaps. For NPEC, the
distances precomputed by the first script are used. For EPIC and ERC they are computed on
demand since the distances are relatively cheap to compute.

The distances in gridworlds are produced by a separate script, exploiting the tabular nature of
the problem. Run:

```bash
# NPEC
python -m evaluating_rewards.analysis.dissimilarity_heatmaps.plot_gridworld_heatmap with paper
# EPIC
python -m evaluating_rewards.analysis.dissimilarity_heatmaps.plot_gridworld_heatmap with paper kind=fully_connected_random_canonical_direct
```

### Distance of Learned Reward Model to Ground Truth

To replicate the results in section 6.2 "Predicting policy performance from reward distance" and
section 6.3 "Sensitivity of reward distance to visitation state distribution", you should run:

```bash
./runners/transfer_point_maze.sh
python -m evaluating_rewards.scripts.pipeline.combined_distances with point_maze_learned_good high_precision
```

The first script, `transfer_point_maze.sh`, 0) trains a synthetic expert policy using RL on the
ground-truth reward; 1) trains reward models via IRL on demonstrations from this expert, and
via preference comparison and regression with labels from the ground-truth reward; 2a) computes
NPEC distance of these rewards from the ground-truth; 2b) trains policies using RL on the learned
reward models; 3) evaluates the resulting policies.

The second script, `combined_distances`, generates a table and also computes the EPIC and ERC distances
of the learned reward models from the ground truth.

## License

This library is licensed under the terms of the Apache license. See
[LICENSE](LICENSE) for more information.

DeepMind holds the copyright to all work prior to October 4th, during the
lead author's (Adam Gleave) internship at DeepMind. Subsequent modifications
conducted at UC Berkeley are copyright of the author.
Disclaimer: This is not an officially supported Google or DeepMind product.
