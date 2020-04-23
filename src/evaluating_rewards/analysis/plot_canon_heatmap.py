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

"""CLI script to plot heatmap of canonical distance between pairs of reward models."""

import functools
import logging
import os
from typing import Any, Iterable, Mapping, Tuple

import gym
from imitation import util
import numpy as np
import pandas as pd
import sacred
from stable_baselines.common import vec_env
import tensorflow as tf

from evaluating_rewards import canonical_sample, datasets, rewards, serialize, tabular
from evaluating_rewards.analysis import dissimilarity_heatmap_config, stylesheets, visualize
from evaluating_rewards.scripts import script_utils

plot_divergence_heatmap_ex = sacred.Experiment("plot_divergence_heatmap")
logger = logging.getLogger("evaluating_rewards.analysis.plot_canon_heatmap")


dissimilarity_heatmap_config.make_config(plot_divergence_heatmap_ex)


@plot_divergence_heatmap_ex.config
def default_config():
    """Default configuration values."""
    n_samples = 4096  # number of samples in dataset
    n_mean_samples = 4096  # number of samples to estimate mean
    heatmap_kwargs = {
        "log": False,
    }
    # TODO(adam): anything specific to CANON...
    # Different kinds, e.g. Pearson vs direct?
    _ = locals()
    del _


@plot_divergence_heatmap_ex.config
def logging_config(env_name, log_root):
    # TODO(adam): include any other important config entries here e.g. kind.
    log_dir = os.path.join(  # noqa: F841  pylint:disable=unused-variable
        log_root, "plot_canon_heatmap", env_name, util.make_unique_timestamp(),
    )


@plot_divergence_heatmap_ex.named_config
def test():
    """Intended for debugging/unit test."""
    # TODO(adam): anything else custom here?
    # Do not include "tex" in styles here: this will break on CI.
    n_samples = 64
    n_mean_samples = 64
    styles = ["paper", "heatmap-1col"]
    _ = locals()
    del _


def load_models(
    env_name: str, reward_cfgs: Iterable[Tuple[str, str]],
) -> Mapping[Tuple[str, str], rewards.RewardModel]:
    venv = vec_env.DummyVecEnv([lambda: gym.make(env_name)])
    return {(kind, path): serialize.load_reward(kind, path, venv) for kind, path in reward_cfgs}


def space_to_sample(space: gym.Space):
    def f(n: int) -> np.ndarray:
        return np.array([space.sample() for _ in range(n)])

    return f


def make_gym_dists(env_name: str) -> Tuple[datasets.SampleDist, datasets.SampleDist]:
    env = gym.make(env_name)
    obs_dist = space_to_sample(env.observation_space)
    act_dist = space_to_sample(env.action_space)

    return obs_dist, act_dist


@plot_divergence_heatmap_ex.main
def plot_divergence_heatmap(
    env_name: str,
    n_samples: int,
    n_mean_samples: int,
    x_reward_cfgs: Iterable[Tuple[str, str]],
    y_reward_cfgs: Iterable[Tuple[str, str]],
    styles: Iterable[str],
    heatmap_kwargs: Mapping[str, Any],
    log_dir: str,
    save_kwargs: Mapping[str, Any],
):
    """Entry-point into script to produce divergence heatmaps.

    Args:
        env_name: the name of the environment to plot rewards for.
        n_samples: the number of samples to estimate the distance with.
        n_mean_samples: the number of samples to estimate the mean reward for canonicalization.
        x_reward_cfgs: tuples of reward_type and reward_path for x-axis (target).
        y_reward_cfgs: tuples of reward_type and reward_path for y-axis (source).
        styles: styles to apply from `evaluating_rewards.analysis.stylesheets`.
        heatmap_kwargs: passed through to `analysis.compact_heatmaps`.
        log_dir: directory to write figures and other logging to.
        save_kwargs: passed through to `analysis.save_figs`.
    """
    # Sacred turns our tuples into lists :(, undo
    x_reward_cfgs = [tuple(v) for v in x_reward_cfgs]
    y_reward_cfgs = [tuple(v) for v in y_reward_cfgs]

    logger.info("Loading models")
    g = tf.Graph()
    with g.as_default():
        sess = tf.Session()
        with sess.as_default():
            reward_cfgs = list(x_reward_cfgs) + list(y_reward_cfgs)
            models = load_models(env_name, reward_cfgs)

    # TODO(adam): make distribution configurable
    obs_dist, act_dist = make_gym_dists(env_name)

    # TODO(adam): support mesh method of computation?
    # TODO(adam): configurable transition generator
    logger.info("Sampling dataset")
    with datasets.iid_transition_generator(obs_dist, act_dist) as iid_transition:
        batch = iid_transition(n_samples)
    with sess.as_default():
        logger.info("Removing shaping")
        deshaped_rew = canonical_sample.sample_canon_shaping(
            models, batch, act_dist, obs_dist, n_mean_samples=n_mean_samples,
        )
        x_deshaped_rew = {cfg: deshaped_rew[cfg] for cfg in x_reward_cfgs}
        y_deshaped_rew = {cfg: deshaped_rew[cfg] for cfg in y_reward_cfgs}
    logger.info("Computing distance")
    dissimilarity = 0.5 * canonical_sample.cross_distance(
        x_deshaped_rew,
        y_deshaped_rew,
        functools.partial(tabular.direct_distance, p=1),
        parallelism=1,
    )
    dissimilarity.index = pd.MultiIndex.from_tuples(
        dissimilarity.index, names=("source_reward_type", "source_reward_path"),
    )
    dissimilarity.columns = pd.MultiIndex.from_tuples(
        dissimilarity.columns, names=("target_reward_type", "target_reward_path"),
    )
    dissimilarity = dissimilarity.stack(level=("target_reward_type", "target_reward_path"))

    with stylesheets.setup_styles(styles):
        figs = visualize.compact_heatmaps(dissimilarity=dissimilarity, **heatmap_kwargs)
        visualize.save_figs(log_dir, figs.items(), **save_kwargs)

    return figs


if __name__ == "__main__":
    script_utils.experiment_main(plot_divergence_heatmap_ex, "plot_divergence_heatmap")
