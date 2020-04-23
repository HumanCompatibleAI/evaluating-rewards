# Copyright 2019, 2020 DeepMind Technologies Limited, Adam Gleave
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

"""Helper methods for transforming configs and indices to more easily human interpretable values."""

import re

import pandas as pd

from evaluating_rewards.analysis import results

TRANSFORMATIONS = {
    r"^evaluating_rewards[_/](.*)-v0": r"\1",
    r"^imitation[_/](.*)-v0": r"\1",
    "^Zero": r"\\zeroreward{}",
    "^PointMassDense": r"\\dense{}",
    "^PointMassGroundTruth": r"\\magnitude{}\\controlpenalty{}",
    "^PointMassSparse": r"\\sparse{}",
    "^PointMazeGroundTruth": "GT",
    r"(.*)(Hopper|HalfCheetah)GroundTruth(.*)": r"\1\2\\running{}\3",
    r"(.*)(Hopper|HalfCheetah)Backflip(.*)": r"\1\2\\backflipping{}\3",
    r"^Hopper(.*)": r"\1",
    r"^HalfCheetah(.*)": r"\1",
    r"^(.*)Backward(.*)": r"\\backward{\1\2}",
    r"^(.*)Forward(.*)": r"\\forward{\1\2}",
    r"^(.*)WithCtrl(.*)": r"\1\\controlpenalty{}\2",
    r"^(.*)NoCtrl(.*)": r"\1\\nocontrolpenalty{}\2",
    "sparse_goal": r"\\sparsegoal{}",
    "transformed_goal": r"\\densegoal{}",
    "center_goal": r"\\centergoal{}",
    "sparse_penalty": r"\\sparsepenalty{}",
    "dirt_path": r"\\dirtpath{}",
    "cliff_walk": r"\\cliffwalk{}",
}
LEVEL_NAMES = {
    # Won't normally use both type and path in one plot
    "source_reward_type": "Source",
    "source_reward_path": "Source",
    "target_reward_type": "Target",
    "target_reward_path": "Target",
}
WHITELISTED_LEVELS = ["source_reward_type", "target_reward_type"]  # never remove these levels


def pretty_rewrite(x):
    if not isinstance(x, str):
        return x

    for pattern, repl in TRANSFORMATIONS.items():
        x = re.sub(pattern, repl, x)
    return x


def rewrite_index(series: pd.Series) -> pd.Series:
    """Replace `source_reward_path` with info extracted from config at that path."""
    if "source_reward_path" in series.index.names:
        new_index = series.index.to_frame(index=False)
        source_reward = results.path_to_config(
            new_index["source_reward_type"], new_index["source_reward_path"]
        )
        new_index = new_index.drop(columns=["source_reward_type", "source_reward_path"])
        new_index = pd.concat([source_reward, new_index], axis=1)
        new_index = pd.MultiIndex.from_frame(new_index)
        series.index = new_index
    return series


def remove_constant_levels(index: pd.MultiIndex) -> pd.MultiIndex:
    index = index.copy()
    levels = index.names
    for level in levels:
        if len(index.get_level_values(level).unique()) == 1 and level not in WHITELISTED_LEVELS:
            index = index.droplevel(level=level)
    return index


def index_reformat(series: pd.Series, preserve_order: bool) -> pd.DataFrame:
    """Helper to reformat labels for ease of interpretability."""
    series = series.copy()
    series = rewrite_index(series)
    series.index = remove_constant_levels(series.index)
    series.index.names = [LEVEL_NAMES.get(name, name) for name in series.index.names]
    series = series.rename(index=pretty_rewrite)

    # Preserve order of inputs
    df = series.unstack("Target")
    if preserve_order:
        df = df.reindex(columns=series.index.get_level_values("Target").unique())
        for level in series.index.names:
            kwargs = {}
            if isinstance(df.index, pd.MultiIndex):
                kwargs = dict(level=level)
            if level != "Target":
                df = df.reindex(index=series.index.get_level_values(level).unique(), **kwargs)
    else:
        df = df.sort_index()
    return df
