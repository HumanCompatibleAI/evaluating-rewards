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

"""Masks to filter different reward types."""

import re
from typing import Any, Callable, Iterable

import pandas as pd

from evaluating_rewards.analysis import results
from evaluating_rewards.serialize import ZERO_REWARD


def compute_mask(
    series: pd.Series,
    matchings: Iterable[results.FilterFn],
    levels: Iterable[str] = ("source_reward_type", "target_reward_type"),
) -> pd.Series:
    """Masks values in `series` not matching any of `matchings`.

    Args:
        series: Input data.
        matchings: A sequence of callables.
        levels: A sequence of levels to match.

    Returns:
        A boolean Series, with an index equal to `series`.
        The value is false iff one of `matchings` returns true.
    """
    xs = []
    for level in levels:
        xs.append(set(series.index.get_level_values(level)))

    index = series.index
    for level in index.names:
        if level not in levels:
            index = index.droplevel(level=level)

    res = pd.Series(False, index=index)
    for matching in matchings:
        res |= index.map(matching)
    res = ~res
    res.index = series.index

    return res


def same(args: Iterable[Any]) -> bool:
    """All values in args are the same."""
    some_arg = next(iter(args))
    return all(arg == some_arg for arg in args)


def replace(pattern: str, replacement: str) -> Callable[[Iterable[str]], Iterable[str]]:
    """Iterable version version of re.sub."""
    pattern = re.compile(pattern)

    def helper(args: Iterable[str]) -> Iterable[str]:
        return tuple(re.sub(pattern, replacement, x) for x in args)

    return helper


def match(pattern: str) -> Callable[[Iterable[str]], Iterable[bool]]:
    """Iterable version of re.match."""
    pattern = re.compile(pattern)

    def helper(args: Iterable[str]) -> Iterable[bool]:
        return tuple(bool(pattern.match(x)) for x in args)

    return helper


def zero(args: Iterable[str]) -> bool:
    """Are any of the arguments the Zero reward function."""
    return any(arg == ZERO_REWARD for arg in args)


def control(args: Iterable[str]) -> bool:
    """Do args differ only in terms of with/without control cost?"""
    args = replace(r"(.*?)(NoCtrl|WithCtrl|)-(.*)", r"\1-\3")(args)
    return same(args)


def sparse_or_dense(args: Iterable[str]) -> bool:
    """Args are both sparse or both dense?"""
    args = replace(r"(.*)(Sparse|Dense)(.*)", r"\1\2")(args)
    return same(args)


def direction(args: Iterable[str]) -> bool:
    """Do args differ only in terms of forward/backward and control cost?"""
    pattern = r"evaluating_rewards/(.*?)(Backward|Forward)(WithCtrl|NoCtrl)(.*)"
    args = replace(pattern, r"\1\4")(args)
    return same(args)


def no_ctrl(args: Iterable[str]) -> bool:
    """Are args all without control cost?"""
    return all("NoCtrl" in arg for arg in args)


def always_true(args: Iterable[str]) -> bool:
    """Constant true."""
    del args
    return True
