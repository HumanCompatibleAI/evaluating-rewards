# Copyright 2019 DeepMind Technologies Limited and Adam Gleave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common functionality for tests."""
import copy
from typing import Dict, Iterator, Tuple, TypeVar

from imitation.testing import envs as test_envs
from stable_baselines.common import vec_env
import pytest


def mark_parametrize_dict(argnames, args, **kwargs):
    ids = list(args.keys())
    argvals = args.values()
    return pytest.mark.parametrize(argnames, argvals, ids=ids, **kwargs)


def mark_parametrize_kwargs(args, **kwargs):
    """Like pytest.mark.parametrize, but taking a dictionary specifying keyword arguments.

    Arguments:
        args: A dict mapping from testcase names to a dictionary.
            The inner dictionary maps from parameter names to values.

    Raises:
        ValueError if the inner dictionaries have different keys from each other.
    """
    ids = []
    argvals = []
    argnames = sorted(list(next(iter(args.values())).keys()))
    for key, test_cfg in args.items():
        ids.append(key)
        sorted_keys = sorted(test_cfg.keys())
        if sorted_keys != argnames:
            raise ValueError(
                "Invalid test configuration: inconsistent argument"
                "names, f{test_cfg} != {argnames}."
            )
        argvals.append([test_cfg[k] for k in argnames])

    return pytest.mark.parametrize(argnames, argvals, ids=ids, **kwargs)


K = TypeVar("K")
V = TypeVar("V")


def combine_dicts(*dicts: Dict[str, Dict[K, V]],) -> Iterator[Tuple[str, Dict[K, V]]]:
    """Return a generator merging together the dictionary arguments.

    Usage example:

    > list(combine_dicts({'a': {'x': 1}, 'b': {'x': 2}},
                         {'c': {'y': 2}, 'd': {'x': 4, 'z': 3}}))
    [('a_c', {'x': 1, 'y': 2}),
     ('a_d', {'x': 4, 'z': 3}),
     ('b_c', {'x': 2, 'y': 2}),
     ('b_d', {'x': 4, 'z': 3})]


    Arguments:
        *dicts: A list of dictionaries, mapping from strings to inner dictionaries.

    Yields:
        Pairs (key, merged) computed from the cartesian product over the inner
        dictionaries. Specifically, if *dicts is an n-length list of dictionaries,
        then for each n-wise combination of inner dictionaries it yields
        (key, merged) where key is the keys of each of the inner dictionaries
        joined with "_" and merged is a dictionary containing key-value pairs from
        each of the inner dictionaries. (If a key occurs multiple times, the
        right-most occurrence takes precedence.)
    """
    head, *tail = dicts
    if not tail:
        for name, cfg in head.items():
            yield name, cfg
    else:
        for head_name, head_cfg in head.items():
            for tail_name, tail_cfg in combine_dicts(*tail):
                name = head_name + "_" + tail_name
                cfg = copy.deepcopy(head_cfg)
                cfg.update(tail_cfg)
                yield name, cfg


make_env = test_envs.make_env_fixture(skip_fn=pytest.skip)

def make_venv(env_name):
    return vec_env.DummyVecEnv([lambda: make_env(env_name)])
