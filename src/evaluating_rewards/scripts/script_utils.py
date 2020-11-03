# Copyright 2019 DeepMind Technologies Limited
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

"""Utility functions to aid in constructing Sacred experiments."""

import os
from typing import Any, Iterable, Mapping, MutableMapping, Optional, TypeVar

from imitation.util import util
import sacred
from sacred import observers

# envs imported for side-effects (registers with Gym)
from evaluating_rewards import envs  # noqa: F401  pylint:disable=unused-import
from evaluating_rewards import serialize


def logging_config(log_root, env_name):
    log_dir = os.path.join(log_root, env_name.replace("/", "_"), util.make_unique_timestamp())
    _ = locals()  # quieten flake8 unused variable warning
    del _


def add_logging_config(experiment, name):
    experiment.add_config({"log_root": os.path.join(serialize.get_output_dir(), name)})
    experiment.config(logging_config)


def sanitize_path(x: Any) -> str:
    """Converts `x` to string and replaces any occurrence of "/" with "_"."""
    return str(x).replace("/", "_")


def add_sacred_symlink(observer: observers.FileStorageObserver):
    """Adds a symbolic link to the output directory of `observer`."""

    def f(log_dir: str) -> None:
        """Adds a symbolic link in log_dir to observer output directory."""
        if observer.dir is None:
            # In a command like print_config that produces no permanent output
            return
        os.makedirs(log_dir)
        # Use relative paths so we can mount the output directory at different paths
        # (e.g. when copying across machines).
        symlink_path = os.path.join(log_dir, "sacred")
        target_path = os.path.relpath(observer.dir, start=log_dir)
        os.symlink(target_path, symlink_path, target_is_directory=True)

    return f


K = TypeVar("K")
V = TypeVar("V")


def recursive_dict_merge(
    dest: MutableMapping[K, V],
    update_by: Mapping[K, V],
    path: Optional[Iterable[str]] = None,
    allow_conflict: bool = False,
) -> MutableMapping[K, V]:
    """Merges update_by into dest recursively."""
    if path is None:
        path = []
    for key in update_by:
        if key in dest:
            if isinstance(dest[key], dict) and isinstance(update_by[key], dict):
                recursive_dict_merge(dest[key], update_by[key], path + [str(key)])
            elif isinstance(dest[key], (tuple, list)) and isinstance(update_by[key], (tuple, list)):
                dest[key] = tuple(set(dest[key]).union(update_by[key]))
            elif dest[key] == update_by[key]:
                pass  # same leaf value
            elif not allow_conflict:
                raise Exception("Conflict at {}".format(".".join(path + [str(key)])))
        else:
            dest[key] = update_by[key]
    return dest


def experiment_main(experiment: sacred.Experiment, name: str, sacred_symlink: bool = True):
    """Main function for experiment."""

    sacred_dir = os.path.join(serialize.get_output_dir(), "sacred", name)
    observer = observers.FileStorageObserver.create(sacred_dir)
    if sacred_symlink:
        experiment.pre_run_hook(add_sacred_symlink(observer))
    experiment.observers.append(observer)
    experiment.run_commandline()
