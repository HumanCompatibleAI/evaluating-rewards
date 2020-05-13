# Copyright 2020 Adam Gleave
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

"""Miscellaneous helper methods."""

import multiprocessing
import multiprocessing.dummy
from typing import Callable, Mapping, Optional, Tuple, TypeVar

import numpy as np
import sklearn.utils

K = TypeVar("K")
V = TypeVar("V")


def bootstrap(
    *inputs, stat_fn, n_samples: int = 100, random_state: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Evaluates stat_fn for n_samples from inputs with replacement.

    Args:
        inputs: The inputs to bootstrap over.
        stat_fn: A function computing a statistic on inputs.
        n_samples: The number of bootstrapped samples to take.
        random_state: Random state passed to `sklearn.utils.resample`.

    Returns:
        n_samples of the distance computed by `distance_fn`, each on an independent sample with
        replacement from `rewa` and `rewb` of the same shape as `rewa` and `rewb`.
    """
    vals = []
    for _ in range(n_samples):
        samples = sklearn.utils.resample(*inputs, random_state=random_state)
        if len(inputs) > 1:
            val = stat_fn(*samples)
        else:
            val = stat_fn(samples)
        vals.append(val)

    return np.array(vals)


def empirical_ci(arr: np.ndarray, alpha: float = 95.0) -> np.ndarray:
    """Computes percentile range in an array of values.

    Args:
        arr: An array.
        alpha: Percentile confidence interval.

    Returns:
        A triple of the lower bound, median and upper bound of the confidence interval
        with a width of alpha.
    """
    percentiles = 50 - alpha / 2, 50, 50 + alpha / 2
    return np.percentile(arr, percentiles)


def cross_distance(
    rewxs: Mapping[K, np.ndarray],
    rewys: Mapping[K, np.ndarray],
    distance_fn: Callable[[np.ndarray, np.ndarray], V],
    parallelism: Optional[int] = None,
    threading: bool = True,
) -> Mapping[Tuple[K, K], V]:
    """Helper function to compute distance between all pairs of rewards from `rewxs` and `rewys`.

    Args:
        rewxs: A mapping from keys to NumPy arrays of shape `(n,)`.
        rewys: A mapping from keys to NumPy arrays of shape `(n,)`.
        distance_fn: A function to compute the distance between two NumPy arrays.
        parallelism: The number of threads/processes to execute in parallel; if not specified,
            defaults to `multiprocessing.cpu_count()`.
        threading: If true, use multi-threading; otherwise, use multiprocessing. For many NumPy
            functions, multi-threading is higher performance since NumPy releases the GIL, and
            threading avoids expensive copying of the arrays needed for multiprocessing.

    Returns:
        A mapping from (i,j) to `distance_fn(rews[i], rews[j])`.
    """
    shapes = set((v.shape for v in rewxs.values()))
    shapes.update((v.shape for v in rewys.values()))
    assert len(shapes) <= 1, "rewards differ in shape"

    tasks = {(kx, ky): (rewx, rewy) for kx, rewx in rewxs.items() for ky, rewy in rewys.items()}

    if parallelism == 1:
        # Only one process? Skip multiprocessing, since creating Pool adds overhead.
        results = [distance_fn(rewx, rewy) for rewx, rewy in tasks.values()]
    else:
        # We want parallelism, use multiprocessing to speed things up.
        module = multiprocessing.dummy if threading else multiprocessing
        with module.Pool(processes=parallelism) as pool:
            results = pool.starmap(distance_fn, tasks.values())

    return dict(zip(tasks.keys(), results))
