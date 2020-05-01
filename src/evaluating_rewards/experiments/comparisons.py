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

"""Experiment helper methods to compare two reward models.

See Colab notebook for use cases.
"""

from typing import Any, Dict

import numpy as np

from evaluating_rewards import comparisons, datasets, rewards


def norm_diff(predicted: np.ndarray, actual: np.ndarray, norm: int):
    r"""Computes the mean $$\ell_p$$ norm of the delta between predicted and actual.

    Normalizes depending on the norm: i.e. for $$\ell_1$$ will divide by the
    number of elements, for $$\ell_2$$ by the square root.

    Arguments:
        predicted: A 1-dimensional array.
        actual: A 1-dimensoinal array.
        norm: The power p in $$\ell_p$$-norm.

    Returns:
        The normalized norm difference between predicted and actual.
    """
    delta = predicted - actual
    if delta.ndim != 1:
        raise TypeError("'predicted' and 'actual' must be 1-dimensional arrays.")
    n = actual.shape[0]
    scale = np.power(n, 1 / norm)
    return np.linalg.norm(delta, ord=norm) / scale


def constant_baseline(
    match: comparisons.RegressModel,
    target: rewards.RewardModel,
    dataset: datasets.TransitionsCallable,
    test_size: int = 4096,
) -> Dict[str, Any]:
    """Computes the error in predictions of the model matched and some baselines.

    Arguments:
        match: The (fitted) match object.
        target: The reward model we are trying to predict.
        dataset: The dataset to evaluate on.
        test_size: The number of samples to evaluate on.

    Returns:
        A dictionary containing summary statistics.
    """
    test_set = dataset(test_size)
    models = {"matched": match.model, "target": target}
    preds = rewards.evaluate_models(models, test_set)

    actual_delta = preds["matched"] - preds["target"]
    return {
        "int_l1": norm_diff(actual_delta, preds["target"], norm=1),
        "int_l2": norm_diff(actual_delta, preds["target"], norm=2),
        "baseline_l1": norm_diff(np.median(preds["target"]), preds["target"], norm=1),
        "baseline_l2": norm_diff(np.mean(preds["target"]), preds["target"], norm=2),
    }
