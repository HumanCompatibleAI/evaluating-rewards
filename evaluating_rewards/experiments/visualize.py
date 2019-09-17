# Copyright 2019 DeepMind Technologies Limited
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

"""Methods to generate plots and visualize data."""

import os
from typing import Iterable, Optional, Tuple

from absl import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Internal dependencies


def plot_shaping_comparison(df: pd.DataFrame,
                            cols: Optional[Iterable[str]] = None,
                            **kwargs) -> pd.DataFrame:
  """Plots return value of experiments.compare_synthetic."""
  if cols is None:
    cols = ["Intrinsic", "Shaping"]
  df = df.loc[:, cols]
  longform = df.reset_index()
  longform = pd.melt(longform, id_vars=["Reward Noise", "Potential Noise"],
                     var_name="Metric", value_name="Distance")
  sns.lineplot(x="Reward Noise", y="Distance", hue="Potential Noise",
               style="Metric", data=longform, **kwargs)
  return longform


def save_fig(path: str, fig: plt.Figure, **kwargs):
  root_dir = os.path.dirname(path)
  os.makedirs(root_dir, exist_ok=True)
  logging.info(f"Saving figure to {path}")
  with open(path, "wb") as f:
    fig.savefig(f, format="pdf", **kwargs)


def save_figs(root_dir: str,
              generator: Iterable[Tuple[str, plt.Figure]],
              **kwargs) -> None:
  for name, fig in generator:
    name = name.replace("/", "_")
    path = os.path.join(root_dir, name + ".pdf")
    save_fig(path, fig, **kwargs)
