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

"""Methods related to generating figures."""

import logging
import os
from typing import Iterable, Tuple

import matplotlib.pyplot as plt


def save_fig(path: str, fig: plt.Figure, fmt: str = "pdf", dpi: int = 300, **kwargs):
    path = f"{path}.{fmt}"
    root_dir = os.path.dirname(path)
    os.makedirs(root_dir, exist_ok=True)
    logging.info(f"Saving figure to {path}")
    kwargs.setdefault("transparent", True)
    with open(path, "wb") as f:
        fig.savefig(f, format=fmt, dpi=dpi, **kwargs)


def save_figs(root_dir: str, generator: Iterable[Tuple[str, plt.Figure]], **kwargs) -> None:
    for name, fig in generator:
        name = name.replace("/", "_")
        path = os.path.join(root_dir, name)
        save_fig(path, fig, **kwargs)
