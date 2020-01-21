"""matplotlib styles."""

import contextlib
import os
import sys
from typing import Iterable, Iterator
import warnings

LATEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latex")

STYLES = {
    # Matching ICML 2020 style
    "paper": {
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "font.size": 10,
        "legend.fontsize": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    },
    "pointmas-2col": {"figure.figsize": (6.75, 5.0625)},
    "heatmap-2col": {"figure.figsize": (6.75, 5.0625)},
    "heatmap-1col": {
        "font.size": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": (3.25, 2.4375),
        "figure.subplot.top": 0.99,
        "figure.subplot.bottom": 0.16,
        "figure.subplot.left": 0.16,
        "figure.subplot.right": 0.91,
    },
    "tex": {
        "backend": "pgf",
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": [r"\usepackage{figsymbols}", r"\usepackage{times}"],
    },
}


@contextlib.contextmanager
def setup_styles(styles: Iterable[str]) -> Iterator[None]:
    """Context manager: uses specified matplotlib styles while in context.

    WARNING: This should be called before any matplotlib

    Args:
        styles: keys of styles defined in `STYLES`.

    Returns:
        A ContextManager. While entered in the context, the specified styles are applied,
        and (if "tex" is one of the styles) the environment variable "TEXINPUTS" is set
        to support custom macros."""
    if "matplotlib" in sys.modules:
        warnings.warn(
            "`setup_styles` should be called before importing matplotlib. "
            "Otherwise custom backends (required for TeX) may not be set."
        )

    old_tex_inputs = os.environ.get("TEXINPUTS")
    try:
        if "tex" in styles:
            import matplotlib  # pylint:disable=import-outside-toplevel

            matplotlib.use("pgf")  # PGF backend best for LaTeX
            os.environ["TEXINPUTS"] = LATEX_DIR + ":"
        styles = [STYLES[style] for style in styles]

        import matplotlib.pyplot as plt  # pylint:disable=import-outside-toplevel

        with plt.style.context(styles):
            yield
    finally:
        if old_tex_inputs is None:
            del os.environ["TEXINPUTS"]
        else:
            os.environ["TEXINPUTS"] = old_tex_inputs
