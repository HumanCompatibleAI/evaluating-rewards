"""matplotlib styles."""

import contextlib
import os
from typing import Iterable, Iterator

LATEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latex")

STYLES = {
    # Matching NeurIPS 2020 style
    "paper": {
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "mathtext.fontset": "cm",
        "font.size": 10,
        "legend.fontsize": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    },
    "huge": {"figure.figsize": (20, 10)},
    "pointmass-2col": {
        "figure.figsize": (5.5, 2.04),
        "figure.subplot.left": 0.2,
        "figure.subplot.right": 1.0,
        "figure.subplot.top": 0.92,
        "figure.subplot.bottom": 0.16,
        "figure.subplot.hspace": 0.2,
        "figure.subplot.wspace": 0.25,
    },
    "heatmap": {"font.size": 8, "xtick.labelsize": 8, "ytick.labelsize": 8},
    "heatmap-1col": {
        "figure.figsize": (5.5, 4.125),
        "figure.subplot.top": 0.99,
        "figure.subplot.right": 0.92,
        "figure.subplot.left": 0.08,
        "figure.subplot.bottom": 0.09,
    },
    "heatmap-2col": {
        "figure.figsize": (2.7, 2.025),
        "figure.subplot.top": 0.99,
        "figure.subplot.bottom": 0.16,
        "figure.subplot.left": 0.17,
        "figure.subplot.right": 0.91,
    },
    "heatmap-2col-fatlabels": {
        "figure.subplot.top": 0.98,
        "figure.subplot.bottom": 0.33,
        "figure.subplot.left": 0.25,
        "figure.subplot.right": 0.84,
    },
    "gridworld-heatmap": {
        "axes.facecolor": "lightgray",
        "image.cmap": "RdBu",
        "hatch.linewidth": 0.1,
    },
    "gridworld-heatmap-2in1": {
        "figure.figsize": (5.5, 2.77),
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.subplot.left": 0.03,
        "figure.subplot.right": 0.96,
        "figure.subplot.top": 0.97,
        "figure.subplot.bottom": 0.03,
        "figure.subplot.wspace": 0.1,
        "figure.subplot.hspace": 0.14,
    },
    "gridworld-heatmap-4in1": {
        "image.cmap": "RdBu",
        "figure.figsize": (5.5, 1.66),
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.subplot.left": 0.03,
        "figure.subplot.right": 0.96,
        "figure.subplot.top": 0.85,
        "figure.subplot.bottom": 0.14,
        "figure.subplot.wspace": 0.1,
    },
    "tex": {
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": [r"\usepackage{figsymbols}", r"\usepackage{times}"],
    },
}


@contextlib.contextmanager
def setup_styles(styles: Iterable[str]) -> Iterator[None]:
    """Context manager: uses specified matplotlib styles while in context.

    Side-effect: if "tex" is in styles, will switch `matplotlib` backend to `pgf`.

    Args:
        styles: keys of styles defined in `STYLES`.

    Returns:
        A ContextManager. While entered in the context, the specified styles are applied,
        and (if "tex" is one of the styles) the environment variable "TEXINPUTS" is set
        to support custom macros."""
    old_tex_inputs = os.environ.get("TEXINPUTS")
    try:
        if "tex" in styles:
            import matplotlib  # pylint:disable=import-outside-toplevel

            # PGF backend best for LaTeX. matplotlib probably already imported:
            # but should be able to switch as non-interactive.
            matplotlib.use("pgf", warn=False, force=True)
            os.environ["TEXINPUTS"] = LATEX_DIR + ":"
        styles = [STYLES[style] for style in styles]

        import matplotlib.pyplot as plt  # pylint:disable=import-outside-toplevel

        with plt.style.context(styles):
            yield
    finally:
        if "tex" in styles:
            if old_tex_inputs is None:
                del os.environ["TEXINPUTS"]
            else:
                os.environ["TEXINPUTS"] = old_tex_inputs
