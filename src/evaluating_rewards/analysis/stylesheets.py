"""matplotlib styles."""

import contextlib
import os
from typing import Iterable, Iterator

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
    "huge": {"figure.figsize": (20, 10)},
    "pointmass-2col": {
        "figure.figsize": (6.75, 2.5),
        "figure.subplot.left": 0.2,
        "figure.subplot.right": 1.0,
        "figure.subplot.top": 0.92,
        "figure.subplot.bottom": 0.16,
        "figure.subplot.hspace": 0.2,
        "figure.subplot.wspace": 0.25,
    },
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
    "gridworld-heatmap": {
        "axes.facecolor": "lightgray",
        "image.cmap": "RdBu",
        "hatch.linewidth": 0.1,
    },
    "gridworld-heatmap-1col": {
        "figure.figsize": (3.25, 2.89),
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.subplot.left": 0.06,
        "figure.subplot.right": 0.91,
        "figure.subplot.top": 0.95,
        "figure.subplot.bottom": 0.09,
    },
    "gridworld-heatmap-1col-narrow": {
        "image.cmap": "RdBu",
        "figure.figsize": (2.5, 2.07),
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.subplot.left": 0.07,
        "figure.subplot.right": 0.89,
        "figure.subplot.top": 0.97,
        "figure.subplot.bottom": 0.1,
    },
    "gridworld-heatmap-1colin3": {
        "image.cmap": "RdBu",
        "figure.figsize": (2.15, 1.78),
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.subplot.left": 0.08,
        "figure.subplot.right": 0.88,
        "figure.subplot.top": 0.97,
        "figure.subplot.bottom": 0.1,
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
