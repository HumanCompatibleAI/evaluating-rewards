"""matplotlib styles."""

import os

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
    "heatmap-2col": {"figure.figsize": (6.75, 5.0625)},
    "heatmap-1col": {
        "font.size": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": (3.25, 2.4375),
        "figure.subplot.top": 0.99,
        "figure.subplot.bottom": 0.16,
        "figure.subplot.left": 0.15,
        "figure.subplot.right": 0.91,
    },
    "tex": {
        "backend": "pgf",
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": [r"\usepackage{figemojis}", r"\usepackage{times}"],
    },
}
