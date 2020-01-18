"""matplotlib styles."""

import os

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))

STYLES = {
    # Matching ICML 2020 style
    "paper": {
        "figure.figsize": (6.75, 9.0),
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "font.size": 10,
        "legend.fontsize": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.subplot.top": 0.99,
        "figure.subplot.bottom": 0.11,
        "figure.subplot.left": 0.16,
        "figure.subplot.right": 0.96,
    },
    "paper-1col": {"figure.figsize": (3.25, 9.0)},
    "tex": {
        "backend": "pgf",
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": [r"\usepackage{figemojis}", r"\usepackage{times}"],
    },
}
