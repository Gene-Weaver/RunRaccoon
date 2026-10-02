"""The RunRaccoon figure theme: journal-appropriate, quiet chrome, the data does the talking.

Colors are a validated colorblind-safe set (checked all-pairs for deuteranopia/protanopia/
tritanopia). train / val / test always get the same color *and* line style, so the encoding
survives grayscale printing.
"""
from __future__ import annotations

from typing import Any

# chrome & ink
SURFACE = "#ffffff"
INK = "#1d1d1b"
INK_2 = "#52514e"        # secondary text: tick labels, axis labels
MUTED = "#898781"        # tertiary text: annotations
GRID = "#e8e7e1"
AXIS = "#c3c2b7"
NEUTRAL = "#3b3a37"      # single, unpaired metrics (no train/val/test role)

# categorical slots, fixed order (never cycled past 8)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

ROLE_STYLE = {
    "train": {"color": PALETTE[0], "linestyle": "-", "label": "train"},
    "val": {"color": PALETTE[1], "linestyle": (0, (4.0, 1.6)), "label": "val"},
    "test": {"color": PALETTE[2], "linestyle": (0, (1.2, 1.4)), "label": "test"},
}

_FONT_PREFERENCE = ["Inter", "Helvetica Neue", "Helvetica", "Arial", "Roboto", "Liberation Sans", "DejaVu Sans"]


def _available_fonts() -> list[str]:
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    return [f for f in _FONT_PREFERENCE if f in have] or ["DejaVu Sans"]


def rc() -> dict[str, Any]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": _available_fonts(),
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 7.0,
        "axes.titlecolor": INK,
        "axes.labelsize": 7.5,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.7,
        "axes.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.xmargin": 0.02,
        "axes.ymargin": 0.08,
        "axes.unicode_minus": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "xtick.major.size": 2.5,
        "ytick.major.size": 0.0,
        "ytick.minor.size": 0.0,
        "xtick.major.width": 0.6,
        "ytick.major.pad": 3.0,
        "lines.linewidth": 1.5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.dash_capstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 7.0,
        "legend.handlelength": 2.2,
        "legend.labelcolor": INK_2,
        "figure.facecolor": SURFACE,
        "figure.dpi": 100,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,          # editable text in Illustrator / journals' PDF checks
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "image.cmap": "Blues",
    }


def apply() -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update(rc())
