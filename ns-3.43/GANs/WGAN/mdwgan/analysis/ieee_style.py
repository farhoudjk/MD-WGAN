"""IEEE IoT-J figure style.

Targets the journal's two-column layout: Times-metric serif at the caption size,
hairline axes, no top/right spines, and figure widths that need no scaling in
LaTeX (scaling a figure rescales its type and breaks the match with body text).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEE two-column geometry (inches).
COL1 = 3.5          # single column
COL2 = 7.16         # full width

# Validated categorical palette: all-pairs CVD check passes.
# Blue / orange / teal / purple, plus a reserved red for reference lines.
C = ["#2f62b8", "#cf5c0a", "#0e8f7a", "#8c2f91"]
ACCENT = "#b02030"
MARKERS = ["o", "s", "^", "D", "v"]
INK = "#000000"
MUTED = "#595959"
GRID = "#d9d9d9"

SERIF = ["Nimbus Roman", "Liberation Serif", "Times New Roman", "DejaVu Serif"]


def apply():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 1000,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,          # embed TrueType: IEEE requires embedded fonts
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": SERIF,
        "mathtext.fontset": "stix",
        # Type sizes: body text is 10 pt, captions 8 pt; axis text sits at 8.
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        # Hairline frame, no chartjunk.
        "axes.linewidth": 0.6,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.4,
        "grid.alpha": 1.0,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,
        "lines.markeredgewidth": 0.0,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.1,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.3,
    })


def panel_label(ax, text, dy=-0.30):
    """IEEE convention: (a)/(b) centred *below* the panel, not inside it."""
    ax.text(0.5, dy, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color=INK)
