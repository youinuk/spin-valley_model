"""One visual idiom for every manuscript and supplement figure.

    from reproduce.figstyle import apply, PALETTE, pair_colors, panel_title

    apply()

Why this exists
---------------
The figures were written at different times and look it. Figs. S1-S4 use
matplotlib defaults -- framed legends, a grey grid, the C0/C1 colour cycle,
titles inside the axes -- while Figs. 2 to 4 use a hand-set palette with no
grid. A reader sees two documents.

This module holds the choices in one place so a producer opts in with one call
and stops carrying its own rcParams block. It changes appearance only: no
producer's data path, selection, or printed numbers go through here.

Applying it to a figure is a one-line change at the top of that producer. It
does not require touching the science below.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

# Two colour namespaces, kept apart on purpose.
#
# ANSATZ identifies WHICH coupling profile a curve or marker belongs to.
# COMPARISON identifies WHICH KIND of variation an estimand measures.
#
# Orange appears in both -- as B_x in Figs. 1 and 2, as cross-ansatz in Figs. 4
# and S5 -- which is tolerable only because the two never share a panel. The
# namespaces are separate so nobody later reads "orange, so B_x" off a
# comparison figure, or reuses one dictionary where the other was meant.

ANSATZ_COLORS = {
    "A":        "#1f6fb4",
    "A_pocket": "#4fb3e8",
    "B_z":      "#2f8f8f",
    "B_x":      "#d66014",
}

COMPARISON_COLORS = {
    "cross_ansatz": "#d66014",
    "cross_seed":   "#1f77b4",
    "ensemble":     "#388e8e",   # a pair of ensemble sizes
}

# Stray-field components. Deliberately the same two colours as the B_z and
# B_x ansaetze: the B_z ansatz is built from |d_x B_z| and B_x from |d_x B_x|,
# so a reader who carries the association across is right.
FIELD_COLORS = {
    "B_z": ANSATZ_COLORS["B_z"],
    "B_x": ANSATZ_COLORS["B_x"],
}

# A short sequence for curves that carry no document-wide meaning: the two
# gradient settings of the B1 check, the three filter curves of B3. Drawn from
# the same family as the semantic colours so the supplement does not introduce
# a second visual vocabulary. Always pair with a line style, so the series
# survives greyscale.
SERIES = ["#1f6fb4", "#d66014", "#2f8f8f", "#8452a1"]
SERIES_STYLES = ["-", "--", "-.", (0, (1, 1))]

NEUTRAL = {
    "grey": "#555555",
    "eps":  "#333333",
    "EZ":   "#b03020",
    "rule": "0.88",
}

# Kept for callers that just want a colour by name; prefer the namespaces.
PALETTE = {**ANSATZ_COLORS, **NEUTRAL,
           "ansatz": COMPARISON_COLORS["cross_ansatz"],
           "seed":   COMPARISON_COLORS["cross_seed"],
           "teal":   COMPARISON_COLORS["ensemble"]}


def tint(colour, amount=0.45):
    """A lighter version of a colour, for the second member of a pair.

    Two tones read better than a hatch at print size: hatching at 8 pt fills
    with visible moire and photocopies badly, while a tint keeps the pairing
    obvious and the panel quiet.
    """
    r, g, b = matplotlib.colors.to_rgb(colour)
    return tuple(c + (1.0 - c) * amount for c in (r, g, b))


def pair_colors(name):
    """Solid and tinted fill for a two-member bar group."""
    base = PALETTE[name]
    return base, tint(base)


def apply(base_size=9.0):
    """Set the shared rcParams. Call once, before creating a figure."""
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans",
        "font.size": base_size,
        "axes.labelsize": base_size,
        "axes.titlesize": base_size + 0.5,
        "xtick.labelsize": base_size - 0.5,
        "ytick.labelsize": base_size - 0.5,
        "legend.fontsize": base_size - 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.titlelocation": "left",
        "axes.titlepad": 4.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.linewidth": 1.3,
        "lines.markersize": 4.0,
        "grid.color": "0.88",
        "grid.linewidth": 0.6,
        "axes.grid": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def light_grid(ax, axis="y"):
    """A faint rule set on ONE axis, behind the data.

    Rules go along the axis the comparison is read on: horizontal for vertical
    bars, vertical for horizontal bars, and none at all for a scatter, where
    they add ink without helping.

    `axis="both"` is allowed for log-log panels spanning several decades, where
    the decade rules are how a reader locates a point at all. It is the only
    case where both directions earn their ink.
    """
    ax.set_axisbelow(True)
    # Style kwargs must not be passed alongside visible=False: matplotlib then
    # enables the grid and warns, which is the opposite of what is asked for.
    for which, on in (("y", axis in ("y", "both")),
                      ("x", axis in ("x", "both"))):
        gax = ax.yaxis if which == "y" else ax.xaxis
        if on:
            gax.grid(True, color=NEUTRAL["rule"], lw=0.6)
        else:
            gax.grid(False)


# Kept under the old name; light_grid(ax, "y") is the same thing.
def hgrid(ax):
    light_grid(ax, "y")


def panel_title(ax, text):
    """Left-aligned panel title.

    A left-aligned title starts at the y axis, so a long one runs past the
    panel and crowds its neighbour. Break it across two lines rather than
    letting it overhang, and give every panel in a figure the same number of
    title lines so their frames stay aligned.
    """
    ax.set_title(text, loc="left")
