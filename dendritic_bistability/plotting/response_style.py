"""Shared colors, markers and percentage axes for sequence responses."""

from matplotlib.ticker import PercentFormatter

STYLES = (
    ("ordered", "#303030", "o", "-"),
    ("adjacent_swap", "#D97706", "^", "--"),
    ("shuffle", "#167AB5", "s", ":"),
    ("reversed", "#783AA6", "D", "-."),
)
LAYERS = dict(ordered=3, adjacent_swap=5, reversed=7, shuffle=9)


def configure_axes(ax, first):
    """Set percentage ticks, the response label and the 100% reference line."""
    ax.set_ylim(-3, 116)
    ax.set_yticks([0, 50, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(100, 0))
    ax.axhline(100, color="#bdbdbd", lw=0.35, ls=":", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
    ax.tick_params(length=2, width=0.45, pad=2, labelsize=7)
    if first:
        label = ax.set_ylabel(r"Peak $\Delta\mathit{u}/\Delta\mathit{u}_{\max}$", fontsize=8, labelpad=3)
        label.set_math_fontfamily("cm")
    else:
        ax.tick_params(labelleft=False)
