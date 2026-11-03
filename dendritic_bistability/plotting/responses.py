"""Figure 1: individual sequence responses normalized within each model."""

import json
import numpy as np
from matplotlib.ticker import PercentFormatter

from .style import plt, save

STYLES = (
    ("ordered", "#59666c", "o", 1),
    ("adjacent_swap", "#e58b25", "^", 0.85),
    ("shuffle", "#2585a8", "s", 0.6),
    ("reversed", "#9354b2", "D", 0.9),
)


def _summary(ax, rows, labels):
    """Plot the ordered, adjacent-swap, shuffled and reversed responses as percentage point groups."""
    maximum = max(r["peak_shaft_response_mv"] for r in rows)
    for rank, (kind, color, marker, alpha) in enumerate(STYLES):
        group = [r for r in rows if r["kind"] == kind]
        group.sort(
            key=lambda r: (
                next(i for i, site in enumerate(r["order"]) if site != i + 1)
                if kind == "adjacent_swap"
                else int(r["name"].rsplit("_", 1)[-1]) if kind == "shuffle" else 0
            )
        )
        spread = 0.32 if kind == "adjacent_swap" else 0.12
        x = rank + (np.linspace(-spread, spread, len(group)) if len(group) > 1 else np.array([0.0]))
        ax.scatter(
            x,
            [100 * r["peak_shaft_response_mv"] / maximum for r in group],
            color=color,
            marker=marker,
            alpha=alpha,
            s=12 if kind == "ordered" else 9,
            edgecolors="none",
            zorder=4 + rank,
        )
    ax.set(xlim=(-0.40, 3.40), ylim=(-3, 116), xticks=[], yticks=(0, 100))
    ax.yaxis.set_major_formatter(PercentFormatter(100, 0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7, length=2, width=0.45, pad=1)
    ax.hlines(100, -0.55, 3.55, color="#b5b5b5", ls=":", lw=0.45, zorder=0)
    ax.set_ylabel(r"Peak $\Delta u/\Delta u_{\max}$", fontsize=7, labelpad=1)
    ax.text(
        0.5,
        104,
        rf"$\Delta u_{{\max}}={maximum:.1f}$ mV",
        transform=ax.get_yaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=5.3,
        color="#666666",
    )
    if labels:
        for x, label, (_, color, _, _) in zip(range(4), ("Ordered", "Swap", "Shuffle", "Reverse"), STYLES):
            ax.text(
                x,
                0.10,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                rotation=90,
                color=color,
                fontsize=6.8,
            )


def _close_y_label(ax):
    """Place the response label one point from the percentage axis."""
    ax.yaxis.set_label_coords(-0.01, 0.445)
    figure = ax.figure
    figure.canvas.draw()
    box = ax.yaxis.label.get_window_extent(figure.canvas.get_renderer())
    axis_x = ax.transAxes.transform((0, 0.445))[0]
    shift = (axis_x - figure.dpi / 72 - box.x1) / ax.bbox.width
    ax.yaxis.set_label_coords(-0.01 + shift, 0.445)


def plot_responses(path, destination):
    """Save the percentage-response panels Aiii and Biii using the Figure 1 dimensions."""
    rows = json.loads(path.read_text())
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.5,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
        }
    )
    for model, panel in (("impedance", "Aiii"), ("front", "Biii")):
        if not any(r["model"] == model for r in rows):
            continue
        figure = plt.figure(figsize=(1.05, 1.42))
        ax = figure.add_axes([0.38, 0.035, 0.58, 0.935])
        _summary(ax, [r for r in rows if r["model"] == model], labels=model == "impedance")
        _close_y_label(ax)
        save(figure, destination / panel, bounds=figure.bbox_inches, dpi=600)
