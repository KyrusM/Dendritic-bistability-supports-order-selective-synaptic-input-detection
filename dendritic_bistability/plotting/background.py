"""Figure S1: input-order and input-interval comparisons across four models."""

import json
import matplotlib
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.transforms import Bbox

from .style import plt, save
from .response_style import STYLES, LAYERS, configure_axes

MODELS = (
    ("head", "GIRK channels on spines"),
    ("shaft", "GIRK channels on the shaft"),
    ("impedance", "Without GIRK channels"),
    ("Branco", "Branco et al. model"),
)
TITLES = (
    "Tolerance to swaps and shuffles",
    "Tolerance to different input intervals",
)


def _orders(ax, rows, maximum, model):
    """Plot sequence responses in the 5-, 20- and 50-ms timing groups."""
    for j, interval in enumerate((5, 20, 50)):
        for rank, (kind, color, marker, _) in enumerate(STYLES):
            group = [r for r in rows if r["interval_ms"] == interval and r["kind"] == kind]
            group.sort(
                key=lambda r: (
                    next(i for i, site in enumerate(r["order"]) if site != i + 1)
                    if kind == "adjacent_swap"
                    else r["name"]
                )
            )
            spread = 0.38 if kind == "adjacent_swap" else 0.14
            offsets = np.linspace(-spread, spread, len(group)) if len(group) > 1 else np.array([0.0])
            ax.scatter(
                5 * j + rank + offsets,
                [100 * r["peak_shaft_response_mv"] / maximum for r in group],
                s=12 if kind == "ordered" else 8,
                marker=marker,
                color=color,
                alpha=0.38 if kind == "shuffle" else 0.8 if rank else 1,
                edgecolors="none",
                zorder=LAYERS[kind],
            )
    for x in (4, 9):
        ax.vlines(x, 0, 100, color="#c8c8c8", lw=0.3, zorder=0)
    ax.set_xlim(-0.6, 13.6)
    ax.set_xticks([1.5, 6.5, 11.5], ["5 ms", "20 ms", "50 ms"])
    ax.tick_params(axis="x", which="major", length=0, pad=4, labelsize=7.5)
    ax.set_xticks([5 * j + k for j in range(3) for k in range(4)], minor=True)
    ax.tick_params(axis="x", which="minor", length=2, width=0.45)
    for i, tick in enumerate(ax.xaxis.get_minor_ticks()):
        tick.tick1line.set_color(STYLES[i % 4][1])
    fifty = [r for r in rows if r["interval_ms"] == 50]
    floor = min(100 * r["peak_shaft_response_mv"] / maximum for r in fifty)
    ordered = next(100 * r["peak_shaft_response_mv"] / maximum for r in fifty if r["kind"] == "ordered")
    for rank, ((kind, color, _, _), label) in enumerate(
        zip(STYLES, ("Ordered", "Swap", "Shuffle", "Reverse"))
    ):
        top = (
            floor - 5
            if model == "impedance"
            else ordered - 5 if model == "head" and kind == "ordered" else 95.0
        )
        ax.text(10 + rank, top, label, color=color, fontsize=7, rotation=90, ha="center", va="top", zorder=15)


def _timing(ax, rows, maximum):
    """Plot median responses across input intervals on a logarithmic axis."""
    intervals = sorted({r["interval_ms"] for r in rows})
    for kind, color, marker, line in STYLES:
        groups = [
            [r for r in rows if r["interval_ms"] == interval and r["kind"] == kind] for interval in intervals
        ]
        ax.plot(
            intervals,
            [100 * np.median([r["peak_shaft_response_mv"] for r in g]) / maximum for g in groups],
            color=color,
            marker=marker,
            ms=2.6,
            lw=0.8,
            ls=line,
            zorder=LAYERS[kind],
        )
    ax.set_xscale("log")
    ax.set_xlim(0.4, 100)
    ax.xaxis.set_major_locator(FixedLocator([1, 10]))
    ax.xaxis.set_minor_locator(FixedLocator(np.r_[np.arange(4, 10) / 10, np.arange(2, 10), np.arange(2, 9) * 10]))

    def label(value, choices):
        """Label only the specified input intervals."""
        return next(
            (text for number, text in choices.items() if np.isclose(value, number, rtol=1e-8, atol=0)), ""
        )

    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: label(value, {1: "1"})))
    ax.xaxis.set_minor_formatter(
        FuncFormatter(lambda value, pos: label(value, {0.5: "0.5", 5: "5", 20: "20", 80: "80"}))
    )
    ax.tick_params(axis="x", which="major", bottom=True, length=3.5, width=0.55, pad=2, labelsize=7)
    ax.tick_params(axis="x", which="minor", bottom=True, length=1.7, width=0.4, pad=2, labelsize=7)
    ax.set_xlabel("Input interval (ms)", fontsize=8, labelpad=3)


def plot_background_comparison(path, destination):
    """Compare input orders and input intervals using normalized peak responses."""
    rows = json.loads(path.read_text())
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "mathtext.fontset": "cm",
            "mathtext.rm": "Arial",
            "font.sans-serif": matplotlib.rcParamsDefault["font.sans-serif"],
        }
    )
    figure = plt.figure(figsize=(180 / 25.4, 260 / 25.4))
    left, right, gap = 0.085, 0.991, 0.035
    width = (right - left - 3 * gap) / 4
    tops = np.array([239.1, 203.9, 163.2, 123.2, 82.5, 41.8]) / 260
    height = 20.9 / 260
    for j, (_, title) in enumerate(MODELS):
        figure.text(
            left + j * (width + gap) + width / 2, 249.22 / 260, title, fontsize=9, ha="center", va="top"
        )
    for i, (stage, title) in enumerate(zip("AB", TITLES[:2])):
        figure.text(0.015, tops[i] + 2.2 / 260, stage, fontsize=10, va="bottom")
        text = figure.text(left, tops[i] + 2.2 / 260, title, fontsize=8.5, va="bottom")
        for j, (model, _) in enumerate(MODELS):
            # The cylindrical timing sweep includes the first row's three input intervals.
            stages = "AB" if stage == "B" and model != "Branco" else stage
            selected = [r for r in rows if r["stage"] in stages and r["model"] == model]
            ax = figure.add_axes([left + j * (width + gap), tops[i] - height, width, height])
            configure_axes(ax, j == 0)
            ax.set_ylim(-3, 122)
            expected = {"A": 90, "B": 240}[stage]
            if len(selected) != expected:
                ax.set_xticks([])
                continue
            maximum = max(r["peak_shaft_response_mv"] for r in selected)
            if stage == "A":
                _orders(ax, selected, maximum, model)
            elif stage == "B":
                _timing(ax, selected, maximum)
            ax.text(
                0.5,
                105,
                rf"$\Delta u_{{\max}}={maximum:.1f}$ mV",
                ha="center",
                va="bottom",
                transform=ax.get_yaxis_transform(),
                fontsize=7,
                color="#666666",
            )
    bounds = Bbox.from_bounds(0, 172 / 25.4, 180 / 25.4, 81 / 25.4)
    save(figure, destination, bounds=bounds, dpi=600)
