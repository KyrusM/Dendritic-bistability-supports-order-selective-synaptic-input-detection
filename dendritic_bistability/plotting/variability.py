"""Plot lognormal variability across GIRK placements."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


FAMILIES = ("NMDA", "AMPA", "GIRK", "SPACING", "NECK", "JOINT")
LABELS = ("NMDA", "AMPA", "GIRK", "Spacing", "Neck", "Joint")
STYLES = (
    ("ordered", "Ordered", "#303030", "o", "-"),
    ("swap", "Swap", "#D97706", "^", "--"),
    ("shuffle", "Shuffle", "#167AB5", "s", ":"),
    ("reverse", "Reverse", "#783AA6", "D", "-."),
)


def read_counts(path):
    """Index propagation percentages by GIRK placement, parameter, SD and input order."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    counts = {(row["placement"], row["parameter"], float(row["sd_percent"]), row["input_order"]):
              float(row["percent_propagated"]) for row in rows}
    return counts, sorted({float(row["sd_percent"]) for row in rows})


def heatmap(ax, counts, placement, kind, sds, color, first):
    """Draw one input-order heatmap with joint variability separated below."""
    values = np.array([[counts[placement, family, sd, kind] for sd in sds]
                       for family in FAMILIES])
    cmap = LinearSegmentedColormap.from_list(kind, ["white", color])
    edges = dict(cmap=cmap, vmin=0, vmax=100, edgecolors="#e6e6e6", linewidth=0.25)
    ax.pcolormesh(np.arange(len(sds) + 1), np.arange(6), values[:5], **edges)
    mesh = ax.pcolormesh(np.arange(len(sds) + 1), [5.25, 6.25], values[5:], **edges)
    centers = np.arange(6) + 0.5
    centers[-1] += 0.25
    ax.set(xlim=(0, len(sds)), ylim=(6.25, 0),
           yticks=centers, yticklabels=LABELS if first else [""] * 6,
           xticks=np.arange(len(sds)) + 0.5, xticklabels=[f"{sd:g}" for sd in sds])
    ax.set_xlabel("SD (% of mean)", labelpad=3)
    ax.tick_params(length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_color("#bbbbbb")
        spine.set_linewidth(0.4)
    for y in (5, 5.25):
        ax.axhline(y, color="#bbbbbb", lw=0.4)
    for row, line in enumerate(values):
        for col, value in enumerate(line):
            if 0 < value < 100:
                rgb = np.array(cmap(value / 100)[:3])
                linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
                luminance = float(linear @ [0.2126, 0.7152, 0.0722])
                ink = "white" if 1.05 / (luminance + 0.05) > (luminance + 0.05) / 0.06444 else "#202020"
                label = f"{value:.1f}".lstrip("0") if value < 1 or value > 99 else f"{value:.0f}"
                ax.text(col + 0.5, centers[row], label, ha="center", va="center", color=ink, fontsize=7)
    return mesh


def joint_summary(ax, counts, placement, sds):
    """Compare ordered, swapped, shuffled and reversed inputs under joint variability."""
    layers = {"ordered": 3, "swap": 5, "reverse": 7, "shuffle": 9}
    for kind, label, color, marker, line in STYLES:
        values = [counts[placement, "JOINT", sd, kind] for sd in sds]
        ax.plot(sds, values, label=label, color=color, marker=marker,
                ls=line, lw=0.9, ms=3, zorder=layers[kind])
    ax.set(xlim=(-0.7, max(sds) + 0.7), ylim=(-3, 105),
           xticks=[0, 5, 10, 20], yticks=[0, 50, 100])
    ax.set_xticks([2], minor=True)
    ax.set_xlabel("SD (% of mean)", labelpad=3)
    ax.set_ylabel("Propagation (%)", labelpad=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, width=0.45, pad=2)
    ax.axhline(100, color="#c6c6c6", lw=0.3, ls=":", zorder=0)


def plot_counts(
    counts, sds, destination, placements=("head", "split", "shaft"),
    placement_labels=("GIRK on spines", "Split GIRK", "GIRK on shaft"),
):
    """Compare GIRK placements with heatmaps and joint-variability summaries."""
    plt.rcParams.update({"font.family": "Arial", "font.size": 8,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.45})
    height = 2.4 if len(placements) == 1 else 3.95 + (len(placements) - 2) * 3.95 * .69 * 1.70 / 2.70
    scale = (2.4 if len(placements) == 1 else 3.95) / height
    fig = plt.figure(figsize=(5.57, height))
    grid = fig.add_gridspec(len(placements), 5, width_ratios=(1, 0.14, 1, 0.50, 1),
                            left=0.157, right=0.981, bottom=0.13 * scale, top=1 - 0.18 * scale,
                            wspace=0, hspace=0.70)
    axes = np.array([[fig.add_subplot(grid[row, col]) for col in (0, 2, 4)]
                     for row in range(len(placements))])
    kinds = ("ordered", "swap")
    colors = ("#303030", "#D97706")
    titles = ("Ordered sequences", "Adjacent swaps", "Joint variability")
    for row, placement in enumerate(placements):
        for col, (kind, color) in enumerate(zip(kinds, colors)):
            mesh = heatmap(axes[row, col], counts, placement, kind, sds, color, col == 0)
            if row == 0:
                box = axes[row, col].get_position()
                cax = fig.add_axes([box.x0, box.y1 + 0.026 * scale, box.width, 0.024 * scale])
                bar = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[])
                bar.solids.set_rasterized(False)
                bar.outline.set_linewidth(0.4)
                cax.text(0.025, 0.43, "0", transform=cax.transAxes,
                         ha="left", va="center", fontsize=6.2, color="#202020")
                cax.text(0.975, 0.43, "100", transform=cax.transAxes,
                         ha="right", va="center", fontsize=6.2,
                         color="#202020" if col == 1 else "white")
                if col == 0:
                    fig.text(box.x0 - 0.009, box.y1 + 0.038 * scale,
                             "% that propagate\nthe front", ha="right", va="center", fontsize=6.2)
        joint_summary(axes[row, 2], counts, placement, sds)
        box = axes[row, 0].get_position()
        name = placement_labels[row]
        fig.text(0.023, (box.y0 + box.y1) / 2, name,
                 rotation=90, ha="center", va="center", fontsize=10)
        heading_offset = (27 if row == 0 else 12) / (72 * fig.get_figheight())
        fig.text(0.023, box.y1 + heading_offset, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[row],
                 fontsize=10, ha="center", va="baseline")
        for col, title in enumerate(titles):
            padding = 22 if row == 0 else 7
            axes[row, col].set_title(title, fontsize=7.2, pad=padding)
            axes[row, col].annotate(("i", "ii", "iii")[col],
                                   xy=(-0.07, 1), xycoords="axes fraction",
                                   xytext=(0, padding), textcoords="offset points",
                                   fontstyle="normal", fontsize=8, va="baseline")
    axes[0, 2].legend(loc="upper right", bbox_to_anchor=(1, 0.95),
                      frameon=False, fontsize=6,
                      handlelength=1.5, handletextpad=0.4,
                      borderaxespad=0.2, labelspacing=0.15)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(destination.with_suffix("." + suffix), dpi=300, facecolor="white")
    plt.close(fig)


def plot_variability(path, destination):
    """Load propagation percentages and save the Figure S6 panels."""
    counts, sds = read_counts(Path(path))
    plot_counts(counts, sds, Path(destination))
