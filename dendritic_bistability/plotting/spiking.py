"""Figure 8: somatic spike counts across soma gaps and imposed-spike timings."""

import json
import numpy as np

from .style import plt, save

STYLES = (
    ("ordered", "Ordered", "#59666c", "o", 3),
    ("adjacent_swap", "Swap", "#ec8c20", "^", 5),
    ("shuffle", "Shuffle", "#268db4", "s", 9),
    ("reversed", "Reverse", "#9254ae", "D", 7),
)


def _counts(ax, rows):
    """Plot individual spike counts in four gap groups with the corresponding axial resistance."""
    centers = np.arange(4) * 3.25
    for i, gap in enumerate((5, 10, 20, 30)):
        for j, (kind, _, color, marker, layer) in enumerate(STYLES):
            selected = sorted(
                [r for r in rows if r["gap_um"] == gap and r["kind"] == kind], key=lambda r: r["name"]
            )
            if not selected:
                continue
            offsets = np.linspace(-0.22, 0.22, len(selected)) if len(selected) > 1 else np.zeros(1)
            ax.scatter(
                centers[i] + 0.65 * j + offsets,
                [len(r["somatic_spikes_ms"]) for r in selected],
                s=9,
                marker=marker,
                color=color,
                alpha=0.45 if kind == "shuffle" else 0.9,
                edgecolors="none",
                zorder=layer,
                clip_on=False,
            )
        if i < 3:
            ax.axvline(centers[i] + 2.60, color="#d4d4d4", lw=0.45, zorder=0)
    ax.set(xlim=(-0.45, centers[-1] + 2.4), ylim=(-0.14, 4.35), yticks=[0, 2, 4])
    ax.set_xticks(centers + 0.975, ["5", "10", "20", "30"])
    ax.set_ylabel("Somatic spikes", fontsize=6.4, labelpad=1)
    ax.set_xlabel("Final spine–soma\ngap (µm)", fontsize=6.4, labelpad=2)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.5)
    scale = 0.04 * 400 / (np.pi * 0.3**2) / 1000
    top = ax.secondary_xaxis("top")
    top.set_xticks(centers + 0.975, [f"{scale*gap:.1f}" for gap in (5, 10, 20, 30)])
    top.set_xlabel("Axial resistance\n(GΩ)", fontsize=5.8, labelpad=2)
    top.tick_params(labelsize=5.8, length=1.7, width=0.5, pad=1)
    top.spines["top"].set_linewidth(0.5)
    for (_, label, color, _, _), y in zip(STYLES, (4.0, 3.5, 3.0, 2.5)):
        ax.text(
            0.99,
            y,
            label,
            color=color,
            fontsize=5.6,
            ha="right",
            va="center",
            transform=ax.get_yaxis_transform(),
        )


def _timings(ax, rows, controls):
    """Plot sequence-evoked spikes against measured imposed-spike timing, excluding the imposed spike."""
    if not rows:
        return
    maximum = max(r["further_spike_count"] for r in rows)
    styles = (
        ("ordered", "Ordered", "#59666c", "-"),
        ("swap_4_5", "Swap 4/5", "#ec8c20", "--"),
        ("swap_7_8", "Swap 7/8", "#b56617", ":"),
    )
    for kind, label, color, line in styles:
        selected = sorted([r for r in rows if r["kind"] == kind], key=lambda r: r["imposed_spike_ms"])
        if not selected:
            continue
        order = selected[0]["order"]
        control = next(
            len(r["somatic_spikes_ms"]) for r in controls if r["gap_um"] == 10 and r["order"] == order
        )
        maximum = max(maximum, control)
        ax.hlines(control, 4, 196, color=color, ls="--", lw=0.65, alpha=0.7, zorder=1)
        times = [r["imposed_spike_ms"] - r["spike_target_ms"] + r["target_after_seed_ms"] for r in selected]
        ax.plot(
            times,
            [r["further_spike_count"] for r in selected],
            color=color,
            ls=line,
            lw=0.8,
            label=label,
            zorder=3,
        )
    ax.set(xlim=(0, 200), ylim=(-0.17, maximum + 0.45), xticks=[10, 90, 190], yticks=list(range(maximum + 1)))
    ax.set_xlabel("Spike time from\nseed (ms)", fontsize=6.4, labelpad=2)
    ax.set_ylabel("Sequence-evoked spikes", fontsize=6.4, labelpad=1)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.5)
    ax.legend(
        handles=list(ax.lines),
        frameon=False,
        fontsize=5.2,
        loc="center left",
        bbox_to_anchor=(0.0, 0.40),
        borderaxespad=0,
        borderpad=0.2,
        handlelength=1.1,
        handletextpad=0.25,
    )


def plot_spiking(path, destination):
    """Save panels Aiii and Biii using the Figure 8 dimensions, markers and line styles."""
    rows = json.loads(path.read_text())
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "mathtext.fontset": "cm",
        }
    )
    controls = [r for r in rows if r["panel"] == "A"]
    for panel in ("A", "B"):
        figure = plt.figure(figsize=(1.45, 1.50))
        ax = figure.add_axes([0.30, 0.22, 0.66, 0.55])
        if panel == "A":
            _counts(ax, controls)
        else:
            _timings(ax, [r for r in rows if r["panel"] == "B"], controls)
        ax.tick_params(labelsize=6, length=2, width=0.45, pad=1)
        save(figure, destination / (panel + "iii"), bounds=figure.bbox_inches, dpi=600)
