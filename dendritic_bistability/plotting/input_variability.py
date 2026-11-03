"""Figure S5: front propagation under timing jitter and additional synaptic inputs."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .response_style import STYLES, LAYERS


PANELS = (
    ("C", "Timing jitter", "jitter_sd_ms", "Timing jitter SD (ms)"),
    ("D", "Glutamate at sequence\nsynapses", "extra_count", "Additional inputs"),
    ("E", "Glutamate at\nintervening spines", "extra_count", "Additional inputs"),
    ("F", r"Shaft GABA$_{\mathrm{A}}$ inputs", "extra_count", "Additional inputs"),
)


def propagation_counts(rows, models=("head", "split", "shaft")):
    """Count successful trials by GIRK placement, perturbation and input order."""
    counts = []
    for model in models:
        for stage, _, key, _ in PANELS:
            selected = [r for r in rows if r["model"] == model and r["stage"] == stage]
            for value in sorted({r[key] for r in selected}):
                for kind, *_ in STYLES:
                    trials = [r for r in selected if r[key] == value and r["kind"] == kind]
                    if not trials or not all(isinstance(r["propagated"], bool) for r in trials):
                        raise ValueError("Each input order requires saved propagation classifications.")
                    successes = sum(r["propagated"] for r in trials)
                    counts.append(dict(model=model, stage=stage, perturbation=value,
                                       input_order=kind, propagated=successes,
                                       trials=len(trials), percent=100 * successes / len(trials)))
    return counts


def plot_input_variability(
    path, destination, models=("head", "split", "shaft"),
    model_labels=("GIRK on spines", "Split GIRK", "GIRK on shaft"),
):
    """Compare GIRK placements under timing jitter and additional inputs."""
    counts = propagation_counts(json.loads(Path(path).read_text()), models)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).with_name("propagation_counts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(counts[0]))
        writer.writeheader()
        writer.writerows(counts)
    plt.rcParams.update({"font.family": "Arial", "font.size": 7.5,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial"})
    height_mm = 48 if len(models) == 1 else 85 + (len(models) - 2) * 85 * .68 * 1.62 / 2.62
    scale = (48 if len(models) == 1 else 85) / height_mm
    fig, axes = plt.subplots(len(models), 4, figsize=(180 / 25.4, height_mm / 25.4), sharey=True, squeeze=False)
    for row, model in enumerate(models):
        for col, (stage, title, key, xlabel) in enumerate(PANELS):
            ax = axes[row, col]
            for kind, color, marker, line in STYLES:
                group = [r for r in counts if r["model"] == model
                         and r["stage"] == stage and r["input_order"] == kind]
                label = {"ordered": "Ordered", "adjacent_swap": "Adjacent swap",
                         "shuffle": "Shuffled", "reversed": "Reversed"}[kind]
                ax.plot([r["perturbation"] for r in group], [r["percent"] for r in group],
                        label=label, color=color, marker=marker, ls=line,
                        lw=.85, ms=2.7, markeredgewidth=.5,
                        markerfacecolor="white" if kind != "ordered" else color,
                        zorder=LAYERS[kind])
            if row == 0:
                ax.set_title(title, fontsize=8.5, pad=6)
            ax.set_xlabel(xlabel, fontsize=8, labelpad=3)
            ax.set_ylim(-4, 105)
            ax.set_yticks([0, 50, 100])
            ax.axhline(100, color="#cccccc", lw=.35, ls=":", zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=2, width=.45, pad=2, labelsize=7)
            for spine in ax.spines.values():
                spine.set_linewidth(.45)
            if key == "jitter_sd_ms":
                ax.set_xscale("symlog", linthresh=1, linscale=.4)
                ax.set_xlim(-.1, 23)
                ax.set_xticks([0, 1, 2, 5, 10, 20], ["0", "1", "2", "5", "10", "20"])
            else:
                ax.set_xlim(-.6, 20.6)
                ax.set_xticks([0, 4, 8, 12, 16, 20])
        axes[row, 0].set_ylabel("Front propagation (%)", fontsize=8, labelpad=3)
    axes[0, 0].legend(loc="center left", bbox_to_anchor=(.01, .50),
                      frameon=False, fontsize=7, handlelength=1.3, labelspacing=.4,
                      borderaxespad=0, handletextpad=.4)
    fig.subplots_adjust(left=.13, right=.99, bottom=.14 * scale,
                        top=1 - .18 * scale, wspace=.28, hspace=.62)
    for col, label in enumerate("ABCD"):
        fig.text(axes[0, col].get_position().x0 - .02, 1 - .065 * scale,
                 label, fontsize=10, va="top")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for row, label in enumerate(model_labels):
        box = axes[row, 0].get_position()
        ylabel = axes[row, 0].yaxis.label.get_window_extent(renderer)
        left = ylabel.transformed(fig.transFigure.inverted()).x0
        fig.text(left - 3.5 / 180, (box.y0 + box.y1) / 2, label,
                 rotation=90, ha="center", va="center", fontsize=9)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(destination.with_suffix("." + extension), dpi=600, facecolor="white")
    plt.close(fig)
