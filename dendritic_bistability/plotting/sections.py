"""Figures S3–S4: conductance boundaries with the paper’s panel layout."""

import xml.etree.ElementTree as ET
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from .style import plt, save

COLORS = {0: "#6a96c8", 1: "#e4ac38", 5: "#ab75ba"}
AMPA_COLORS = {0: "#009e73", 1: COLORS[1], 5: "#d55e00"}
STYLES = {0: ":", 1: "-", 5: "--"}


def _style():
    """Set fonts, colors and axis-label sizes for Figures S3–S4."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.labelcolor": "#555555",
            "axes.edgecolor": "#555555",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "mathtext.default": "regular",
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
        }
    )


def _read(path):
    """Read a propagation grid and its spacing and imbalance coordinates."""
    with np.load(path, allow_pickle=False) as data:
        classes, bounds = data["classes"].copy(), data["bounds"].copy()
    x, r = [np.linspace(a, b, n) for (a, b), n in zip(bounds, classes.shape)]
    return classes, x, r


def _draw(ax, data, x, r, colors=COLORS):
    """Overlay order-selective ranges and ordered and swapped propagation boundaries."""
    for multiplier in (0, 1, 5):
        classes = data[multiplier]
        if np.any(classes == 1):
            ax.contourf(
                x,
                r,
                (classes == 1).T.astype(float),
                levels=[0.5, 1.5],
                colors=[colors[multiplier]],
                alpha=0.23,
                zorder=multiplier + 1,
            )
    for layer, multiplier in enumerate((1, 5, 0)):
        for bit in (1, 2):
            mask = ((data[multiplier] & bit) > 0).T.astype(float)
            if mask.min() != mask.max():
                contours = ax.contour(
                    x,
                    r,
                    mask,
                    levels=[0.5],
                    colors=[colors[multiplier]],
                    linestyles=STYLES[multiplier],
                    linewidths=1.4,
                    zorder=20 + layer,
                )
                if multiplier == 0:
                    contours.set_capstyle("round")
    ax.set_xlim(0, x[-1])
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_ylim(r[0], r[-1])
    ax.set_yticks([-0.8, -0.4, 0, 0.4])
    ax.grid(alpha=0.12)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def _axes(ax, x, row, col, *, upper_labels=True, limits=None):
    """Add physical spacing, normalized axial resistance and imbalance axes."""
    ax.grid(False)
    ax.set_xlim(0, x[-1])
    ax.set_xticks([0, 10, 20])
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="both", direction="in", width=0.6, length=3.6, pad=3)
    ax.tick_params(axis="x", which="minor", direction="in", width=0.5, length=1.2)
    if limits is None:
        limits = (-0.4, 0.4) if row == 0 else (-0.8, 0)
    ax.set_ylim(*limits)
    ticks = [-0.4, 0, 0.4] if row == 0 else [-0.8, -0.4, 0]
    if limits == (-0.7, 0.1):
        ticks = [-0.4, 0]
    ax.set_yticks(ticks)
    ax.set_yticklabels(["0" if t == 0 else ("−" if t < 0 else "") + f"{abs(t):g}".lstrip("0") for t in ticks])
    ax.set_xlabel("Δx (µm)", labelpad=3)
    if col == 0:
        ax.set_ylabel(r"$\Delta G_{\mathrm{c}}/2G_{\mathrm{c}}$", labelpad=5)
    else:
        ax.tick_params(labelleft=False)
    if row == 0 and upper_labels:
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")
    scale = 0.23059782865759065
    top = ax.secondary_xaxis("top", functions=(lambda v: v * scale, lambda v: v / scale))
    top.set_xticks([0, 2.5, 5])
    top.set_xticklabels(["0", "2.5", "5"])
    top.tick_params(direction="in", length=3.6, width=0.6, pad=3)
    if row == 0:
        top.set_xlabel(r"$R_{\mathrm{s}}G_{\mathrm{c}}$", labelpad=4)


def _subscripts(path):
    """Keep conductance and resistance subscripts editable in SVG editors."""
    ns = {"s": "http://www.w3.org/2000/svg"}
    ET.register_namespace("", ns["s"])
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.parse(path)
    for node in tree.findall(".//s:text", ns):
        spans = list(node)
        signature = "".join(span.text or "" for span in spans)
        if signature == "RGsc":
            parts = [("R", False), ("s", True), ("G", False), ("c", True)]
        elif signature == "ΔG/2Gcc":
            parts = [("ΔG", False), ("c", True), ("/2G", False), ("c", True)]
        elif signature.replace("\u00a0", " ").strip() in ("GAMPA (nS)", "G (nS)AMPA"):
            parts = [("G", False), ("AMPA", True), ("\u00a0(nS)", False)]
        else:
            continue
        for span in spans:
            node.remove(span)
        node.set("style", "font-family: Helvetica, Arial, sans-serif; font-size: 9px")
        for text, subscript in parts:
            attributes = {"font-size": "70%", "baseline-shift": "sub"} if subscript else {}
            ET.SubElement(node, "{" + ns["s"] + "}tspan", attributes).text = text
    tree.write(path, encoding="utf-8", xml_declaration=True)


def plot_conductance_ranges(folder, destination):
    """Plot scaled background conductances in A–B and large AMPA conductances in C."""
    _style()
    figure = plt.figure(figsize=(180 / 25.4, 174 / 25.4))
    for row in range(3):
        bottom, height = (0.695, 0.395, 0.095)[row], 0.21
        for col, placement in enumerate(("shaft", "head")):
            left = 0.105 if col == 0 else 0.57
            ax = figure.add_axes([left, bottom, 0.395, height])
            ax.set_gid(f"panel_{'ABC'[row]}{'i' if col == 0 else 'ii'}")
            if row < 2:
                ampa = (0.175, 1.4)[row]
                files = {key: folder / f"{placement}_{key}_{ampa:g}.npz" for key in (0, 1, 5)}
            else:
                files = {0: folder / "large_ampa" / f"{placement}_5.npz",
                         1: folder / f"{placement}_1_1.4.npz",
                         5: folder / "large_ampa" / f"{placement}_10.npz"}
            data = {}
            for key, path in files.items():
                data[key], x, r = _read(path)
            _draw(ax, data, x, r, AMPA_COLORS if row == 2 else COLORS)
            _axes(ax, x, 0 if row == 0 else 1, col, upper_labels=False)
            if row < 2:
                ax.set_xlabel("")
            figure.text(left - 0.047, bottom + height + 0.008,
                        "i" if col == 0 else "ii", fontsize=11, color="#444444", va="baseline")
            if col == 0 and row < 2:
                ax.text(0.95, 0.12 if row else 0.48,
                        rf"$G_{{\mathrm{{AMPA}}}}={ampa:g}$ nS",
                        transform=ax.transAxes, ha="right", color="#555555", fontsize=9)
        figure.text(0.018, bottom + height + 0.008, "ABC"[row],
                    fontsize=15, color="#444444", va="baseline")
    for x, title in ((0.303, "Shaft GIRK"), (0.768, "Spine-head GIRK")):
        figure.text(x, 0.98, title, ha="center", color="#555555", fontsize=9)
    handles = [Line2D([], [], color=COLORS[m], linestyle=STYLES[m], linewidth=1.6,
                      label=f"{m}×") for m in (0, 1, 5)]
    figure.axes[0].legend(handles=handles, loc="upper right", ncol=3, frameon=False,
                          handlelength=2.4, columnspacing=1.3, borderaxespad=0.8)
    handles = [Line2D([], [], color=AMPA_COLORS[k], linestyle=STYLES[k], linewidth=1.6,
                      label=f"{ampa:g}") for k, ampa in ((1, 1.4), (0, 5.0), (5, 10.0))]
    figure.axes[4].legend(handles=handles, title=r"$G_{\mathrm{AMPA}}$ (nS)",
                          loc="upper right", ncol=3, frameon=False, handlelength=2.4,
                          columnspacing=1.3, borderaxespad=0.5, title_fontsize=9)
    save(figure, destination, bounds="tight", dpi=600, padding=0.06)
    _subscripts(destination.with_suffix(".svg"))


def plot_swap_tolerance(folder, destination):
    """Plot Figure S4 ordered, adjacent-swap and two-away-swap boundaries and shaded ranges."""
    _style()
    figure = plt.figure(figsize=(180 / 25.4, 124 / 25.4))
    for row, ampa in enumerate((0.175, 1.4)):
        bottom = 0.505 if row == 0 else 0.09
        for col, placement in enumerate(("shaft", "head")):
            left = 0.105 if col == 0 else 0.57
            classes, x, r = _read(folder / f"{placement}_{ampa:g}.npz")
            ax = figure.add_axes([left, bottom, 0.395, 0.29])
            regions = (
                ((classes & 1) > 0) & ((classes & 2) == 0),
                ((classes & 3) == 3) & ((classes & 4) == 0),
            )
            for mask, color in zip(regions, ("#b58120", "#2779af")):
                if mask.any():
                    ax.contourf(x, r, mask.T.astype(float), levels=[0.5, 1.5], colors=[color], alpha=0.23)
            for bit, color in ((1, "#218c45"), (2, "#b58120"), (4, "#2779af")):
                mask = (classes & bit) > 0
                if mask.min() != mask.max():
                    ax.contour(x, r, mask.T.astype(float), levels=[0.5], colors=[color], linewidths=1.4)
            _axes(ax, x, row, col, limits=(-0.4, 0.4) if row == 0 else (-0.7, 0.1))
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
            figure.text(
                left - 0.047,
                bottom + 0.302,
                "i" if col == 0 else "ii",
                fontsize=11,
                color="#444444",
                ha="left",
                va="bottom",
            )
        figure.text(
            0.018,
            bottom + 0.303,
            "A" if row == 0 else "B",
            fontsize=15,
            color="#444444",
            ha="left",
            va="bottom",
        )
    figure.text(0.303, 0.90, "Shaft GIRK", ha="center", color="#555555", fontsize=9)
    figure.text(0.768, 0.90, "Spine-head GIRK", ha="center", color="#555555", fontsize=9)
    save(figure, destination, bounds="tight", dpi=600, padding=0.06)
    _subscripts(destination.with_suffix(".svg"))
