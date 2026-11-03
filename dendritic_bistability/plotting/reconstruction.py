"""Figure S7: reconstructed branch and current-step input resistance."""

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image

from .style import plt
from . import reconstruction_morphology as anatomy
from .reconstruction_surface import surface_png, SURFACE_BOX

STYLES = [
    ("ordered", "Ordered", "#59666c", "o", 1.0),
    ("swap", "Swap", "#e58b25", "^", 0.85),
    ("shuffle", "Shuffle", "#2585a8", "s", 0.6),
    ("reversed", "Reverse", "#9354b2", "D", 0.9),
]


def pdf_png(pdf_path, png_path, dpi):
    """Rasterize the Figure S7 PDF for PNG export."""
    command = shutil.which("pdftoppm")
    if command is None:
        raise RuntimeError("Install Poppler (pdftoppm) to export the Figure S7 PNG.")
    subprocess.run(
        [
            command,
            "-png",
            "-r",
            str(dpi),
            "-singlefile",
            str(pdf_path),
            str(png_path.with_suffix("")),
        ],
        check=True,
        capture_output=True,
    )


def kind(row):
    """Identify an input-order group from its saved name."""
    return (
        "swap"
        if row["name"].startswith("swap_")
        else "shuffle" if row["name"].startswith("shuffle_") else row["name"]
    )


def format_2d(ax):
    """Set the two-dimensional panel axes and tick sizes."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6, length=2, width=0.45, pad=1)
    ax.xaxis.label.set_size(6.5)
    ax.yaxis.label.set_size(6.5)
    ax.xaxis.labelpad = 1
    ax.yaxis.labelpad = 1


def summary(ax, rows):
    """Plot spike counts for all 28 input orders."""
    largest = max(r["soma_spike_count"] for r in rows)
    for rank, (k, label, color, marker, alpha) in enumerate(STYLES):
        group = [r for r in rows if kind(r) == k]
        group.sort(key=lambda r: int(r["name"].split("_")[1]) if k in ["swap", "shuffle"] else 0)
        spread = 0.32 if k == "swap" else 0.12
        x = rank + (np.linspace(-spread, spread, len(group)) if len(group) > 1 else np.array([0.0]))
        ax.scatter(
            x,
            [r["soma_spike_count"] for r in group],
            color=color,
            marker=marker,
            alpha=alpha,
            s=12 if k == "ordered" else 9,
            edgecolors="none",
            zorder=8 if k == "shuffle" else 4 + rank,
        )
        ax.text(
            rank,
            0.10,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            rotation=90,
            color=color,
            fontsize=6.8,
        )
    ax.set(
        xlim=(-0.40, 3.40), xticks=[], ylim=(-0.03 * largest, 1.16 * largest), yticks=[0, largest]
    )
    ax.set_ylabel("Somatic spikes", fontsize=7, labelpad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6, length=2, width=0.45, pad=1)


def rin(ax, good):
    """Fit steady voltage change against current and report the slope resistance."""
    currents = tuple(range(-40, 141, 20))
    display = sorted((row for row in good if row["current_pA"] in currents),
                     key=lambda row: row["current_pA"])
    if len(display) != len(currents):
        raise ValueError("The input-resistance fit requires all ten subthreshold current steps.")
    current = np.array([row["current_pA"] for row in display])
    response = np.array([row["steady_local_mv"] - row["baseline_local_mv"] for row in display])
    slope, intercept = np.polyfit(current, response, 1)
    residual = response - (slope * current + intercept)
    r_squared = 1 - np.sum(residual ** 2) / np.sum((response - response.mean()) ** 2)
    grid = np.linspace(current.min(), current.max(), 200)
    ax.plot(grid, slope * grid + intercept, color="#777777", lw=0.9)
    ax.plot(current, response, "o", color="black", ms=2.4, mec="white", mew=0.25)
    ax.set(
        xlim=(-50, 150), xticks=[-40, 0, 40, 80, 120],
        ylim=(-12, 40), yticks=[-10, 0, 10, 20, 30, 40],
        xlabel=r"$I$ (pA)", ylabel=r"$\Delta u$ (mV)",
    )
    format_2d(ax)
    ax.text(0.04, 0.96, f"$R_{{\\mathrm{{in}}}}$ = {1000 * slope:.1f} MΩ\n$R^2$ = {r_squared:.4f}",
            transform=ax.transAxes, va="top", fontsize=6)
    return dict(resistance_mohm=float(1000 * slope), r_squared=float(r_squared),
                intercept_mv=float(intercept), current_range_pa=[-40, 140], point_count=len(display))


def axis_labels(fig, image_ax, surface_report, anatomy_report):
    """Match the lowercase axis wording and orientation of manuscript Fig. 1Ai."""
    fw, fh = fig.get_size_inches()
    pixels = image_ax.images[0].get_array().shape[:2]

    def point(xy):
        """Map a surface image point into figure inches."""
        return (
            image_ax.transData.transform([xy[0] * pixels[1] - 0.5, xy[1] * pixels[0] - 0.5])
            / fig.dpi
        )

    def box(q):
        """Return a tick label's bounding box in figure inches."""
        b = q["bbox_fraction"]
        a = point(b[:2])
        c = point(b[2:])
        return np.array([min(a[0], c[0]), min(a[1], c[1]), max(a[0], c[0]), max(a[1], c[1])])

    tick_labels = surface_report["tick_labels"]["labels"]
    time = {q["text"]: box(q) for q in tick_labels if q["axis"] == "time"}
    p0 = (time["0"][:2] + time["0"][2:]) / 2
    p1 = (time["200"][:2] + time["200"][2:]) / 2
    tangent = (p1 - p0) / np.linalg.norm(p1 - p0)
    at_time = (p0 + p1) / 2 + 0.13 * np.array([-tangent[1], tangent[0]])
    voltage = [box(q) for q in tick_labels if q["axis"] == "voltage"]
    at_voltage = [
        min(b[0] for b in voltage) - 0.10,
        (min(b[1] for b in voltage) + max(b[3] for b in voltage)) / 2,
    ]
    anchors = {k: point(v) for k, v in surface_report["image_anchors"].items()}
    at_position = [
        (anchors["frame_tip"][0] + anchors["frame_soma"][0]) / 2,
        anatomy_report["axes_box_inches"][1] - 0.015,
    ]
    positions = [
        ("position (µm)", at_position, anatomy.camera_angle()),
        ("time (ms)", at_time, np.rad2deg(np.arctan2(tangent[1], tangent[0]))),
        ("voltage (mV)", at_voltage, 90),
    ]
    artists = [
        fig.text(
            xy[0] / fw,
            xy[1] / fh,
            text,
            fontsize=6.5,
            color=".25",
            rotation=angle,
            ha="center",
            va="center",
            rotation_mode="anchor",
        )
        for text, xy, angle in positions
    ]
    return artists, [
        dict(text=t, position_inches=list(map(float, xy)), rotation_degrees=float(a))
        for t, xy, a in positions
    ]


def morphology_payload(geometry):
    """Arrange the cable sections and their projection for the anatomical inset."""
    soma = dict(geometry["soma"], name="soma")
    shaft = [dict(q, name=f"cell_shaft_{j}") for j, q in enumerate(geometry["shaft"])]
    spines = [
        dict(q[key], name=f"{key}_{q['id']}")
        for q in geometry["spines"]
        for key in ["neck", "head"]
    ]
    return dict(geometry["projection"], soma=soma, sections=[soma] + shaft + spines)


def plot_reconstruction(results, geometry_path, destination):
    """Render the S7 sequence and current-step panels."""
    if shutil.which("pdftoppm") is None:
        raise RuntimeError("Install Poppler (pdftoppm) to export the Figure S7 PNG.")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.5,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "savefig.transparent": True,
        }
    )
    results, destination = Path(results), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    geometry = json.loads(Path(geometry_path).read_text())
    rows = json.loads((results / "sequences.json").read_text())
    good = sorted(
        [
            r
            for r in json.loads((results / "resistance.json").read_text())
            if r["valid_subthreshold"]
        ],
        key=lambda r: r["steady_local_mv"],
    )
    payload = morphology_payload(geometry)
    angle = anatomy.camera_angle()
    width, height = 6.70, 2.10
    top = height - 0.035
    fig = plt.figure(figsize=(width, height))
    fig.text(0.015 / width, top / height, "A", fontsize=9, weight="normal", va="top")
    surface_reports = []
    for col, case in enumerate(["ordered", "swap_3_4"]):
        row = dict(
            next(r for r in rows if r["name"] == case),
            recording_path=results / "orders" / (case + ".npz"),
        )
        image_path = destination / ("Ai.png" if col == 0 else "Aii.png")
        surface_reports.append(surface_png(row, image_path, geometry))
        left = 0.155 if col == 0 else 1.80
        ax = fig.add_axes(
            [left / width, 0.42 / height, SURFACE_BOX[0] / width, SURFACE_BOX[1] / height]
        )
        with Image.open(image_path) as image:
            ax.imshow(np.asarray(image), interpolation="lanczos")
        if col == 0:
            ordered_image_ax = ax
        ax.axis("off")
        fig.text(
            (left + 0.09) / width,
            (top - 0.22) / height,
            "i" if col == 0 else "ii",
            fontsize=7,
            fontstyle="normal",
        )
    anatomy_report, _ = anatomy.place(fig, ordered_image_ax, payload, angle, surface_reports[0])
    axis_labels(fig, ordered_image_ax, surface_reports[0], anatomy_report)
    ax = fig.add_axes(
        [4.10 / width, 0.30 / height, 0.60 / width, 1.57 / height]
    )
    summary(ax, rows)
    fig.text(
        3.95 / width,
        (top - 0.07) / height,
        "iii",
        fontsize=7,
        fontstyle="normal",
    )
    ax = fig.add_axes(
        [5.16 / width, 0.30 / height, 1.25 / width, 1.57 / height]
    )
    fit = rin(ax, good)
    (destination / "resistance_fit.json").write_text(json.dumps(fit, indent=2) + "\n")
    fig.text(
        4.84 / width,
        top / height,
        "B",
        fontsize=9,
        weight="normal",
        va="top",
    )
    fig.text(
        5.785 / width,
        top / height,
        "Input resistance",
        fontsize=7.2,
        ha="center",
        va="top",
    )
    fig.canvas.draw()
    for ext in ["svg", "pdf"]:
        fig.savefig(destination / ("FigureS7." + ext), dpi=600)
    pdf_png(destination / "FigureS7.pdf", destination / "FigureS7.png", 600)
    plt.close(fig)
