"""Conductance boundaries and minimum spacing."""

from pathlib import Path
import json

import numpy as np
from .style import plt, save


def read_grid(path):
    """Read propagation classes and coordinate bounds from an NPZ or JSON grid."""
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return np.asarray(data["classes"]), np.asarray(data["bounds"])
    with np.load(path, allow_pickle=False) as data:
        return data["classes"].copy(), data["bounds"].copy()


def plot_surfaces(files, destination):
    """Plot three-dimensional ordered and swapped propagation boundaries."""
    from skimage.measure import marching_cubes
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    figure = plt.figure(figsize=(3.4 * len(files), 3.1))
    for i, path in enumerate(files):
        grid, bounds = read_grid(path)
        step = (bounds[:, 1] - bounds[:, 0]) / (np.asarray(grid.shape) - 1)
        ax = figure.add_subplot(1, len(files), i + 1, projection="3d")
        surfaces = ((0, "#30a45b"), (1, "#d45048"))
        for bit, color in surfaces:
            field = ((grid >= 0) & ((grid & (1 << bit)) > 0)).astype(float)
            if field.min() == field.max():
                continue
            vertices, faces, _, _ = marching_cubes(field, 0.5, spacing=step, allow_degenerate=False)
            vertices += bounds[:, 0]
            ax.add_collection3d(
                Poly3DCollection(
                    vertices[faces], facecolor=color, alpha=0.35, edgecolor="none", rasterized=True
                )
            )
        ax.set(
            xlim=bounds[0],
            ylim=bounds[1],
            zlim=bounds[2],
            xlabel=r"$\Delta x$ (µm)",
            ylabel=r"$\Delta G_c/(2G_c)$",
            zlabel=r"$G_{\mathrm{AMPA}}$ (nS)",
        )
        ax.set_title("Shaft GIRK" if i == 0 else "Spine-head GIRK")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_visible(False)
            axis._axinfo["grid"]["linewidth"] = 0.3
            axis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.6)
        ax.view_init(25, -60)
    save(figure, destination)


def plot_spacing_minima(files, destination, width=0.1):
    """Plot minimum spacing and qualifying imbalance intervals for the largest order-selective component."""
    from ..sweeps import minimum_spacing

    figure, (band, ax) = plt.subplots(
        1, 2, figsize=(4.5, 3.0), gridspec_kw={"width_ratios": [1, 2]}, sharey=True
    )
    key = "ampa_ns"
    conversion = 4.0 * 4.0 / (np.pi * 0.3**2) * 4.075 / 1000.0
    for path, color in zip(files, (".4", ".7")):
        grid, bounds = read_grid(path)
        curve = minimum_spacing(grid, bounds, width, largest_component=True, control_name=key)
        points = [p for p in curve if p["spacing_um"] is not None]
        z = [p[key] for p in points]
        distance = [p["spacing_um"] for p in points]
        ax.plot(distance, z, color=color, label=Path(path).stem.split("_")[-1])
        for i in range(max((len(p["imbalance_ranges"]) for p in points), default=0)):
            limits = [
                p["imbalance_ranges"][i] if i < len(p["imbalance_ranges"]) else [np.nan, np.nan]
                for p in points
            ]
            band.fill_betweenx(z, np.array(limits)[:, 0], np.array(limits)[:, 1], color=color, alpha=0.3)
    band.set(xlabel=r"$\Delta G_c/(2G_c)$", ylabel=r"$G_{\mathrm{AMPA}}$ (nS)")
    ax.set_xlabel(r"$\Delta x$ (µm)")
    top = ax.secondary_xaxis("top", functions=((lambda x: conversion * x), (lambda x: x / conversion)))
    top.set_xlabel(r"$R_sG_c$")
    ax.legend(frameon=False)
    figure.tight_layout()
    save(figure, destination)


def plot_minima(curves, destination):
    """Plot Figure 7 minimum-spacing curves with their conductance-imbalance projections."""
    from .projections import project_minima
    from matplotlib.lines import Line2D
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as PlotPath

    data = project_minima(curves)
    figure = plt.figure(figsize=(7.1, 4.2))
    outer = figure.add_gridspec(1, 2, left=0.07, right=0.99, bottom=0.15, top=0.94, wspace=0.23)
    for column, placement in enumerate(("shaft", "head")):
        layout = outer[column].subgridspec(
            2, 2, width_ratios=(1, 2.2), height_ratios=(0.8, 1.25), wspace=0.1, hspace=0.12
        )
        key = figure.add_subplot(layout[0, 0])
        key.axis("off")
        key.set_title("GIRK on " + placement, fontsize=8, loc="left")
        top = figure.add_subplot(layout[0, 1])
        left = figure.add_subplot(layout[1, 0])
        main = figure.add_subplot(layout[1, 1])
        legend = []
        for q in sorted(data["curves"], key=lambda row: row["width"]):
            if q["placement"] != placement:
                continue
            color, width = q["color"], q["width"]
            for line in q["main"]:
                main.plot(*np.asarray(line).T, color=color, lw=1.2)
            for ax, prefix in ((left, "left"), (top, "top")):
                for line in q[prefix + "_lines"]:
                    ax.plot(*np.asarray(line).T, color=color, lw=0.45)
                paths = []
                for polygon in q[prefix + "_polygons"]:
                    vertices = np.asarray(polygon)
                    area = np.sum(
                        vertices[:, 0] * np.roll(vertices[:, 1], -1)
                        - vertices[:, 1] * np.roll(vertices[:, 0], -1)
                    )
                    if area < 0:
                        vertices = vertices[::-1]
                    paths.append(
                        PlotPath(
                            np.vstack([vertices, vertices[0]]),
                            [PlotPath.MOVETO]
                            + [PlotPath.LINETO] * (len(vertices) - 1)
                            + [PlotPath.CLOSEPOLY],
                        )
                    )
                if paths:
                    ax.add_patch(
                        PathPatch(
                            PlotPath.make_compound_path(*paths),
                            facecolor=color,
                            edgecolor="none",
                            alpha=0.12 if width == 0.1 else 0.08,
                        )
                    )
            if width == 0.1:
                top.plot(*np.asarray(q["hairline"]).T, color=q["reference_color"], lw=0.35)
                units = "nS/µm" if placement == "shaft" else "nS/head"
                legend.append(Line2D([], [], color=color, label=f"{q['kir']:g}"))
        r = data["displayed_r_range"][placement]
        x = data["x_ranges"][placement]
        main.set(xlim=x, ylim=(0, 1.4), xlabel=r"$\Delta x$ (µm)")
        left.set(xlim=r, ylim=(0, 1.4), xlabel=r"$\Delta G_c/(2G_c)$", ylabel=r"$G_{\mathrm{AMPA}}$ (nS)")
        top.set(xlim=x, ylim=(-r[1], -r[0]))
        from matplotlib.ticker import FuncFormatter

        top.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: "0" if abs(value) < 1e-12 else f"{-value:g}")
        )
        main.tick_params(labelleft=False)
        top.tick_params(labelbottom=False)
        if placement == "head":
            for ax in (main, top):
                ax.axvline(10 / 1.4, color="black", lw=0.35)
        key.legend(
            handles=legend,
            title=units,
            title_fontsize=5.5,
            frameon=False,
            fontsize=5.5,
            loc="upper left",
            borderaxespad=0,
        )
    save(figure, destination)
