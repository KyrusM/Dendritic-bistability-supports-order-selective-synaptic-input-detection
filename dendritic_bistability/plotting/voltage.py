"""Voltage surfaces for the sequence examples in Figures 1 and 8."""

import numpy as np
from PIL import Image
from matplotlib.colors import LightSource, to_rgb

from .style import plt
from .visibility import densify, interpolate_surface, visibility


def frame(ax):
    """Draw the native thin three-dimensional frame with transparent panes and no grid."""
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(True)
        axis.pane.set_fill(False)
        axis.pane.set_alpha(1)
        axis.pane.set_edgecolor("#a1adb5")
        axis.pane.set_linewidth(0.3)
        axis.line.set_visible(True)
        axis.line.set_color("#a1adb5")
        axis.line.set_linewidth(0.3)
        axis._axinfo["grid"]["linewidth"] = 0


def surface(ax, result, window=None):
    """Plot the voltage surface and inputs, with a black soma trace for the active model."""
    p, times = result["parameters"], result["time_ms"]
    active = p["soma"] == "active"
    spatial = slice(None, None, 1 if active else 2)
    origin = p["seed_time_ms"] + p["interval_ms"]
    window = window or (-p["interval_ms"], float(times[-1] - origin))
    indices = np.flatnonzero((times >= origin + window[0]) & (times <= origin + window[1]))[::4]
    x, y = result["shaft_positions_um"][spatial], times[indices] - origin
    voltage = result["shaft_mv"][spatial, :][:, indices]
    length = float(result["shaft_positions_um"][-1])

    # Shade the shaft voltage surface.
    xx, yy = np.meshgrid(x, y, indexing="ij")
    gx, gy = np.gradient(
        voltage * 0.74 / (160.0 if active else 110.0),
        x * 1.45 / length,
        y * 1.15 / (window[1] - window[0]),
    )
    normals = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1)[..., None]
    diffuse = np.clip(normals @ LightSource(azdeg=315, altdeg=55).direction, 0, 1)
    dark, body, bright = (np.array(to_rgb(c)) for c in ("#e88910", "#ff9913", "#ffc11a"))
    colors = (
        dark
        + (body - dark) * np.minimum(diffuse / 0.82, 1)[..., None]
        + (bright - body) * np.maximum((diffuse - 0.82) / 0.18, 0)[..., None]
    )
    ax.plot_surface(
        xx,
        yy,
        voltage,
        facecolors=colors,
        shade=False,
        linewidth=0,
        rcount=len(x),
        ccount=len(y),
        antialiased=False,
        rasterized=True,
        zorder=1,
    )

    # Set the orthographic camera and the direction used to test trace visibility.
    top = 65.0 if active else 15.0
    bounds = [(0.0, length), window, (-95.0, top)]
    ax.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2], zticks=(-90, 0))
    ax.set_proj_type("ortho")
    ax.view_init(elev=29, azim=-62)
    ax.set_box_aspect((1.45, 1.15, 0.74), zoom=0.98)
    ax.zaxis._axinfo["juggled"] = (1, 2, 0)
    direction = np.cross(ax.get_proj()[0, :3], ax.get_proj()[1, :3])
    direction *= 1 if direction[2] > 0 else -1
    direction /= np.max(abs(direction))

    def line(points, color, width, head=False, alpha=1.0):
        """Plot the visible portions of a voltage trace on or above the shaft surface."""
        points = densify(points)
        shown = visibility(points, x, y, voltage, direction, on_surface=not head)
        if head:
            above = np.flatnonzero(
                points[:, 2] > interpolate_surface(x, y, voltage, points[:, 0], points[:, 1]) + 0.1
            )
            shown[: above[0] if len(above) else len(points)] = True
        points[~shown] = np.nan
        ax.plot(*points.T, color=color, lw=width, alpha=alpha, zorder=8)

    # Add shaft and spine-head traces at the input sites and event times.
    event_times = list(result["input_times_ms"])
    if p["seeded"]:
        event_times.append(p["seed_time_ms"])
    for at in event_times:
        if origin + window[0] <= at <= origin + window[1]:
            j = np.argmin(abs(times - at))
            line(
                np.column_stack([x, np.full_like(x, at - origin), result["shaft_mv"][spatial, j]]),
                "#242424", 0.12, alpha=0.7
            )
    for i, position in enumerate(result["spine_positions_um"]):
        seed = i < p["seed_count"]
        if seed and not p["seeded"]:
            continue
        line(
            np.column_stack([np.full_like(y, position), y, result["site_shaft_mv"][i, indices]]),
            "#242424",
            0.045 if seed else 0.11,
            alpha=0.45 if seed else 0.8,
        )
        line(
            np.column_stack([np.full_like(y, position), y, result["head_mv"][i, indices]]),
            "#287fa8",
            0.17 if seed else 0.62,
            head=True,
            alpha=0.45 if seed else 1.0,
        )

    # Mark the seed and sequence inputs on the spine-head traces.
    events = [(i, p["seed_time_ms"], "#888888") for i in range(p["seed_count"]) if p["seeded"]]
    reversed_order = np.array_equal(result["input_sites"], np.arange(p["sequence_count"], 0, -1))
    events += [
        (p["seed_count"] + site - 1, at, "black" if reversed_order or site > rank + 1 else "white")
        for rank, (site, at) in enumerate(zip(result["input_sites"], result["input_times_ms"]))
    ]
    for i, at, color in events:
        if origin + window[0] <= at <= origin + window[1]:
            j = np.argmin(abs(times - at))
            ax.scatter(
                result["spine_positions_um"][i],
                at - origin,
                result["head_mv"][i, j],
                s=24,
                facecolors=color,
                edgecolors="#555555",
                linewidths=0.20,
                depthshade=False,
                zorder=12,
            )
    if active and "soma_mv" in result:
        keep = (times >= origin + window[0]) & (times <= origin + window[1])
        ax.plot(
            np.full(keep.sum(), length), times[keep] - origin,
            result["soma_mv"][keep], color="black", lw=1.0, zorder=16,
        )
    frame(ax)
    ax.set_xlabel("Position from tip (µm)", labelpad=-3)
    ax.set_ylabel("Time from input 1 (ms)", labelpad=-3)
    ax.set_zlabel("Voltage (mV)", labelpad=-3)
    ax.tick_params(labelsize=7, pad=-1)
    ax.yaxis.set_ticks_position("upper")


def _pulse_marker(ax, result):
    """Mark the imposed somatic spike and current onset in magenta."""
    from matplotlib.patches import FancyArrowPatch, ArrowStyle
    from mpl_toolkits.mplot3d import proj3d

    time = result["time_ms"]
    p = result["parameters"]
    origin = p["seed_time_ms"] + p["interval_ms"]
    spike = result["measurements"]["imposed_spike_ms"]
    selected = (time >= spike - 1) & (time <= spike + 3)
    length = result["shaft_positions_um"][-1]
    ax.plot(
        np.full(selected.sum(), length),
        time[selected] - origin,
        result["soma_mv"][selected],
        color="#b43b84",
        lw=1.5,
        zorder=17,
    )
    resting = np.median(result["soma_mv"][time < p["seed_time_ms"]])
    ax.figure.canvas.draw()
    tip = ax.transData.transform(
        proj3d.proj_transform(length, result["pulse_ms"] - origin, resting, ax.get_proj())[:2]
    )
    tail = tip - np.array([0.0, 20 * ax.figure.dpi / 72])
    ends = ax.figure.transFigure.inverted().transform(np.array([tail, tip]))

    def arrowhead(path, mutation_size, linewidth, aspect_ratio=1.0):
        """Draw the filled arrowhead at current onset."""
        paths, filled = ArrowStyle("-|>")(path, mutation_size, linewidth, aspect_ratio)
        return paths[-1], filled[-1]

    ax.figure.add_artist(
        FancyArrowPatch(
            *ends,
            transform=ax.figure.transFigure,
            arrowstyle=arrowhead,
            mutation_scale=13,
            color="#b43b84",
            lw=1.8,
            shrinkA=0,
            shrinkB=0,
            clip_on=False,
            zorder=18
        )
    )


def plot_examples(results, destination, window=None):
    """Save individual 600-dpi voltage panels with transparent margins."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    for i, result in enumerate(results.values()):
        figure = plt.figure(figsize=(3.5, 2.9))
        ax = figure.add_axes(
            [0.09, 0.075, 0.88, 0.88],
            projection="3d",
            computed_zorder=False,
        )
        surface(ax, result, window)
        for artist in (*ax.collections, *ax.lines):
            artist.set_clip_on(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xticks((0, 50, 100), labels=("", "", ""))
        ticks = (200, 400) if result["parameters"]["soma"] == "active" else (0, 100, 200)
        ax.set_yticks(ticks, labels=[""] * len(ticks))
        ticks = (-90, 0, 60) if result["parameters"]["soma"] == "active" else (-90, 0)
        ax.set_zticks(ticks, labels=[""] * len(ticks))
        ax.tick_params(labelsize=20, pad=-4, length=2, width=0.5)
        ax.yaxis.set_label_position("upper")
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_verticalalignment("top")
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_verticalalignment("bottom")
        for tick in ax.zaxis.get_major_ticks():
            tick.label1.set_horizontalalignment("right")
        if result.get("pulse_ms") is not None:
            _pulse_marker(ax, result)
        path = destination.with_name(destination.name + ("i" if i == 0 else "ii")).with_suffix(".png")
        figure.savefig(path, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.04)
        plt.close(figure)
        with Image.open(path) as image:
            image = image.convert("RGBA")
            a, b, c, d = image.getchannel("A").getbbox()
            image.crop((max(0, a - 12), max(0, b - 12), min(image.width, c + 12), min(image.height, d + 12))).save(
                path, dpi=(600, 600)
            )
        print(path)
