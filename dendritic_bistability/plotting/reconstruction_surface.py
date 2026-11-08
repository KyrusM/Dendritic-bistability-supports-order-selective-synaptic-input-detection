"""Voltage surfaces and label placement for Figure S7."""

import numpy as np
from PIL import Image
from matplotlib.colors import LightSource, to_rgb
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import proj3d

from .style import plt
from .voltage import frame
from .visibility import densify, interpolate_surface, visibility
from ..reconstruction import SEEDS, SEQUENCE

BLUE, BLACK = "#287fa8", "#242424"
SURFACE_BOX = (1.95, 1.49)
SURFACE_FEATURE_SCALE = 1 / 1.40


def surface(ax, row):
    """Draw the recorded shaft surface, spine traces and input markers."""
    data = row["_render_data"]
    length = row["native_branch_length_um"]
    origin = row["first_input_time_ms"]
    ymin, ymax = row["time_bounds_ms"]
    times = data["surface_t"]
    ti = np.flatnonzero((times >= origin + ymin) & (times <= origin + ymax))[::4]
    xi = np.argsort(length - data["surface_position_um"])
    x, y = length - data["surface_position_um"][xi], times[ti] - origin
    v = data["surface_v"][np.ix_(xi, ti)]
    xx, yy = np.meshgrid(x, y, indexing="ij")
    gx, gy = np.gradient(v * 0.74 / 160.0, x * 1.45 / length, y * 1.15 / (ymax - ymin))
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
        v,
        facecolors=colors,
        shade=False,
        linewidth=0,
        rcount=len(x),
        ccount=len(y),
        antialiased=False,
        rasterized=True,
        zorder=1,
    )
    bounds = [(0.0, length), (ymin, ymax), (-95.0, 65.0)]
    ax.set(
        xlim=bounds[0],
        ylim=bounds[1],
        zlim=bounds[2],
        xticks=(0, 50, 100),
        yticks=(0, 100, 200),
        zticks=(-90, -45, 0),
    )
    ax.set_proj_type("ortho")
    ax.view_init(elev=29, azim=-62)
    ax.set_box_aspect((1.45, 1.15, 0.74), zoom=0.98)
    ax.tick_params(labelsize=7, pad=-1)
    ax.set_xlabel("Position (µm)", labelpad=-5)
    ax.set_ylabel("Time (ms)", labelpad=-5)
    ax.set_zlabel("$u$ (mV)", labelpad=-4)
    ax.zaxis._axinfo["juggled"] = (1, 2, 0)

    matrix = ax.get_proj()
    direction = np.cross(matrix[0, :3], matrix[1, :3])
    if direction[2] < 0:
        direction *= -1
    direction /= np.max(abs(direction))

    def line(points, color, width, head=False, alpha=1):
        """Draw visible trace portions using the orthographic camera direction."""
        dense = densify(points)
        visible = visibility(dense, x, y, v, direction, on_surface=not head)
        if head:
            local = interpolate_surface(x, y, v, dense[:, 0], dense[:, 1])
            crossing = np.flatnonzero(dense[:, 2] > local + 0.1)
            visible[: crossing[0] if len(crossing) else len(dense)] = True
        dense[~visible] = np.nan
        ax.plot(*dense.T, color=color, lw=width, alpha=alpha, zorder=8)

    for at in np.unique(data["event_times"]):
        if origin + ymin <= at <= origin + ymax:
            j = np.argmin(abs(times - at))
            line(
                np.column_stack([x, np.full_like(x, at - origin), data["surface_v"][xi, j]]),
                BLACK,
                0.12,
                alpha=0.7,
            )
    physical_times = np.asarray(data.get("physical_t", data["t"]))
    physical_indices = np.flatnonzero(
        (physical_times >= origin + ymin) & (physical_times <= origin + ymax)
    )[::4]
    physical_y = physical_times[physical_indices] - origin
    physical_x = length - data["physical_position_um"]
    for i, pos in enumerate(physical_x):
        seed = i < row["seed_count"]
        line(
            np.column_stack(
                [
                    np.full_like(physical_y, pos),
                    physical_y,
                    data["physical_shaft_v"][i, physical_indices],
                ]
            ),
            BLACK,
            0.045 if seed else 0.11,
            alpha=0.45 if seed else 0.8,
        )
        line(
            np.column_stack(
                [
                    np.full_like(physical_y, pos),
                    physical_y,
                    data["physical_head_v"][i, physical_indices],
                ]
            ),
            BLUE,
            0.17 if seed else 0.62,
            head=True,
            alpha=0.45 if seed else 1.0,
        )
    for site, at in zip(data["event_sites"], data["event_times"]):
        if int(site) == 0:
            j = np.argmin(abs(physical_times - at))
            for seed_index in range(row["seed_count"]):
                ax.scatter(
                    physical_x[seed_index],
                    at - origin,
                    data["physical_head_v"][seed_index, j],
                    s=24,
                    color="#888888",
                    edgecolors="#555555",
                    linewidths=0.20,
                    depthshade=False,
                    zorder=12,
                )
            continue
        i = int(site) - 1 + row["seed_count"]
        j = np.argmin(abs(physical_times - at))
        event_v = data["physical_head_v"][i, j]
        ax.scatter(
            physical_x[i],
            at - origin,
            event_v,
            s=24,
            color=(
                "black"
                if row["kind"] == "reversed"
                or (row["kind"] == "adjacent_swap" and int(site) in row.get("black_sites", []))
                else "white"
            ),
            edgecolors="#555555",
            linewidths=0.20,
            depthshade=False,
            zorder=12,
        )
    frame(ax)
    return dict(
        surface_points=len(x),
        time_samples=len(y),
        length_um=length,
        voltage_min_mv=float(v.min()),
        voltage_max_mv=float(v.max()),
    )


def capture_render_anchors(fig, ax, example):
    """Record projected frame/trace points before the surface PNG is cropped."""
    fig.set_dpi(600)
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.04)
    length = example["native_branch_length_um"]
    front = ax.get_ylim()[0]
    bottom = ax.get_zlim()[0]
    data = example["_render_data"]
    origin = example["first_input_time_ms"]
    j = np.flatnonzero(data["t"] >= origin + front)[0]
    points = {
        "frame_tip": (0, front, bottom),
        "frame_soma": (length, front, bottom),
        "black_trace_start": (length, data["t"][j] - origin, data["soma_v"][j]),
    }
    result = {}
    for name, point in points.items():
        x, y, _ = proj3d.proj_transform(*point, ax.get_proj())
        xy = ax.transData.transform((x, y)) / fig.dpi
        result[name] = [(xy[0] - box.x0) * 600, (box.y1 - xy[1]) * 600]
    ticks = []
    renderer = fig.canvas.get_renderer()
    for axis_name, axis in [("position", ax.xaxis), ("time", ax.yaxis), ("voltage", ax.zaxis)]:
        for tick in axis.get_major_ticks():
            label = tick.label1
            if not label.get_visible() or not label.get_text():
                continue
            b = label.get_window_extent(renderer)
            ticks.append(
                dict(
                    axis=axis_name,
                    text=label.get_text(),
                    source_fontsize_pt=label.get_fontsize(),
                    bbox_pixels=[
                        b.x0 - box.x0 * 600,
                        box.y1 * 600 - b.y1,
                        b.x1 - box.x0 * 600,
                        box.y1 * 600 - b.y0,
                    ],
                )
            )
    return dict(points=result, ticks=ticks, tightbox_inches=list(box.bounds))


def crop_render_anchors(points, crop):
    """Map surface anchors into cropped image coordinates."""
    width, height = crop[2] - crop[0], crop[3] - crop[1]
    return {
        name: [(xy[0] - crop[0]) / width, (xy[1] - crop[1]) / height]
        for name, xy in points["points"].items()
    }


def crop_tick_labels(anchors, crop, inset_inches=SURFACE_BOX):
    """Map tick labels to the cropped image and its final printed size."""
    width, height = crop[2] - crop[0], crop[3] - crop[1]
    inset_scale = min(inset_inches[0] / width, inset_inches[1] / height)
    labels = []
    for tick in anchors["ticks"]:
        b = np.asarray(tick["bbox_pixels"]) - np.array([crop[0], crop[1], crop[0], crop[1]])
        labels.append(
            dict(
                axis=tick["axis"],
                text=tick["text"],
                bbox_fraction=(b / np.array([width, height, width, height])).tolist(),
                final_fontsize_pt=tick["source_fontsize_pt"] * 600 * inset_scale,
            )
        )
    return dict(
        labels=labels,
        target_fontsize_pt=6,
        final_size_tolerance_pt=0.05,
    )


def surface_png(row, path, geometry, source_tick_size=10.6):
    """Render one surface with six-point ticks at its final printed size."""
    spines = {q["id"]: q for q in geometry["spines"]}
    with np.load(row["recording_path"]) as z:
        t, st = z["time_ms"], z["surface_time_ms"]
        ids = [list(z["ids"]).index(i) for i in SEEDS + SEQUENCE]
        physical = np.array([spines[i]["attachment_arc_um"] for i in SEEDS + SEQUENCE])
        positions = np.r_[0, z["surface_arc_um"], physical]
        voltage = np.vstack(
            [
                np.interp(st, t, z["soma_v_mv"]),
                z["surface_v_mv"],
                [np.interp(st, t, u) for u in z["shaft_v_mv"][ids]],
            ]
        )
        positions, indices = np.unique(positions, return_index=True)
        events = [(0, 1000.0)] + [
            (SEQUENCE.index(e["id"]) + 1, e["onset_ms"])
            for e in row["actual_synapses"]
            if e["id"] in SEQUENCE
        ]
        events.sort(key=lambda e: e[1])
        data = dict(
            t=t,
            surface_t=st,
            surface_position_um=positions,
            surface_v=voltage[indices],
            physical_position_um=physical,
            physical_head_v=z["head_v_mv"][ids],
            physical_shaft_v=z["shaft_v_mv"][ids],
            soma_v=z["soma_v_mv"],
            event_sites=np.array([e[0] for e in events]),
            event_times=np.array([e[1] for e in events]),
        )
    ex = dict(
        _render_data=data,
        kind="ordered" if row["name"] == "ordered" else "adjacent_swap",
        seed_count=4,
        first_input_time_ms=1020.0,
        native_branch_length_um=geometry["length_um"],
        time_bounds_ms=(-20.0, 240.0),
        black_sites=[] if row["name"] == "ordered" else [4],
    )
    fig = plt.figure(figsize=(3.5, 2.9))
    ax = fig.add_axes([0.09, 0.075, 0.88, 0.88], projection="3d", computed_zorder=False)
    report = surface(ax, ex)
    data = ex["_render_data"]
    sel = (data["t"] >= 1000) & (data["t"] <= 1260)
    ax.plot(
        np.full(sel.sum(), geometry["length_um"]),
        data["t"][sel] - 1020,
        data["soma_v"][sel],
        color="black",
        lw=1,
        zorder=16,
    )
    for artist in (*ax.lines, *ax.collections):
        artist.set_clip_on(False)
    ax.set(xticks=[0, 100, 200], yticks=[0, 100, 200], zticks=[-90, 0])
    ax.tick_params(labelsize=7, pad=-1)
    ax.yaxis.set_ticks_position("upper")
    ax.yaxis.set_label_position("upper")
    ax.set_xlabel("Position from tip (µm)", fontsize=8, labelpad=-3)
    ax.set_ylabel("Time from input 1 (ms)", fontsize=8, labelpad=-3)
    ax.set_zlabel("$u$ (mV)", fontsize=8, labelpad=-3)
    for line in ax.lines:
        line.set_linewidth(line.get_linewidth() * SURFACE_FEATURE_SCALE)
    for collection in ax.collections:
        collection.set_linewidths(collection.get_linewidths() * SURFACE_FEATURE_SCALE)
        if hasattr(collection, "get_sizes"):
            collection.set_sizes(collection.get_sizes() * SURFACE_FEATURE_SCALE**2)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_linewidth(axis.line.get_linewidth() * SURFACE_FEATURE_SCALE)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticks([0, 100, 200], labels=["0", "100", "200"])
    ax.set_yticks([0, 100, 200], labels=["0", "100", "200"])
    ax.set_zticks([-90, 0, 60], labels=["−90", "0", "60"])
    ax.tick_params(labelsize=source_tick_size, pad=-4, length=2, width=0.5)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_verticalalignment("top")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_verticalalignment("bottom")
    for tick in ax.zaxis.get_major_ticks():
        tick.label1.set_horizontalalignment("right")
    ax.yaxis.set_ticks_position("upper")
    ax.yaxis.set_label_position("upper")
    anchors = capture_render_anchors(fig, ax, ex)
    if row["name"] != "ordered":
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    fig.savefig(
        path,
        dpi=600,
        transparent=True,
        bbox_inches=Bbox.from_bounds(*anchors["tightbox_inches"]),
        pad_inches=0,
    )
    plt.close(fig)
    with Image.open(path) as image:
        image = image.convert("RGBA")
        a, b, c, d = image.getchannel("A").getbbox()
        tick_boxes = np.array([q["bbox_pixels"] for q in anchors["ticks"]])
        crop = (
            int(np.floor(min(a, tick_boxes[:, 0].min()))) - 12,
            int(np.floor(min(b, tick_boxes[:, 1].min()))) - 12,
            int(np.ceil(max(c, tick_boxes[:, 2].max()))) + 12,
            int(np.ceil(max(d, tick_boxes[:, 3].max()))) + 12,
        )
        image.crop(crop).save(path, dpi=(600, 600))
        report["image_anchors"] = crop_render_anchors(anchors, crop)
        report["tick_labels"] = crop_tick_labels(anchors, crop, SURFACE_BOX)
    final_size = report["tick_labels"]["labels"][0]["final_fontsize_pt"]
    if abs(final_size - 6) > 0.04:
        return surface_png(row, path, geometry, source_tick_size * 6 / final_size)
    return report
