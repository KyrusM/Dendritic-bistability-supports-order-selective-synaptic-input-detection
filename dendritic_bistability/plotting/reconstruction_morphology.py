"""Anatomical inset aligned to the S7 voltage surface."""

import copy
import numpy as np
from matplotlib.patches import Circle, Polygon
from mpl_toolkits.mplot3d import proj3d

from .style import plt

DOT_OUTER_RADIUS, DOT_GAP = 3.3, 0.65


def tube(ax, points, radii, color, zorder, rounded_union=False):
    """Draw a tapered projected tube at its physical diameter, with round ends."""
    points = np.asarray(points)
    radii = np.broadcast_to(radii, len(points))

    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]
    points, radii = points[keep], radii[keep]
    if len(points) == 1:
        ax.add_patch(Circle(points[0], radii[0], fc=color, ec="none", zorder=zorder))
        return
    if rounded_union:

        for j in range(len(points) - 1):
            direction = points[j + 1] - points[j]
            direction /= np.linalg.norm(direction)
            normal = np.array([-direction[1], direction[0]])
            corners = [
                points[j] + normal * radii[j],
                points[j + 1] + normal * radii[j + 1],
                points[j + 1] - normal * radii[j + 1],
                points[j] - normal * radii[j],
            ]
            ax.add_patch(Polygon(corners, fc=color, ec="none", zorder=zorder))
        for point, radius in zip(points, radii):
            ax.add_patch(Circle(point, radius, fc=color, ec="none", zorder=zorder))
        return
    tangent = np.gradient(points, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    edge = normal * radii[:, None]
    polygon = np.vstack([points + edge, (points - edge)[::-1]])
    ax.add_patch(Polygon(polygon, closed=True, fc=color, ec="none", zorder=zorder))
    for index in [0, -1]:
        ax.add_patch(Circle(points[index], radii[index], fc=color, ec="none", zorder=zorder))


def place_dots(annotations, parts, radii):
    """Place each marker near its own head, clearing all anatomy and markers."""

    starts = np.vstack([p[:-1] for p in parts])
    ends = np.vstack([p[1:] for p in parts])
    radius = np.concatenate([np.maximum(r[:-1], r[1:]) for r in radii])
    vector = ends - starts
    length2 = np.maximum(np.sum(vector * vector, axis=1), 1e-20)

    def membrane_clearance(points):
        """Return each point's clearance from the projected membrane."""
        relative = points[:, None, :] - starts[None, :, :]
        fraction = np.clip(np.einsum("pni,ni->pn", relative, vector) / length2, 0, 1)
        distance = np.linalg.norm(relative - fraction[:, :, None] * vector, axis=2)
        return np.min(distance - radius, axis=1) - DOT_OUTER_RADIUS

    placed = []
    for row in annotations:
        tip = np.array(row["head_xy_um"])
        outward = np.array(row["outward_xy"])
        preferred = np.arctan2(outward[1], outward[0])
        offsets = np.deg2rad(np.arange(-180, 180, 5))

        angles = preferred + offsets
        directions = np.column_stack([np.cos(angles), np.sin(angles)])
        for displacement in np.arange(1.5, 12.01, 0.25):
            candidates = tip + displacement * directions
            clear = membrane_clearance(candidates) >= DOT_GAP
            if placed:
                distance = np.linalg.norm(candidates[:, None, :] - np.array(placed), axis=2)
                clear &= np.min(distance, axis=1) >= 2 * DOT_OUTER_RADIUS + DOT_GAP
            if clear.any():
                score = np.abs(offsets) + 0.05 * np.abs(directions[:, 0])
                selected = np.flatnonzero(clear)[np.argmin(score[clear])]
                dot = candidates[selected]
                row["dot_xy_um"] = dot.tolist()
                row["head_offset_um"] = float(displacement)
                placed.append(dot)
                break
        else:
            raise AssertionError(f"No clear marker position for spine {row['id']}")


def camera_angle():
    """Match the schematic tilt to the surface's projected spatial edge."""
    fig = plt.figure(figsize=(3.5, 2.9))
    ax = fig.add_axes([0.09, 0.075, 0.88, 0.88], projection="3d")
    ax.set(xlim=(0, 198.1063362644), ylim=(-20, 600), zlim=(-95, 65))
    ax.set_proj_type("ortho")
    ax.view_init(elev=29, azim=-62)
    ax.set_box_aspect((1.45, 1.15, 0.74))
    fig.canvas.draw()
    points = []
    for x in [0, 198.1063362644]:
        xx, yy, _ = proj3d.proj_transform(x, -20, -95, ax.get_proj())
        points.append(ax.transData.transform((xx, yy)))
    delta = np.array(points[1]) - points[0]
    angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
    plt.close(fig)
    return angle


def drawing(payload, angle):
    """Project the cable geometry and place seed and sequence markers."""
    center = np.array(payload["center"])
    basis = np.array(payload["basis"])
    origin = np.array(payload["projection_origin"])
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    unrotated = [(np.asarray(q["points"]) - center) @ basis - origin for q in payload["sections"]]
    radii = [np.asarray(q["radii"]) for q in payload["sections"]]
    annotations = copy.deepcopy(payload["annotations"])
    place_dots(annotations, unrotated, radii)
    parts = [q @ rotation for q in unrotated]
    dots = [np.array(q["dot_xy_um"]) @ rotation for q in annotations]
    bounds = []
    for xy, r in zip(parts, radii):
        bounds.extend([xy - r[:, None], xy + r[:, None]])
    for xy in dots:
        bounds.extend([xy[None, :] - 3.3, xy[None, :] + 3.3])
    bounds = np.vstack(bounds)
    lo, hi = bounds.min(axis=0) - 2, bounds.max(axis=0) + 2

    soma = parts[0]
    rr = radii[0]
    j = int(np.argmin(soma[:, 0] - rr))
    soma_start = np.array([soma[j, 0] - rr[j], soma[j, 1]])
    shaft_indices = [i for i, q in enumerate(payload["sections"]) if "_shaft_" in q["name"]]
    tip = parts[shaft_indices[-1]][-1].copy()
    return dict(
        parts=parts,
        radii=radii,
        dots=dots,
        annotations=annotations,
        soma_start=soma_start,
        tip=tip,
        lo=lo,
        hi=hi,
        soma_section_name=payload["soma"]["name"],
        names=[q["name"] for q in payload["sections"]],
    )


def draw(ax, d):
    """Draw tapered membrane outlines and input markers."""
    for xy, r, name in zip(d["parts"], d["radii"], d["names"]):
        is_soma = name == d["soma_section_name"]
        tube(
            ax,
            xy,
            r,
            "#000000" if is_soma else "#788184",
            4 if is_soma else 2,
            rounded_union=is_soma,
        )
    for xy, entry in zip(d["dots"], d["annotations"]):
        ax.add_patch(
            Circle(
                xy,
                3.0,
                fc="#888888" if entry["role"] == "seed" else "#2585a8",
                ec="white",
                lw=0.16,
                zorder=6,
            )
        )
    ax.set(xlim=(d["lo"][0], d["hi"][0]), ylim=(d["lo"][1], d["hi"][1]), aspect="equal")
    ax.axis("off")


def place(fig, image_ax, payload, angle, surface_report):
    """Align the anatomical tip and soma to the surface frame and soma trace."""
    d = drawing(payload, angle)
    fig.canvas.draw()
    image = image_ax.images[0]
    rows, cols = image.get_array().shape[:2]

    def image_point(fraction):
        """Map an image coordinate to its printed position in inches."""
        pixel = np.array([fraction[0] * cols - 0.5, fraction[1] * rows - 0.5])
        return image_ax.transData.transform(pixel) / fig.dpi

    anchors = {name: image_point(p) for name, p in surface_report["image_anchors"].items()}
    target = anchors["black_trace_start"][0]
    a, b = anchors["frame_tip"], anchors["frame_soma"]
    slope = (b[1] - a[1]) / (b[0] - a[0])

    scale = (target - a[0]) / (d["soma_start"][0] - d["tip"][0])
    d["display_scale"] = scale
    span = d["hi"] - d["lo"]
    left = a[0] - (d["tip"][0] - d["lo"][0]) * scale
    centers = np.vstack(d["parts"] + list(np.array(d["dots"])[:, None, :]))
    rr = np.concatenate(d["radii"] + [np.full(len(d["dots"]), 3.3)])
    x = left + (centers[:, 0] - d["lo"][0]) * scale
    axis_y = a[1] + slope * (x - a[0])
    bottom = float(
        np.min(
            axis_y
            - 0.10
            - (centers[:, 1] - d["lo"][1]) * scale
            - rr * scale * np.sqrt(1 + slope * slope)
        )
    )
    width, height = span * scale
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes([left / fw, bottom / fh, width / fw, height / fh])
    draw(ax, d)
    fig.canvas.draw()
    return dict(axes_box_inches=[float(left), float(bottom), float(width), float(height)]), d
