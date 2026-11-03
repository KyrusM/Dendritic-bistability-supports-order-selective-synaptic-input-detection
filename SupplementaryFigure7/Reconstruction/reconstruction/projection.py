"""The paper's fixed anatomical projection and input-marker positions."""

import numpy as np


DOT_OUTER_RADIUS = 1.3
DOT_GAP = 0.65


def place_dots(annotations, parts, radii):
    """Place input markers clear of the projected membrane and each other."""
    starts = np.vstack([p[:-1] for p in parts])
    ends = np.vstack([p[1:] for p in parts])
    radius = np.concatenate([np.maximum(r[:-1], r[1:]) for r in radii])
    vector = ends - starts
    length2 = np.maximum(np.sum(vector * vector, axis=1), 1e-20)

    def membrane_clearance(points):
        """Return each marker's minimum distance from the projected membrane."""
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


def projection_data(geometry, layout):
    """Project the reconstructed input sites and place their figure markers."""
    center, basis, origin = (
        np.array(layout[key]) for key in ("center", "basis", "projection_origin")
    )

    def project(points):
        """Apply the fixed paper camera to three-dimensional points."""
        return (np.asarray(points) - center) @ basis - origin

    seeds = [121, 118, 114, 108]
    sequence = [101, 87, 72, 63, 45, 31, 7]
    sections = (
        [geometry["soma"]]
        + geometry["shaft"]
        + [spine[key] for spine in geometry["spines"] for key in ("neck", "head")]
    )
    parts = [project(section["points"]) for section in sections]
    radii = [np.asarray(section["radii"]) for section in sections]
    annotations = []
    for spine in geometry["spines"]:
        identity = spine["id"]
        if identity not in seeds + sequence:
            continue
        tip = project(spine["head"]["points"])[-1]
        start = project(spine["neck"]["points"])[0]
        direction = layout["marker_directions"].get(str(identity))
        if direction is None:
            direction = ((tip - start) / np.linalg.norm(tip - start)).tolist()
        annotations.append(
            dict(
                id=identity,
                role="seed" if identity in seeds else "sequence",
                head_xy_um=tip.tolist(),
                outward_xy=direction,
            )
        )
    place_dots(annotations, parts, radii)
    return dict(
        center=layout["center"],
        basis=layout["basis"],
        projection_origin=layout["projection_origin"],
        annotations=annotations,
    )
