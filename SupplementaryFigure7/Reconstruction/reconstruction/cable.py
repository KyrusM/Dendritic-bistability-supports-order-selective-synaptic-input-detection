"""Reduce the measured membrane geometry to the Figure S7 cable."""

import math

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .surface import arc, point, project, section
from .soma_surface import tube, exit_point
from .projection import projection_data


SEEDS = [121, 118, 114, 108]
SEQUENCE = [101, 87, 72, 63, 45, 31, 7]


def soma_profile(vertices, faces, soma_row, guide):
    """Measure the soma profile along the original basal-branch direction."""
    soma = soma_row[2:5]
    direction = guide[0] - soma
    first = np.linalg.norm(direction)
    direction /= first
    keep = np.linalg.norm(vertices - soma, axis=1) < 20
    selected = faces[np.all(keep[faces], axis=1)]
    used, inverse = np.unique(selected, return_inverse=True)
    proximal = trimesh.Trimesh(vertices[used], inverse.reshape(-1, 3), process=False)
    proximal.merge_vertices(digits_vertex=6)
    cuts = []
    for at in np.arange(-10, first + 0.001, 0.25):
        cut = section(proximal, soma + at * direction, direction, offset_limit=5, radius_limit=12)
        if cut:
            cuts.append(dict(axis_um=float(at), **cut))
    boundary = next(
        i for i, cut in enumerate(cuts) if cut["axis_um"] > 0 and cut["radius_um"] < 1.2
    )
    return dict(
        points=[cut["xyz_um"] for cut in cuts[: boundary + 1]],
        radii=[cut["radius_um"] for cut in cuts[: boundary + 1]],
    )


def assemble_cable(vertices, faces, swc, extracted, cuts, spines, soma):
    """Complete the proximal cable and attach spines while preserving neck resistance."""
    rows = extracted["rows"]
    soma_row = swc[swc[:, 1] == 1][0]
    path = np.vstack([soma_row, rows])
    origin = soma_row[2:5]
    guide = extracted["shaft"]
    guide_arc = arc(guide)
    v, f = tube(soma["points"], soma["radii"], sides=32)
    surface = exit_point(origin, path[1, 2:5], v, f)
    prefix = np.vstack([surface, path[1, 2:5], guide[0]])
    prefix = prefix[np.r_[True, np.linalg.norm(np.diff(prefix, axis=0), axis=1) > 1e-7]]
    prefix_arc = arc(prefix)
    at = np.unique(np.r_[np.arange(0, prefix_arc[-1], 0.25), prefix_arc[-1]])
    points = np.array([point(x, prefix_arc, prefix) for x in at])[:-1].tolist()
    near = cKDTree(path[:2, 2:5]).query(vertices)[0] < 14
    selected = faces[np.all(near[faces], axis=1)]
    used, inverse = np.unique(selected, return_inverse=True)
    proximal = trimesh.Trimesh(vertices[used], inverse.reshape(-1, 3), process=False)
    proximal.merge_vertices(digits_vertex=6)
    radii, proximal_cuts = [], []
    for x, center in zip(at[:-1], points):
        normal = point(min(x + 0.35, prefix_arc[-1]), prefix_arc, prefix) - point(
            max(x - 0.35, 0), prefix_arc, prefix
        )
        normal /= np.linalg.norm(normal)
        cut = section(proximal, np.asarray(center), normal)
        radius = cut["radius_um"] if cut else float(path[1, 5])
        radii.append(radius)
        proximal_cuts.append(dict(position_um=float(x), measured=bool(cut), radius_um=radius))
    measured = [cut for cut in cuts if "radius_um" in cut]
    sample_at = [cut["position_um"] for cut in measured]
    sample_r = [cut["radius_um"] for cut in measured]
    at = np.unique(np.r_[0, np.arange(0.25, guide_arc[-1], 0.25), guide_arc[-1]])
    points.extend([point(x, guide_arc, guide).tolist() for x in at])
    radii.extend(np.interp(at, sample_at, sample_r).tolist())
    p, r = np.asarray(points), np.asarray(radii)
    distance = arc(p)

    for spine in spines:
        neck = np.asarray(spine["neck_model_xyz_um"])
        attachment = project(neck[0], p, distance)
        anchor = point(attachment, distance, p)
        length = arc(neck)[-1]
        resistance = spine.get("neck_resistance_MOhm") or 4 * length / (
            np.pi * spine["neck_model_radius_um"] ** 2
        )
        neck = np.vstack([anchor, neck])
        spine.update(
            shaft_attachment_arc_um=attachment,
            neck_model_xyz_um=neck.tolist(),
            neck_model_radius_um=float(np.sqrt(4 * arc(neck)[-1] / (np.pi * resistance))),
            preserved_neck_target_MOhm=float(resistance),
        )

    first = min(
        (cut for cut in proximal_cuts if cut["measured"]), key=lambda cut: cut["position_um"]
    )
    r[distance < first["position_um"] - 1e-9] = first["radius_um"]
    selected_ids = set(rows[:, 0].astype(int))
    junctions = sorted(
        set(
            int(row[6])
            for row in swc
            if int(row[0]) not in selected_ids
            and int(row[1]) in (3, 4)
            and int(row[6]) in selected_ids
        )
    )
    lookup = {int(row[0]): row for row in swc}
    boundaries = [project(lookup[i][2:5], p, distance) for i in junctions]
    return dict(
        soma=soma, points=p, radii=r, distance=distance, spines=spines, junctions=boundaries
    )


def section_geometry(points, radii, nseg, origin):
    """Store section coordinates and diameters at NEURON's pt3d precision."""
    points = np.asarray(points)
    radius = np.broadcast_to(radii, len(points))
    return dict(
        points=np.asarray(points - origin, dtype=np.float32).astype(float).tolist(),
        radii=(np.asarray(2 * radius, dtype=np.float32).astype(float) / 2).tolist(),
        nseg=nseg,
    )


def export_geometry(cable, projection):
    """Export the S7 cable with 2-µm necks and their measured resistance targets."""
    soma, p, r, distance = (cable[key] for key in ("soma", "points", "radii", "distance"))
    origin = np.asarray(soma["points"]).mean(axis=0)
    spines = {spine["display_id"]: spine for spine in cable["spines"]}
    attachments = {identity: spine["shaft_attachment_arc_um"] for identity, spine in spines.items()}
    driven = [attachments[i] for i in [SEEDS[-1]] + SEQUENCE]
    dx = float(min(abs(np.diff(driven))) / 12)
    boundaries = np.unique(np.r_[0, list(attachments.values()), cable["junctions"], distance[-1]])
    shaft = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        at = np.r_[lo, distance[(distance > lo) & (distance < hi)], hi]
        row = section_geometry(
            [point(x, distance, p) for x in at],
            np.interp(at, distance, r),
            max(1, math.ceil((hi - lo) / dx)),
            origin,
        )
        row["arc_start_um"] = float(lo)
        shaft.append(row)
    output_spines = []
    for identity, spine in spines.items():
        neck, head = np.asarray(spine["neck_model_xyz_um"]), np.asarray(spine["head_xyz_um"])
        direction = neck[-1] - neck[0]
        direction /= np.linalg.norm(direction)
        end = neck[0] + 2.0 * direction
        head = head + end - neck[-1]
        radius = float(np.sqrt(8 / (np.pi * spine["preserved_neck_target_MOhm"])))
        output_spines.append(
            dict(
                id=identity,
                attachment_arc_um=attachments[identity],
                neck=section_geometry([neck[0], end], radius, 1, origin),
                head=section_geometry(head, spine["head_radius_um"], 1, origin),
            )
        )
    result = dict(
        source=dict(
            dataset="MICrONS",
            cell_id="864691135117980637",
            tip_swc_id=7140,
            doi="10.1038/s41586-025-08790-w",
        ),
        soma=section_geometry(soma["points"], soma["radii"], 1, origin),
        shaft=shaft,
        length_um=float(distance[-1]),
        spines=output_spines,
        projection=projection,
    )
    result["projection"] = projection_data(result, projection)
    return result
