"""Neck resistance and head-area reductions from membrane cross-sections."""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .surface import arc, point, project, crop_mesh, section, normal_at, compact_cut


def ordered_candidates(extracted, soma_row):
    """Number spine paths by their distance along the released SWC."""
    guide = np.vstack([soma_row[2:5], extracted["rows"][:, 2:5]])
    guide_arc = arc(guide)
    vertices = extracted["vertices"]
    candidates = []
    for candidate in extracted["spines"]:
        path = gaussian_filter1d(
            vertices[candidate["path_nodes"]], sigma=0.8, axis=0, mode="nearest"
        )
        candidates.append(
            dict(
                distance=project(vertices[candidate["base_node"]], guide, guide_arc),
                points=path,
                head_center=np.asarray(
                    candidate.get("head_center_um", vertices[candidate["tip_node"]].tolist())
                ),
                neck_end=candidate.get("neck_length_um"),
            )
        )
    candidates.sort(key=lambda row: row["distance"])
    return candidates


def reduce_spine(mesh, candidate, identity, shaft, shaft_arc):
    """Measure neck resistance in MΩ at 4 MΩ µm and construct the spine geometry."""
    path, head_center, neck_end = (candidate[key] for key in ("points", "head_center", "neck_end"))
    distance = arc(path)
    local = crop_mesh(mesh, path)
    samples = []
    for at in np.arange(0.025, distance[-1] - 0.025, 0.025):
        cut = section(
            local, point(at, distance, path), normal_at(at, distance, path), radius_limit=0.8
        )
        if cut:
            samples.append(dict(arc_um=float(at), **cut))
    base, valid_base = None, False
    if neck_end:
        for cut in samples:
            at = cut["arc_um"]
            if at >= neck_end - 0.05:
                break
            if cut["radius_um"] > 0.35:
                continue
            report, _ = compact_cut(local, at, distance, path, head_center)
            if report.get("valid"):
                base, valid_base = cut, True
                break
    if base is None and samples:
        base = samples[0]
    if base is None:
        raise ValueError(f"Spine {identity} has no resolved membrane cross-section.")
    usable = [cut for cut in samples if cut["arc_um"] >= base["arc_um"]]
    if neck_end and base["arc_um"] < neck_end:
        cut = section(
            local,
            point(neck_end, distance, path),
            normal_at(neck_end, distance, path),
            radius_limit=0.8,
        )
        if cut:
            usable.append(dict(arc_um=float(neck_end), **cut))
    usable = sorted(
        {round(cut["arc_um"], 9): cut for cut in usable}.values(), key=lambda cut: cut["arc_um"]
    )
    neck = [cut for cut in usable if neck_end and cut["arc_um"] <= neck_end + 1e-9]
    resistance, head_area = None, None
    if len(neck) >= 3 and valid_base and abs(neck[-1]["arc_um"] - neck_end) < 1e-8:
        neck_points = np.asarray([cut["xyz_um"] for cut in neck])
        neck_radii = np.array([cut["radius_um"] for cut in neck])
        ds = np.linalg.norm(np.diff(neck_points, axis=0), axis=1)
        resistance = float(4 / np.pi * np.sum(ds / (neck_radii[:-1] * neck_radii[1:])))
        neck_radius = float(np.sqrt(4 * ds.sum() / (np.pi * resistance)))
        report, _ = compact_cut(local, neck_end, distance, path, head_center)
        if report.get("valid"):
            head_area = report["area_um2"]
    if resistance is not None:
        vector = head_center - neck_points[-1]
        head_length = max(2 * np.linalg.norm(vector), 0.08)
        axis = vector / np.linalg.norm(vector)
        head_points = np.array(
            [head_center - axis * head_length / 2, head_center + axis * head_length / 2]
        )
        area = (
            head_area
            if head_area is not None
            else 4 * np.pi * max(cut["radius_um"] for cut in usable) ** 2
        )
        head_radius = float(area / (2 * np.pi * head_length))
    else:
        start, end = np.asarray(usable[0]["xyz_um"]), np.asarray(usable[-1]["xyz_um"])
        midpoint = (start + end) / 2
        neck_radius = float(np.median([cut["radius_um"] for cut in usable]))
        neck_points, head_points = np.array([start, midpoint]), np.array([midpoint, end])
        head_radius = neck_radius
    return dict(
        display_id=identity,
        neck_resistance_MOhm=resistance,
        neck_model_xyz_um=neck_points.tolist(),
        neck_model_radius_um=neck_radius,
        head_xyz_um=head_points.tolist(),
        head_radius_um=head_radius,
        shaft_attachment_arc_um=project(neck_points[0], shaft, shaft_arc),
    )


def refine_spines(extracted, soma_row):
    """Reduce all spine paths in their anatomical order."""
    shaft = extracted["shaft"]
    shaft_arc = arc(shaft)
    candidates = ordered_candidates(extracted, soma_row)
    spines = []
    for identity, candidate in enumerate(candidates, 1):
        spines.append(reduce_spine(extracted["mesh"], candidate, identity, shaft, shaft_arc))
        if identity % 10 == 0:
            print(f"Measured spines: {identity}/{len(candidates)}", flush=True)
    return spines
