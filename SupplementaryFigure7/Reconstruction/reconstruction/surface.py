"""Surface cross-sections and equivalent spine-head measurements.

MICrONS Consortium (2025), doi:10.1038/s41586-025-08790-w.
"""

import collections
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def arc(points):
    """Return cumulative distance along a polyline."""
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def point(x, s, xyz):
    """Interpolate a position along a polyline."""
    return np.array([np.interp(x, s, xyz[:, d]) for d in range(3)])


def project(p, xyz, s):
    """Project a point onto a polyline and return its arc distance."""
    delta = np.diff(xyz, axis=0)
    t = np.clip(
        np.einsum("ij,ij->i", p - xyz[:-1], delta)
        / np.maximum(np.einsum("ij,ij->i", delta, delta), 1e-20),
        0,
        1,
    )
    near = xyz[:-1] + t[:, None] * delta
    i = np.argmin(np.sum((near - p) ** 2, axis=1))
    return float(s[i] + t[i] * (s[i + 1] - s[i]))


def crop_mesh(mesh, points, radius=4):
    """Crop membrane triangles around a spine centerline."""
    keep = np.linalg.norm(mesh.vertices - points.mean(axis=0), axis=1) < radius
    faces = mesh.faces[np.all(keep[mesh.faces], axis=1)]
    used, inverse = np.unique(faces, return_inverse=True)
    return trimesh.Trimesh(mesh.vertices[used], inverse.reshape(-1, 3), process=False)


def contours(lines):
    """Join plane-intersection segments into closed contours."""
    if len(lines) == 0:
        return []
    pts = np.round(lines.reshape(-1, 3), 4)
    unique, inv = np.unique(pts, axis=0, return_inverse=True)
    adj = collections.defaultdict(set)
    for a, b in inv.reshape(-1, 2):
        if a != b:
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))
    seen = set()
    polys = []
    for start in adj:
        if start in seen:
            continue
        component = set()
        stack = [start]
        while stack:
            a = stack.pop()
            if a in component:
                continue
            component.add(a)
            stack.extend(adj[a] - component)
        seen |= component
        if len(component) < 4 or any(len(adj[a]) != 2 for a in component):
            continue
        sequence = [start]
        prev = None
        cur = start
        for _ in range(len(component)):
            nxt = next(b for b in adj[cur] if b != prev)
            if nxt == start:
                break
            sequence.append(nxt)
            prev, cur = cur, nxt
        if len(sequence) == len(component):
            polys.append(unique[sequence])
    return polys


def coarse_section(mesh, point, normal):
    """Measure a closed contour containing the centerline."""
    lines = trimesh.intersections.mesh_plane(mesh, normal, point)
    loops = contours(lines)
    valid = []
    for poly in loops:
        rel = poly - point
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            area = abs(float(np.sum(np.cross(rel, np.roll(rel, -1, axis=0)) @ normal) / 2))
        distance = float(np.linalg.norm(poly.mean(axis=0) - point))
        axis = np.eye(3)[np.argmin(np.abs(normal))]
        u = np.cross(normal, axis)
        u /= np.linalg.norm(u)
        w = np.cross(normal, u)
        xx = np.sum(rel * u, axis=1)
        yy = np.sum(rel * w, axis=1)
        x2 = np.roll(xx, -1)
        y2 = np.roll(yy, -1)
        cross = (yy > 0) != (y2 > 0)
        ix = xx[cross] - yy[cross] * (x2[cross] - xx[cross]) / (y2[cross] - yy[cross])
        contains = bool(np.sum(ix > 0) % 2)
        if area > 0.0003 and distance < 0.5 and contains:
            valid.append((distance, area, poly))
    if not valid:
        return None
    distance, area, poly = min(valid, key=lambda x: x[0])
    return dict(area_um2=area, radius_um=float(np.sqrt(area / np.pi)), centroid_offset_um=distance)


def components(n, edges):
    """Label connected components of a surface graph."""
    if len(edges) == 0:
        return n, np.arange(n)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)


def measure(mesh, origin, normal, center):
    """Measure uncapped membrane area beyond the neck endpoint."""
    vertices, faces, _ = trimesh.intersections.slice_faces_plane(
        mesh.vertices, mesh.faces, plane_normal=normal, plane_origin=origin
    )
    cut = trimesh.Trimesh(vertices, faces, process=False)
    assert np.isfinite(cut.vertices).all()
    cut.merge_vertices(digits_vertex=8)
    ncomponents, labels = components(len(cut.faces), cut.face_adjacency)
    centroids = cut.vertices[cut.faces].mean(axis=1)
    distance = np.sum((centroids - center) ** 2, axis=1)
    keep = labels == labels[int(np.argmin(distance))]
    faces = cut.faces[keep]
    used, inverse = np.unique(faces, return_inverse=True)
    selected = trimesh.Trimesh(cut.vertices[used], inverse.reshape(-1, 3), process=False)
    counts = np.bincount(selected.edges_unique_inverse)
    boundary = selected.edges_unique[counts == 1]
    vertices, remap = np.unique(boundary, return_inverse=True)
    edges = remap.reshape(-1, 2)
    loops, _ = components(len(vertices), edges)
    degree = np.bincount(edges.ravel(), minlength=len(vertices))
    max_extent = float(np.max(selected.extents))
    plane_error = (
        float(np.max(np.abs(np.einsum("ij,j->i", selected.vertices[vertices] - origin, normal))))
        if len(vertices)
        else None
    )
    valid = bool(
        len(boundary)
        and loops == 1
        and np.all(degree == 2)
        and np.max(counts) == 2
        and max_extent < 3
        and 0.01 < selected.area < 20
        and plane_error < 1e-7
    )
    report = dict(area_um2=float(selected.area), valid=valid)
    return report, selected


def section(mesh, centre, normal, offset_limit=0.5, radius_limit=2):
    """Measure the centered area-equivalent radius of a membrane cross-section."""
    loops = contours(trimesh.intersections.mesh_plane(mesh, normal, centre))
    axis = np.eye(3)[np.argmin(abs(normal))]
    u = np.cross(normal, axis)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    matches = []
    for poly in loops:
        rel = poly - centre
        x, y = np.einsum("ij,j->i", rel, u), np.einsum("ij,j->i", rel, v)
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        signed = cross.sum() / 2
        if abs(signed) < 0.0003:
            continue
        cx = np.sum((x + np.roll(x, -1)) * cross) / (6 * signed)
        cy = np.sum((y + np.roll(y, -1)) * cross) / (6 * signed)
        centroid = centre + cx * u + cy * v
        offset = np.hypot(cx, cy)
        radius = np.sqrt(abs(signed) / np.pi)
        x2, y2 = np.roll(x, -1), np.roll(y, -1)
        c = (y > 0) != (y2 > 0)
        hits = x[c] - y[c] * (x2[c] - x[c]) / (y2[c] - y[c])
        contains = np.sum(hits > 0) % 2
        if contains and offset < offset_limit and radius < radius_limit:
            matches.append(
                dict(
                    xyz_um=centroid.tolist(),
                    radius_um=float(radius),
                    area_um2=float(abs(signed)),
                    offset_um=float(offset),
                )
            )
    return min(matches, key=lambda a: a["offset_um"]) if matches else None


def normal_at(q, s, p, half=0.12):
    """Estimate the centerline tangent around an arc position."""
    n = point(q + half, s, p) - point(q - half, s, p)
    return n / np.linalg.norm(n)


def compact_cut(mesh, q, s, p, head):
    """Measure a spine head and check its boundary topology."""
    try:
        report, surface = measure(mesh, point(q, s, p), normal_at(q, s, p), head)
        return report, surface
    except (ValueError, IndexError, AssertionError):
        return {"valid": False}, None
