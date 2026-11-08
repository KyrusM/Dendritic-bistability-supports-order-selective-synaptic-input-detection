"""Soma surfaces and soma-to-shaft intersections."""

import numpy as np


def tube(points, radii, sides=12):
    """Construct closed triangular frusta around a centerline."""
    points = np.asarray(points, dtype=float)
    radii = np.asarray(radii, dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-10]
    points = points[keep]
    radii = radii[keep]
    if len(points) < 2:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    vertices = []
    faces = []
    previous = None
    for j, p in enumerate(points):
        tangent = points[min(j + 1, len(points) - 1)] - points[max(j - 1, 0)]
        tangent /= np.linalg.norm(tangent)
        if previous is None:
            axis = np.eye(3)[np.argmin(np.abs(tangent))]
            u = np.cross(axis, tangent)
        else:
            u = previous - np.dot(previous, tangent) * tangent
        u /= np.linalg.norm(u)
        previous = u
        v = np.cross(tangent, u)
        v /= np.linalg.norm(v)
        for k in range(sides):
            a = k * 2 * np.pi / sides
            vertices.append(p + radii[j] * (np.cos(a) * u + np.sin(a) * v))
    for j in range(len(points) - 1):
        for k in range(sides):
            a = j * sides + k
            b = j * sides + (k + 1) % sides
            c = a + sides
            d = b + sides
            faces.extend([[a, b, c], [b, d, c]])
    for k in range(1, sides - 1):
        faces.append([0, k + 1, k])
        end = (len(points) - 1) * sides
        faces.append([end, end + k, end + k + 1])
    return np.asarray(vertices), np.asarray(faces, dtype=int)


def exit_point(origin, end, vertices, faces):
    """Find where a soma-to-dendrite ray exits the soma surface."""
    d = end - origin
    a = vertices[faces[:, 0]]
    e1 = vertices[faces[:, 1]] - a
    e2 = vertices[faces[:, 2]] - a
    p = np.cross(np.broadcast_to(d, e2.shape), e2)
    det = np.einsum("ij,ij->i", e1, p)
    valid = abs(det) > 1e-12
    inv = np.zeros(len(det))
    inv[valid] = 1 / det[valid]
    tvec = origin - a
    u = np.einsum("ij,ij->i", tvec, p) * inv
    q = np.cross(tvec, e1)
    v = np.einsum("ij,j->i", q, d) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    valid &= (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8) & (t > 1e-8) & (t < 1)
    assert valid.any(), ("No soma exit", end.tolist())
    return origin + float(t[valid].min()) * d
