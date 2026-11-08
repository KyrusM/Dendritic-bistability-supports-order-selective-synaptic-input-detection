"""Extract a dendritic path and its spine candidates from the MICrONS mesh."""

import networkx as nx
import numpy as np
import skeletor
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from .surface import arc, point, coarse_section


def branch_rows(swc, tip):
    """Follow the selected dendritic tip back to its primary branch."""
    lookup = {int(row[0]): row for row in swc}
    identities = [tip]
    while int(lookup[identities[-1]][6]) in lookup:
        parent = int(lookup[identities[-1]][6])
        if int(lookup[parent][1]) not in (3, 4):
            break
        identities.append(parent)
    return np.array([lookup[i] for i in identities[::-1]])


def extract_path(vertices, faces, rows):
    """Crop the path surface and recover its smoothed wavefront centerline."""
    native = rows[:, 2:5]
    keep = cKDTree(native).query(vertices, workers=2)[0] < 4.0
    selected_faces = faces[np.all(keep[faces], axis=1)]
    used = np.unique(selected_faces)
    mapping = np.full(len(vertices), -1, dtype=np.int32)
    mapping[used] = np.arange(len(used))
    mesh = trimesh.Trimesh(vertices[used], mapping[selected_faces], process=False)
    mesh.merge_vertices(digits_vertex=6)
    skeleton = skeletor.skeletonize.by_wavefront(mesh, waves=1, step_size=1, progress=False)
    skeleton.reindex()
    points, edges = skeleton.vertices, skeleton.edges
    graph = nx.Graph()
    graph.add_nodes_from(range(len(points)))
    graph.add_weighted_edges_from(
        (int(a), int(b), float(np.linalg.norm(points[a] - points[b]))) for a, b in edges
    )
    component = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(component).copy()
    nodes = np.array(sorted(component))
    distance, nearest = cKDTree(points[nodes]).query(native[[0, -1]])
    if distance.max() > 2.0:
        raise ValueError("The surface skeleton does not span the selected SWC path.")
    shaft_nodes = nx.shortest_path(
        graph, int(nodes[nearest[0]]), int(nodes[nearest[1]]), weight="weight"
    )
    shaft = gaussian_filter1d(points[shaft_nodes], sigma=1.0, axis=0, mode="nearest")
    return dict(
        rows=rows, mesh=mesh, vertices=points, graph=graph, shaft_nodes=shaft_nodes, shaft=shaft
    )


def shaft_sections(extracted):
    """Sample area-equivalent shaft radii perpendicular to its centerline."""
    shaft, mesh = extracted["shaft"], extracted["mesh"]
    distance = arc(shaft)
    sections = []
    for at in np.arange(1.0, distance[-1] - 1.0, 0.75):
        center = point(at, distance, shaft)
        normal = point(at + 0.75, distance, shaft) - point(at - 0.75, distance, shaft)
        normal /= np.linalg.norm(normal)
        cut = coarse_section(mesh, center, normal)
        sections.append(dict(position_um=float(at), center_um=center.tolist(), **(cut or {})))
    return sections


def spine_candidates(extracted):
    """Identify side paths and estimate their head centers and neck endpoints."""
    mesh, graph, v = extracted["mesh"], extracted["graph"], extracted["vertices"]
    shaft = extracted["shaft_nodes"]
    off = graph.copy()
    shaftset = set(shaft)
    off.remove_nodes_from(shaft)
    spines = []
    for component in nx.connected_components(off):
        anchors = [(n, t) for n in component for t in graph[n] if t in shaftset]
        if len(anchors) != 1:
            continue
        _, base = anchors[0]
        sub = graph.subgraph(component | {base})
        dist = nx.single_source_dijkstra_path_length(sub, base)
        tip = max(dist, key=dist.get)
        path = nx.shortest_path(sub, base, tip, weight="weight")
        if not 0.7 <= dist[tip] <= 3.5 or len(path) < 6:
            continue
        points = gaussian_filter1d(v[path], sigma=0.8, axis=0, mode="nearest")
        t = arc(points)
        sp = dict(base_node=int(base), tip_node=int(tip), path_nodes=list(map(int, path)))
        near = np.linalg.norm(mesh.vertices - points.mean(axis=0), axis=1) < 4.0
        ff = mesh.faces[np.all(near[mesh.faces], axis=1)]
        used = np.unique(ff)
        mapping = np.full(len(mesh.vertices), -1, dtype=int)
        mapping[used] = np.arange(len(used))
        local = trimesh.Trimesh(mesh.vertices[used], mapping[ff], process=False)
        cuts = []
        for x in np.arange(0.15, t[-1] - 0.1, 0.08):
            center = point(x, t, points)
            normal = point(x + 0.12, t, points) - point(x - 0.12, t, points)
            normal /= np.linalg.norm(normal)
            m = coarse_section(local, center, normal)
            cuts.append(
                dict(
                    distance_um=float(x),
                    center_um=center.tolist(),
                    **(m or dict(status="no_closed_contour"))
                )
            )
        good = [q for q in cuts if "radius_um" in q]
        distal = [q for q in good if q["distance_um"] > 0.5 * t[-1]]
        if len(good) >= 5 and distal:
            head = max(distal, key=lambda q: q["radius_um"])
            neckend = max(0.3, head["distance_um"] - head["radius_um"])
            neck = [q for q in good if q["distance_um"] <= neckend]
            if len(neck) >= 3:
                sp.update(
                    neck_length_um=float(neck[-1]["distance_um"]), head_center_um=head["center_um"]
                )
        spines.append(sp)
    return spines
