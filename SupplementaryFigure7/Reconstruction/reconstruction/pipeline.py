"""Build the S7 geometry from the archived MICrONS mesh and SWC."""

import json

import numpy as np

from .mesh import branch_rows, extract_path, shaft_sections, spine_candidates
from .spines import refine_spines
from .cable import soma_profile, assemble_cable, export_geometry


def prepare_geometry(folder):
    """Measure the cell surface and return the complete Figure S7 cable geometry."""
    swc = np.loadtxt(folder / "cell.swc")
    soma_row = swc[swc[:, 1] == 1][0]
    with np.load(folder / "cell_surface.npz", allow_pickle=False) as source:
        vertices, faces = source["vertices_nm"].astype(float) / 1000, source["faces"]
    print("Extracting branch 7140", flush=True)
    branch = extract_path(vertices, faces, branch_rows(swc, 7140))
    cuts = shaft_sections(branch)
    branch["spines"] = spine_candidates(branch)
    print("Measuring spine necks and heads", flush=True)
    spines = refine_spines(branch, soma_row)
    print("Measuring the soma", flush=True)
    soma_branch = extract_path(vertices, faces, branch_rows(swc, 8990))
    soma = soma_profile(vertices, faces, soma_row, soma_branch["shaft"])
    cable = assemble_cable(vertices, faces, swc, branch, cuts, spines, soma)
    projection = json.loads((folder / "projection.json").read_text())
    return export_geometry(cable, projection)
