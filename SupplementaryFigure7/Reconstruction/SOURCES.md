# Sources

MICrONS Consortium (2025), *Functional connectomics spanning multiple areas of mouse visual cortex*, Nature 640, 435–447, [doi:10.1038/s41586-025-08790-w](https://doi.org/10.1038/s41586-025-08790-w).

The reconstruction uses cell **864691135117980637**, with dendritic tip **SWC node 7140**. The soma profile uses the original basal-branch direction toward SWC node 8990.

| Input | Source |
|---|---|
| `inputs/cell.swc` | Released proofread skeleton: [MICrONS static links](https://storage.googleapis.com/microns-static-links/skel/swc/proofread/864691135117980637.swc). Coordinates and radii are in µm. |
| `inputs/cell_surface.npz` | LOD0 mesh from the [MICrONS m1300 segmentation](https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg_m1300/info). Vertices are in nm; faces index the original vertices. |
| `inputs/projection.json` | Camera basis and input-marker directions used in Figure S7. |

The mesh retains the released coordinates and connectivity. Its native cloud mesh was stored as NumPy arrays for local processing. `inputs/sources.json` records the input checksums.

Wavefront skeletonization uses [Skeletor](https://github.com/navis-org/skeletor). Surface operations use [Trimesh](https://github.com/mikedh/trimesh). Their versions are specified in `requirements.txt`. The measurement and cable-reduction functions were written for this study.
