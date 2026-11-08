"""Minimum-spacing curves and conductance-imbalance projections."""

import math
import numpy as np
from matplotlib.colors import to_rgb


COLORS = {"shaft": ("#ae7717", "#2166ac", "#7a3e9d"), "head": ("#18846b", "#ce5145", "#555555")}


def _segments(rows, overlap=False):
    """Split curves at missing spacings and, if requested, nonoverlapping imbalance intervals."""
    segments, current = [], []
    for row in rows:
        if row["spacing_um"] is None:
            if current:
                segments.append(current)
                current = []
            continue
        if current and overlap:
            previous = current[-1]["r_range"]
            low, high = row["r_range"]
            if min(previous[1], high) < max(previous[0], low) - 1e-12:
                segments.append(current)
                current = []
        current.append(row)
    if current:
        segments.append(current)
    return segments


def project_minima(curves):
    """Convert minimum-spacing curves into the spacing, AMPA and imbalance projections for Figure 7."""
    output = []
    ranges = {"shaft": [0.0, 22.0], "head": [0.0, 42.86554276427627]}
    for key, source in curves.items():
        placement = key.split("_", 1)[0]
        level = 0 if "_full_" in key else 1 if "_half_" in key else 2
        width = float(key.rsplit("_", 1)[-1])
        rows = [dict(row, r_range=row.get("r_range", row.get("imbalance_range"))) for row in source]
        rgb = np.asarray(to_rgb(COLORS[placement][level]))
        kir = (1.15 if placement == "shaft" else 5.7) / 2**level
        item = dict(
            placement=placement,
            width=width,
            kir=kir,
            color=(rgb if width == 0.1 else 0.55 + 0.45 * rgb).tolist(),
            reference_color=rgb.tolist(),
            main=[],
            left_lines=[],
            left_polygons=[],
            top_lines=[],
            top_polygons=[],
        )
        # Project spacing and imbalance against AMPA conductance.
        for segment in _segments(rows):
            item["main"].append([[p["spacing_um"], p["ampa_ns"]] for p in segment])
            low = [[p["r_range"][0], p["ampa_ns"]] for p in segment]
            high = [[p["r_range"][1], p["ampa_ns"]] for p in segment]
            item["left_lines"].extend([low, high])
            if len(segment) > 1:
                item["left_polygons"].append(high + low[::-1])

        # Project continuous imbalance intervals against spacing.
        for segment in _segments(rows, overlap=True):
            low = [[p["spacing_um"], -p["r_range"][0]] for p in segment]
            high = [[p["spacing_um"], -p["r_range"][1]] for p in segment]
            item["top_lines"].extend([low, high, [low[0], high[0]], [low[-1], high[-1]]])
            item["top_polygons"].extend(
                [low[i], low[i + 1], high[i + 1], high[i]]
                for i in range(len(low) - 1)
                if low[i][0] != low[i + 1][0]
            )

        # Add the 2.4-nS NMDA reference curve.
        x = np.linspace(*ranges[placement], 1201)
        g = kir * x if placement == "shaft" else np.full_like(x, kir)
        item["hairline"] = np.column_stack([x, -(2.4 - g) / (2.4 + g)]).tolist()
        output.append(item)
    displayed = {}
    for placement in ("shaft", "head"):
        values = [
            v
            for name, rows in curves.items()
            if name.startswith(placement)
            for row in rows
            if row["spacing_um"] is not None
            for v in row.get("r_range", row.get("imbalance_range"))
        ]
        displayed[placement] = (
            [
                round(math.floor((min(values) - 0.025) / 0.05) * 0.05, 2),
                round(math.ceil((max(values) + 0.025) / 0.05) * 0.05, 2),
            ]
            if values
            else [-0.99, 0.99]
        )
    return dict(curves=output, displayed_r_range=displayed, x_ranges=ranges)
