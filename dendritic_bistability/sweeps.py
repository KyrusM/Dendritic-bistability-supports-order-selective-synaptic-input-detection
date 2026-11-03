"""Adaptive conductance and spacing sweeps."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
import json
import multiprocessing
from pathlib import Path

import numpy as np

from .model import Parameters
from .simulation import compare_sequences
from .results import model_hash


@dataclass(frozen=True)
class PropagationSweep:
    """Specify input orders and channel settings for a conductance and spacing sweep."""

    placement: str
    gc_ns: float = 4.075
    ampa_ns: float = 0.35
    background_multiplier: float = 1.0
    fixed_girk: float | None = None
    nmda_limit_ns: float | None = None
    orders: tuple = ((1, 2, 3, 4, 5, 6), (1, 2, 3, 5, 4, 6))

    def __call__(self, point):
        """Test propagation at one spacing and conductance combination.

        Return one bit per propagating input order, or -2 for NMDA conductance above the specified limit."""
        # Convert normalized imbalance to NMDA and GIRK conductances.
        spacing, imbalance = point[:2]
        ampa = point[2] if len(point) == 3 else self.ampa_ns
        girk = self.gc_ns * (1 - imbalance)
        if self.fixed_girk is not None:
            girk = self.fixed_girk * (spacing if self.placement == "shaft" else 1.0)
        nmda = (
            self.gc_ns * (1 + imbalance)
            if self.fixed_girk is None
            else girk * (1 + imbalance) / (1 - imbalance)
        )
        if self.nmda_limit_ns is not None and nmda > self.nmda_limit_ns:
            return -2
        p = Parameters(
            spacing_um=spacing,
            seed_count=5,
            sequence_count=6,
            seed_ampa_ns=2.0,
            soma="none",
            buffer_count=max(10, int(np.ceil(24 / spacing))),
            nmda_ns=nmda,
            ampa_ns=ampa,
            girk_head_ns=girk if self.placement == "head" else 0.0,
            girk_shaft_ns_per_um=girk / spacing if self.placement == "shaft" else 0.0,
            seed_time_ms=100.0,
            initial_voltage_mv=-87.0,
            record_ms=0.25,
            absolute_tolerance=1e-7,
            maximum_step_ms=0.15,
            stop_ms=300.0,
            observe_recovery=False,
            background_multiplier=self.background_multiplier,
        )
        results = compare_sequences(
            p, {str(i): list(order) for i, order in enumerate(self.orders)}, record_surface=False,
        )
        return sum(
            (1 << i) for i, result in enumerate(results.values()) if result["measurements"]["propagated"]
        )


def _evaluate(task):
    """Evaluate one sweep point and return its grid index and propagation class."""
    function, index, point = task
    return index, function(point)


def _refine(values):
    """Insert grid midpoints, preserving matching neighboring classes and marking the others for simulation."""
    for axis in range(values.ndim):
        shape = list(values.shape)
        shape[axis] = 2 * shape[axis] - 1
        refined = np.full(shape, -1, dtype=np.int8)
        even = [slice(None)] * values.ndim
        even[axis] = slice(0, None, 2)
        refined[tuple(even)] = values
        odd, left, right = even.copy(), even.copy(), even.copy()
        odd[axis], left[axis], right[axis] = slice(1, None, 2), slice(None, -1), slice(1, None)
        a, b = values[tuple(left)], values[tuple(right)]
        refined[tuple(odd)] = np.where(a == b, a, -1)
        values = refined
    return values


def adaptive_sweep(function, bounds, destination, *, minimum_depth=4, depth=8, workers=1):
    """Refine propagation boundaries and save a restart checkpoint after each grid level."""
    bounds, destination = np.asarray(bounds, dtype=float), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    settings = json.dumps(
        {"sweep": asdict(function), "model": asdict(Parameters()), "model_code": model_hash()}, sort_keys=True
    )
    initial_depth, values = min(minimum_depth, depth), None

    # Resume a saved grid only when its bounds and model settings match.
    if destination.exists():
        with np.load(destination, allow_pickle=False) as saved:
            if not np.array_equal(saved["bounds"], bounds):
                raise ValueError("Saved sweep bounds differ from the requested bounds.")
            if str(saved["settings"]) != settings:
                raise ValueError("Saved sweep parameters differ from the requested parameters.")
            values, initial_depth = saved["classes"].copy(), int(saved["depth"]) + 1

    # Refine the grid and simulate the points left unclassified.
    for level in range(initial_depth, depth + 1):
        size = 2 ** (level - 1) + 1
        values = np.full((size,) * len(bounds), -1, np.int8) if values is None else _refine(values)
        axes = [np.linspace(a, b, size) for a, b in bounds]
        tasks = [
            (function, tuple(index), tuple(axes[j][i] for j, i in enumerate(index)))
            for index in np.argwhere(values == -1)
        ]
        if workers == 1:
            results = map(_evaluate, tasks)
            for index, value in results:
                values[index] = value
        else:
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                for index, value in pool.map(_evaluate, tasks, chunksize=1):
                    values[index] = value
        np.savez_compressed(destination, classes=values, bounds=bounds, depth=level, settings=settings)
        destination.with_suffix(".json").write_text(
            json.dumps(
                dict(classes=values.tolist(), bounds=bounds.tolist(), depth=level), separators=(",", ":")
            )
        )
        print(f"Depth {level}: {len(tasks)} simulations", flush=True)
    return values


def contiguous_band(values, coordinates, minimum_width):
    """Return the widest continuous true interval meeting the required coordinate width."""
    bands = contiguous_bands(values, coordinates, minimum_width)
    return max(bands, key=lambda band: coordinates[band[1]] - coordinates[band[0]]) if bands else None


def contiguous_bands(values, coordinates, minimum_width):
    """Return all continuous true intervals meeting the required coordinate width."""
    edges = np.diff(np.r_[False, values, False].astype(int))
    bands = [
        (a, b)
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1)
        if coordinates[b] - coordinates[a] >= minimum_width - 1e-12
    ]
    return bands


def minimum_spacing(classes, bounds, width=0.1, largest_component=False, control_name="ampa_ns"):
    """Find the first spacing with a sufficiently wide order-selective interval at each control value."""
    mask = classes == 1
    # Optionally retain the largest connected order-selective range.
    if largest_component:
        from scipy.ndimage import label

        components, _ = label(mask, structure=np.ones((3, 3, 3)))
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        mask = components == np.argmax(sizes) if sizes.max() else np.zeros_like(mask)
    spacing, imbalance, ampa = [np.linspace(a, b, n) for (a, b), n in zip(bounds, classes.shape)]
    curves = []

    # Scan spacings from smallest to largest at each control value.
    for k, conductance in enumerate(ampa):
        point = {control_name: float(conductance), "spacing_um": None, "imbalance_range": None}
        for i, distance in enumerate(spacing):
            bands = contiguous_bands(mask[i, :, k], imbalance, width)
            if bands:
                band = max(bands, key=lambda b: imbalance[b[1]] - imbalance[b[0]])
                point.update(
                    spacing_um=float(distance),
                    imbalance_range=[float(imbalance[band[0]]), float(imbalance[band[1]])],
                    imbalance_ranges=[[float(imbalance[a]), float(imbalance[b])] for a, b in bands],
                )
                break
        curves.append(point)
    return curves


