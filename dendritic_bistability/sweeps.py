"""Adaptive conductance and spacing sweeps."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
import hashlib
import itertools
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

    def parameters(self, point):
        """Return dendrite parameters for one sweep point."""
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
        return Parameters(
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

    def __call__(self, point):
        """Return one bit per propagating order, or -2 above the NMDA conductance limit."""
        p = self.parameters(point)
        if self.nmda_limit_ns is not None and p.nmda_ns > self.nmda_limit_ns:
            return -2
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
    models = [asdict(function.parameters(point)) for point in itertools.product(*bounds)]
    fixed = {name: value for name, value in models[0].items() if all(model[name] == value for model in models)}
    settings = json.dumps(
        {"sweep": asdict(function), "model": fixed,
         "varying_model_parameters": [name for name in models[0] if name not in fixed],
         "model_code": model_hash()}, sort_keys=True
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




@dataclass(frozen=True)
class ConductanceCondition:
    """Specify GIRK placement and conductance for one sweep."""

    name: str
    placement: str
    girk: float

    def manifest(self):
        """Return the parameters stored with each saved sweep grid."""
        p = Parameters()
        return dict(asdict(self), neck_length_um=p.neck_length_um,
                    neck_resistance_gohm=p.neck_resistance_gohm,
                    resistivity_ohm_cm=p.resistivity_ohm_cm,
                    spine_resistivity_ohm_cm=p.spine_resistivity_ohm_cm)


UNKNOWN = -1
OUTSIDE = -2
NMDA_LIMIT_NS = 9.6


def domain_mask(condition, bounds, size):
    """Identify grid points within the per-synapse NMDA conductance limit."""
    spacing = np.linspace(*bounds[0], size)[:, None]
    imbalance = np.linspace(*bounds[1], size)[None, :]
    girk = condition["girk"] * (spacing if condition["placement"] == "shaft" else 1.0)
    admitted = girk * (1 + imbalance) / (1 - imbalance) <= NMDA_LIMIT_NS + 1e-12
    return np.broadcast_to(admitted[:, :, None], (size, size, size)).copy()


def refine_grid(grid, admitted):
    """Infer midpoint classes from agreeing, admissible neighbors."""
    for axis in range(3):
        shape = list(grid.shape)
        shape[axis] = 2 * shape[axis] - 1
        finer = np.full(shape, UNKNOWN, np.int8)
        even = [slice(None)] * 3
        even[axis] = slice(0, None, 2)
        odd, left, right = even.copy(), [slice(None)] * 3, [slice(None)] * 3
        odd[axis], left[axis], right[axis] = slice(1, None, 2), slice(None, -1), slice(1, None)
        a, b = grid[tuple(left)], grid[tuple(right)]
        finer[tuple(even)] = grid
        finer[tuple(odd)] = np.where((a == b) & (a >= 0), a, UNKNOWN)
        grid = finer
    grid[~admitted] = OUTSIDE
    return grid


def first_order(coarse, point):
    """Choose the first input order from the neighboring coarse-grid classes."""
    if coarse is None:
        return "ordered"
    axes = [[i // 2] if i % 2 == 0 else [i // 2, i // 2 + 1] for i in point]
    classes = [int(coarse[p]) for p in itertools.product(*axes)]
    return "swapped" if classes.count(3) > classes.count(0) else "ordered"


def boundary_points(grid, admitted):
    """Locate propagation transitions and the analytical conductance boundary."""
    propagation, conductance = np.zeros(grid.shape, bool), np.zeros(grid.shape, bool)
    for axis in range(3):
        left, right = [slice(None)] * 3, [slice(None)] * 3
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(left), tuple(right)
        changed = (grid[a] != grid[b]) & admitted[a] & admitted[b]
        propagation[a] |= changed
        propagation[b] |= changed
        crossing = admitted[a] != admitted[b]
        conductance[a] |= crossing & admitted[a]
        conductance[b] |= crossing & admitted[b]
    return propagation, conductance


def _write_json(path, data):
    """Replace a small progress checkpoint only after its complete write."""
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def _write_grid(path, grid, bounds, depth, settings, known):
    """Save a completed grid with its direct classifications and settings."""
    indices = sorted(known)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as output:
        np.savez_compressed(
            output, classes=grid, admitted=grid >= 0, bounds=bounds, depth=depth,
            settings=np.asarray(settings),
            direct_indices=np.asarray(indices, dtype=np.int16).reshape(-1, 3),
            direct_classes=np.asarray([known[index][0] for index in indices], dtype=np.int8),
            both_orders_evaluated=np.asarray([known[index][1] for index in indices], dtype=bool),
        )
    temporary.replace(path)


def _settings(condition, bounds, extension, base_hash=None):
    """Return model settings and source hash for checkpoint validation."""
    folder = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    files = list(folder.glob("*.py")) + [folder / "parameters.json"]
    files += list((folder / "mechanisms").glob("*.mod"))
    files += list((folder / "mechanisms").glob("*.inc"))
    for path in sorted(files):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return json.dumps(dict(condition=condition, bounds=bounds.tolist(), extension=extension,
                           base_grid_sha256=base_hash, model_code=digest.hexdigest()),
                      sort_keys=True, separators=(",", ":"))


def _load_points(path, settings):
    """Load evaluated points or start a new checkpoint."""
    known = {}
    if not path.exists():
        path.write_text(json.dumps(dict(settings=settings)) + "\n")
        return known
    valid_bytes = 0
    with path.open("rb") as stream:
        first = stream.readline()
        if not first or json.loads(first).get("settings") != settings:
            raise ValueError("Saved Propagation sweep point settings differ from this run.")
        valid_bytes = stream.tell()
        while True:
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                break
            for i, j, k, category, both in json.loads(line)["points"]:
                if category not in (0, 1, 2, 3):
                    raise ValueError("A saved point has an invalid propagation class.")
                known[int(i), int(j), int(k)] = int(category), bool(both)
            valid_bytes = stream.tell()
    # An interrupted write leaves only its final, unfinished bundle to repeat.
    with path.open("r+b") as stream:
        stream.truncate(valid_bytes)
    return known


def _evaluate_bundle(task):
    """Evaluate AMPA values at fixed spacing and imbalance."""
    from .simulation import evaluate_propagation_bundle

    condition, spacing, imbalance, requests = task
    return evaluate_propagation_bundle(condition, spacing, imbalance, requests)


class _PropagationSampler:
    """Manage the saved classifications and grid levels for one condition."""

    def __init__(self, condition, bounds, destination, extension):
        """Prepare checkpoints and reuse the base boundary for extensions."""
        self.condition = dict(condition.manifest(), extension=bool(extension))
        self.bounds = np.asarray(bounds, dtype=float)
        self.destination = Path(destination)
        self.folder = self.destination.with_suffix("")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.base_plane = None
        base_hash = None
        if extension:
            base_path = self.destination.parent / (self.condition["name"] + ".npz")
            if base_path == self.destination or not base_path.exists():
                raise ValueError("Run the base Propagation sweep condition before its separately named extension.")
            with np.load(base_path, allow_pickle=False) as base:
                if (int(base["depth"]) != 8 or base["classes"].shape != (129, 129, 129)
                        or not np.array_equal(base["bounds"][1:], self.bounds[1:])
                        or base["bounds"][0, 1] != self.bounds[0, 0]):
                    raise ValueError("The extension must start at the completed base grid's last spacing.")
                original = json.loads(str(base["settings"]))["condition"]
                if dict(original, extension=True) != self.condition:
                    raise ValueError("The base and extension must share their scientific configuration.")
                self.base_plane = base["classes"][-1].copy()
            base_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
        self.settings = _settings(self.condition, self.bounds, extension, base_hash)
        self.points_path = self.folder / "classified_points.jsonl"
        self.known = _load_points(self.points_path, self.settings)
        if not extension and any(category == 2 for category, _ in self.known.values()):
            raise RuntimeError("A saved swapped-only result requires inspection before resuming this base sweep.")

    def _coordinates(self, index):
        """Convert a grid index to spacing, imbalance and AMPA."""
        return tuple(float(lo + (hi - lo) * i / 128)
                     for i, (lo, hi) in zip(index, self.bounds))

    def insert_known(self, grid, scale):
        """Restore directly evaluated classifications onto their grid positions."""
        for index, (category, _) in self.known.items():
            if all(i % scale == 0 for i in index):
                grid[tuple(i // scale for i in index)] = category

    def _tasks(self, points, coarse, scale, force_pair):
        """Group up to eight AMPA values per spacing and imbalance."""
        groups = {}
        for point in sorted(set(tuple(map(int, point)) for point in points)):
            index = tuple(i * scale for i in point)
            if index in self.known and (not force_pair or self.known[index][1]):
                continue
            _, _, ampa = self._coordinates(index)
            groups.setdefault(index[:2], []).append(dict(
                ampa_index=index[2], ampa_ns=ampa, first_order=first_order(coarse, point),
                force_pair=force_pair,
            ))
        for xy, requests in sorted(groups.items()):
            spacing, imbalance, _ = self._coordinates((*xy, 0))
            for offset in range(0, len(requests), 8):
                yield self.condition, spacing, imbalance, requests[offset:offset + 8], xy

    def _record(self, task, results):
        """Checkpoint a finished bundle and retain its order-evaluation flags."""
        requests, xy = task[3], task[4]
        if set(results) != {row["ampa_index"] for row in requests}:
            raise ValueError("A Propagation sweep bundle returned different AMPA indices than requested.")
        records = []
        for request in requests:
            k = request["ampa_index"]
            result = results[k]
            category, both = int(result["classification"]), bool(result["both_orders_evaluated"])
            if category not in (0, 1, 2, 3):
                raise ValueError("A Propagation sweep bundle returned an invalid propagation class.")
            if request["force_pair"] and not both:
                raise ValueError("An interpolation check requires both input orders.")
            records.append([*xy, k, category, int(both)])
        with self.points_path.open("a") as output:
            output.write(json.dumps(dict(points=records), separators=(",", ":")) + "\n")
        for i, j, k, category, both in records:
            self.known[i, j, k] = category, bool(both)
        if not self.condition["extension"] and any(row[3] == 2 for row in records):
            raise RuntimeError("Swapped-only propagation contradicts the sequence-nesting assumption; "
                               "the measured classifications remain in classified_points.jsonl.")

    def evaluate(self, pool, points, coarse, scale, force_pair=False):
        """Evaluate ascending AMPA bundles within each spacing/imbalance pair."""
        points = sorted(set(tuple(map(int, point)) for point in points))
        tasks = iter(self._tasks(points, coarse, scale, force_pair))
        completed = 0
        if pool is None:
            for task in tasks:
                self._record(task, _evaluate_bundle(task[:4]))
                completed += 1
        else:
            tasks = list(tasks)
            results = pool.map(_evaluate_bundle, (task[:4] for task in tasks), chunksize=1)
            for completed, (task, result) in enumerate(zip(tasks, results), 1):
                self._record(task, result)
                if completed % 50 == 0:
                    print(f"  {completed} AMPA bundles completed", flush=True)
        return {point: self.known[tuple(i * scale for i in point)][0] for point in points}

    def check_interpolation(self, pool, grid, coarse, admitted, scale, depth):
        """Verify inferred classes and expand checks around mismatches."""
        path = self.folder / f"interpolation_depth_{depth}.json"
        if path.exists():
            plan = json.loads(path.read_text())
            if plan["settings"] != self.settings:
                raise ValueError("Saved Propagation sweep interpolation settings differ from this run.")
            if plan["complete"]:
                return
        else:
            inferred = admitted.copy()
            for index, (_, both) in self.known.items():
                if both and all(i % scale == 0 for i in index):
                    inferred[tuple(i // scale for i in index)] = False
            if self.base_plane is not None:
                inferred[0] = False
            propagation, conductance = boundary_points(grid, admitted)
            rng = np.random.default_rng(20261004 + depth + (100 if self.condition["placement"] == "head" else 0))
            selected = set()
            for region, count in ((propagation, 64), (conductance, 32), (admitted, 128)):
                choices = np.argwhere(region & inferred)
                if len(choices):
                    selected.update(tuple(map(int, choices[i]))
                                    for i in rng.choice(len(choices), min(count, len(choices)), replace=False))
            points = sorted(selected)
            plan = dict(settings=self.settings, selected=[list(p) for p in points],
                        predicted=[int(grid[p]) for p in points], waves=[], complete=False)
            _write_json(path, plan)

        points = [tuple(point) for point in plan["selected"]]
        answers = self.evaluate(pool, points, coarse, scale, force_pair=True)
        frontier = set()
        for point, expected in zip(points, plan["predicted"]):
            grid[point] = answers[point]
            if expected != answers[point]:
                frontier.add(point)
        visited, wave = set(), 0
        while frontier:
            if wave < len(plan["waves"]):
                record = plan["waves"][wave]
                points = [tuple(point) for point in record["points"]]
            else:
                neighbors = set()
                for point in frontier:
                    neighbors.update(p for p in itertools.product(
                        *(range(max(0, i - 1), min(grid.shape[0], i + 2)) for i in point)) if admitted[p])
                points = sorted(neighbors - visited)
                if not points:
                    break
                record = dict(points=[list(point) for point in points],
                              predicted=[int(grid[point]) for point in points])
                plan["waves"].append(record)
                _write_json(path, plan)
            answers = self.evaluate(pool, points, coarse, scale, force_pair=True)
            frontier = set()
            for point, expected in zip(points, record["predicted"]):
                grid[point] = answers[point]
                if expected != answers[point]:
                    frontier.add(point)
            visited.update(points)
            wave += 1
        plan["complete"] = True
        _write_json(path, plan)

    def run(self, depth, pool):
        """Refine and save each grid level through the requested depth."""
        grid = None
        for level in range(4, depth + 1):
            path = self.folder / f"depth_{level}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if str(saved["settings"]) != self.settings:
                        raise ValueError("Saved Propagation sweep grid settings differ from this run.")
                    grid = saved["classes"].copy()
                    size = 2 ** (level - 1) + 1
                    admitted = domain_mask(self.condition, self.bounds, size)
                    if (int(saved["depth"]) != level or grid.shape != (size, size, size)
                            or not np.array_equal(saved["bounds"], self.bounds)
                            or not np.all((grid[admitted] >= 0) & (grid[admitted] <= 3))
                            or not np.all(grid[~admitted] == OUTSIDE)):
                        raise ValueError("A saved Propagation sweep grid has inconsistent dimensions or classes.")
                continue
            size, scale = 2 ** (level - 1) + 1, 2 ** (8 - level)
            admitted = domain_mask(self.condition, self.bounds, size)
            coarse = grid
            grid = np.full((size, size, size), UNKNOWN, np.int8) if coarse is None else refine_grid(coarse, admitted)
            grid[~admitted] = OUTSIDE
            if self.base_plane is not None:
                grid[0] = self.base_plane[::scale, ::scale]
            self.insert_known(grid, scale)
            points = np.argwhere(grid == UNKNOWN)
            answers = self.evaluate(pool, points, coarse, scale)
            for point, category in answers.items():
                grid[point] = category
            if level >= 6:
                self.check_interpolation(pool, grid, coarse, admitted, scale, level)
            self.insert_known(grid, scale)
            if not (np.all(grid[admitted] >= 0) and np.all(grid[~admitted] == OUTSIDE)):
                raise ValueError("The completed Propagation sweep grid contains unclassified or inadmissible values.")
            if self.base_plane is not None:
                grid[0] = self.base_plane[::scale, ::scale]
            _write_grid(path, grid, self.bounds, level, self.settings, self.known)
            _write_grid(self.destination, grid, self.bounds, level, self.settings, self.known)
            print(f"Depth {level}: {len(points)} requested points; grid {size}³ saved", flush=True)
        # Select the requested saved level even when a deeper checkpoint also exists.
        matches = False
        if self.destination.exists():
            with np.load(self.destination, allow_pickle=False) as saved:
                matches = (str(saved["settings"]) == self.settings and int(saved["depth"]) == depth
                           and np.array_equal(saved["classes"], grid))
        if not matches:
            _write_grid(self.destination, grid, self.bounds, depth, self.settings, self.known)
        return grid


def sample_propagation_grid(condition, bounds, destination, *, depth=8, workers=1, extension=False):
    """Sample one Propagation sweep condition, retaining completed grids and classifications."""
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Propagation sweep bounds must give increasing finite spacing, imbalance and AMPA intervals.")
    if bounds[0, 0] <= 0 or bounds[2, 0] < 0 or bounds[1, 0] <= -1 or bounds[1, 1] >= 1:
        raise ValueError("Propagation sweep requires positive spacing, nonnegative AMPA and imbalance strictly between -1 and 1.")
    if depth not in range(4, 9) or workers < 1:
        raise ValueError("Propagation sweep depth must be 4–8 and workers must be positive.")
    sampler = _PropagationSampler(condition, bounds, destination, extension)
    if workers == 1:
        return sampler.run(depth, None)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return sampler.run(depth, pool)


def minimum_selective_spacing(classes, bounds, width):
    """Find the smallest sampled spacing with a qualifying selective interval."""
    spacing, imbalance, ampa = [
        np.linspace(*extent, count) for extent, count in zip(bounds, classes.shape)
    ]
    rows = []
    for k, conductance in enumerate(ampa):
        row = dict(ampa_ns=float(conductance), spacing_um=None, r_range=None)
        for i, distance in enumerate(spacing):
            edges = np.diff(np.r_[False, classes[i, :, k] == 1, False].astype(int))
            bands = [
                (a, b)
                for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1)
                if imbalance[b] - imbalance[a] >= width - 1e-12
            ]
            if bands:
                a, b = max(bands, key=lambda band: imbalance[band[1]] - imbalance[band[0]])
                row.update(
                    spacing_um=float(distance),
                    r_range=[float(imbalance[a]), float(imbalance[b])],
                    spacing_bracket_um=[float(spacing[max(0, i - 1)]), float(distance)],
                    at_lower_spacing_limit=i == 0,
                )
                break
        rows.append(row)
    return rows


def minimum_spacing_curves(base_path, extension_path=None):
    """Extract both imbalance widths, using an extension where the base has no qualifying interval."""
    with np.load(base_path, allow_pickle=False) as data:
        curves = {
            width: minimum_selective_spacing(data["classes"], data["bounds"], width)
            for width in (0.05, 0.1)
        }
    if extension_path is None:
        return curves
    with np.load(extension_path, allow_pickle=False) as data:
        extended = {
            width: minimum_selective_spacing(data["classes"], data["bounds"], width)
            for width in curves
        }
    for width, rows in curves.items():
        combined = []
        for row in rows:
            if row["spacing_um"] is not None:
                combined.append(row)
                continue
            match = next(
                (point for point in extended[width] if abs(point["ampa_ns"] - row["ampa_ns"]) < 1e-12),
                None,
            )
            if match is not None:
                combined.append(match)
        curves[width] = combined
    return curves


def run_fixed_girk_sweep(condition, folder, bounds, *, depth=8, extension_bounds=None,
                         extension_depth=7, workers=1, rerun=True):
    """Sample one fixed GIRK conductance and save minimum spacing at two imbalance widths."""
    folder = Path(folder)
    path = folder / f"{condition.name}.npz"
    extension = folder / f"{condition.name}_extension.npz" if extension_bounds is not None else None
    if rerun:
        sample_propagation_grid(condition, bounds, path, depth=depth, workers=workers)
        if extension is not None:
            sample_propagation_grid(condition, extension_bounds, extension,
                                    depth=extension_depth, workers=workers, extension=True)
    curves = {
        f"{condition.name}_width_{width}": rows
        for width, rows in minimum_spacing_curves(path, extension).items()
    }
    (folder / f"{condition.name}_curves.json").write_text(json.dumps(curves, indent=2) + "\n")
    return curves
