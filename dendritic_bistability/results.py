"""Read and write simulation results."""

from pathlib import Path
import hashlib
import json

import numpy as np


def model_hash():
    """Identify the model source and parameters associated with saved sweep results."""
    folder = Path(__file__).resolve().parent
    files = [
        folder / name
        for name in (
            "model.py",
            "simulation.py",
            "analysis.py",
            "sweeps.py",
            "variability.py",
            "parameters.json",
        )
    ]
    files += list((folder / "mechanisms").glob("*.mod"))
    files += list((folder / "mechanisms").glob("*.inc"))
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_result(path, result):
    """Save voltage arrays or grids with their parameters and measurements in a compressed NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    metadata = {k: v for k, v in result.items() if k not in arrays}
    np.savez_compressed(path, **arrays, metadata=np.asarray(json.dumps(metadata)))


def load_result(path):
    """Read arrays and metadata from a saved NPZ file."""
    with np.load(path, allow_pickle=False) as data:
        return {
            **json.loads(str(data["metadata"])),
            **{k: data[k].copy() for k in data.files if k != "metadata"},
        }


def save_example(path, result, minimum_end_ms=0.0):
    """Save example voltages through plateau collapse and recovery, omitting later baseline."""
    p, t = result["parameters"], result["time_ms"]
    start = result.get("stimulus_start_ms", p["seed_time_ms"])
    baseline = np.searchsorted(t, start) - 1
    voltage = np.vstack([result["head_mv"], result["shaft_mv"], result.get("soma_mv", np.empty((0, len(t))))])
    away = np.max(abs(voltage - voltage[:, baseline, None]), axis=0) > p["recovery_tolerance_mv"]
    last = t[np.flatnonzero(away)[-1]] if away.any() else start
    end = max(
        max(last, max(result["input_times_ms"])) + p["recovery_ms"] + p["record_ms"],
        minimum_end_ms + p["record_ms"],
    )
    keep = (t >= start - p["interval_ms"]) & (t <= end)
    saved = dict(result, recorded_time_window_ms=[float(t[keep][0]), float(t[keep][-1])])
    for name in ("time_ms", "head_mv", "shaft_mv", "site_shaft_mv", "soma_mv"):
        if name in saved:
            saved[name] = saved[name][..., keep]
    save_result(path, saved)
