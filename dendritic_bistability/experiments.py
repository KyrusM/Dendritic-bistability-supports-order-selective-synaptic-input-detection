"""Figure protocols."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
import numpy as np

from .model import Parameters
from .simulation import run_sequence
from .results import save_example
from .analysis import spike_times


def simulate_trial(task):
    """Run one sequence and return its plateau, voltage and spike measurements.

    Align an imposed spike when requested; save voltages for selected figure examples."""
    trial, base, output, plot_end_ms = task
    values = dict(trial.get("parameters", {}))
    p = replace(base, **values)
    kwargs = dict(
        input_times=trial.get("input_times_ms"),
        factors=trial.get("factors"),
        background=trial.get("background"),
        record_surface=p.observe_recovery or plot_end_ms is not None,
    )

    pulse = trial.get("pulse_ms")
    target = trial.get("spike_target_ms")
    if target is None:
        result = run_sequence(p, trial["order"], pulse_ms=pulse, **kwargs)
    else:
        # Align the directly evoked spike to the requested time.
        pulse = target - 2.0
        for _ in range(4):
            result = run_sequence(p, trial["order"], pulse_ms=pulse, **kwargs)
            spikes = spike_times(result["time_ms"], result["soma_mv"])
            direct = spikes[(spikes >= pulse) & (spikes <= pulse + 8.0)]
            if len(direct) != 1:
                raise RuntimeError("The current pulse must evoke one identifiable spike.")
            error = direct[0] - target
            if abs(error) <= 0.25:
                result["measurements"]["imposed_spike_ms"] = float(direct[0])
                first = p.seed_time_ms + p.interval_ms
                result["measurements"]["further_spike_count"] = int(
                    sum(spikes >= first) - (direct[0] >= first)
                )
                break
            pulse -= error
        else:
            raise RuntimeError("The imposed spike did not reach the requested timing.")

    # Retain voltage examples and return the response measurements.
    result["condition"] = {k: v for k, v in trial.items() if k not in ("parameters", "factors", "background")}
    if plot_end_ms is not None:
        save_example(
            Path(output) / (trial["name"] + ".npz"), result,
            minimum_end_ms=min(result["input_times_ms"]) + plot_end_ms,
        )
    measured = {k: v for k, v in result["measurements"].items() if k != "sequence_plateaus"}
    return {**result["condition"], **measured,
            "observation_end_ms": result["observation_end_ms"]}


def run_trials(parameters, trials, destination, workers=1, limit=None, recordings=(), plot_end_ms=0.0):
    """Run the specified conditions and save response summaries and selected voltage examples."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    selected = trials if limit is None else trials[:limit]
    tasks = [
        (trial, parameters, str(destination), plot_end_ms if trial["name"] in recordings else None)
        for trial in selected
    ]
    rows = []

    def collect(results):
        """Collect response measurements and print progress."""
        for row in results:
            rows.append(row)
            print(f"{len(rows)}/{len(tasks)} {row['name']}", flush=True)

    if workers == 1:
        collect(map(simulate_trial, tasks))
    else:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            collect(pool.map(simulate_trial, tasks))
    (destination / "responses.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


_branco = None


def _native_trial(task):
    """Run one condition in the published Branco model."""
    global _branco
    from .branco import BrancoDendrite

    trial, output, keep_recording = task
    if _branco is None:
        _branco = BrancoDendrite()
    path = Path(output) / (trial["name"] + ".npz") if keep_recording else None
    return _branco.run(trial, path)


def _background_parameters(parameters, trial):
    """Observe jitter and additional-input assays through 80 ms after the last input."""
    if trial["stage"] not in ("C", "D", "E", "F"):
        return parameters
    times = list(trial["input_times_ms"])
    times += [at for group in trial.get("background", {}).values() for _, at in group]
    return replace(parameters, stop_ms=max(times) + 80.0, observe_recovery=False)


def run_background_comparison(parameters, trials, destination, workers=1, limit=None):
    """Compare input timing and additional inputs across the cylindrical and published Branco models."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    trials = trials if limit is None else trials[:limit]
    rows = []
    # Run the published control and cylindrical models in separate processes.
    for native in (True, False):
        selected = [t for t in trials if (t["model"] == "Branco") == native]
        if not selected:
            continue
        function = _native_trial if native else simulate_trial
        tasks = (
            [(t, str(destination), False) for t in selected]
            if native
            else [(t, _background_parameters(parameters, t), str(destination), None)
                  for t in selected]
        )
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for row in pool.map(function, tasks):
                rows.append(row)
                print(f"{len(rows)}/{len(trials)} {row['name']}", flush=True)
    (destination / "responses.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows
