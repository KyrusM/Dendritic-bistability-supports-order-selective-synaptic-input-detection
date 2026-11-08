"""Front propagation under synaptic and geometric variability."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from neuron import h

from .analysis import measure, spike_times
from .model import CHANNELS, build_dendrite
from .results import model_hash
from .variability_sampling import FAMILIES, SDS, SAMPLES, SHUFFLES, GIRK_CONDUCTANCES


def _write_json(path, value):
    """Replace a JSON file after its complete contents have been written."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def _model_parameters(model):
    """Read the instantiated geometry, membrane properties and channel conductances."""
    p = model.parameters
    sections = [*model.sections, *model.necks, *model.heads, model.soma]
    membrane = [dict(
        length_um=section.L, diameter_um=section(0.5).diam, nseg=section.nseg,
        resistivity_ohm_cm=section.Ra, capacitance_uf_cm2=section.cm,
        leak_s_cm2=section.g_pas, leak_reversal_mv=section.e_pas,
        area_um2=sum(h.area(seg.x, sec=section) for seg in section),
    ) for section in sections]
    head_area = np.array([10 * h.area(0.5, sec=head) for head in model.heads])
    return dict(
        shaft_sections=len(model.sections), shaft_compartments=sum(section.nseg for section in model.sections),
        spine_count=len(model.heads),
        sections=membrane, spine_positions_um=model.positions.tolist(),
        neck_resistances_gohm=[
            0.04 * neck.Ra * neck.L / (np.pi * neck(0.5).diam**2) / 1000
            for neck in model.necks
        ],
        nmda_ns=[head(0.5).nmda.g_at_e * area for head, area in zip(model.heads, head_area)],
        ampa_ns=[head(0.5).ampa.g_peak * area for head, area in zip(model.heads, head_area)],
        girk_head_ns=[head(0.5).girk.g_at_e * area for head, area in zip(model.heads, head_area)],
        shaft_positions_um=[
            float(a + seg.x * section.L)
            for a, section in zip(model.boundaries[:-1], model.sections) for seg in section
        ],
        girk_shaft_ns_per_um=[
            seg.girk.g_at_e * 10 * np.pi * seg.diam for section in model.sections for seg in section
        ],
        shaft_channels={
            name: {key: getattr(model.sections[0], key) for key in values}
            for name, values in CHANNELS["shaft"]["mechanisms"].items()
        },
        channel_globals={name: getattr(h, name) for name in CHANNELS["globals"]},
        temperature_c=h.celsius, seed_time_ms=p.seed_time_ms, interval_ms=p.interval_ms,
        head_half_axial_resistance_mohm=[
            0.04 * head.Ra * head.L / (2 * np.pi * head(0.5).diam**2) for head in model.heads
        ],
        membrane_area_um2=sum(section["area_um2"] for section in membrane),
        capacitance_pf=sum(section["area_um2"] * section["capacitance_uf_cm2"] * 0.01 for section in membrane),
        passive_leak_ns=sum(section["area_um2"] * section["leak_s_cm2"] * 10 for section in membrane),
    )


def _record(model):
    """Record time and head, shaft and somatic voltages at the specified interval."""
    interval = model.parameters.record_ms

    def vector(reference):
        """Attach a sampled recording vector to one NEURON variable."""
        return h.Vector().record(reference, interval)

    return dict(
        time_ms=vector(h._ref_t),
        head_mv=[vector(head(0.5)._ref_v) for head in model.heads],
        shaft_mv=[vector(seg._ref_v) for section in model.sections for seg in section],
        site_shaft_mv=[vector(model.shaft_at(float(x))._ref_v) for x in model.positions],
        soma_mv=vector(model.soma(0.5)._ref_v),
    )


def _voltages(recordings, model):
    """Copy the current recordings and their input identities and times."""
    return {
        **{name: np.array(values) for name, values in recordings.items()},
        "input_sites": model.input_sites, "input_times_ms": model.input_times,
        "stimulus_start_ms": model.parameters.seed_time_ms,
    }


def _recovered(data, end, p):
    """Check voltage recovery throughout the final 200 ms of observation."""
    t = data["time_ms"]
    baseline = np.searchsorted(t, p.seed_time_ms) - 1
    late = t >= end - p.recovery_ms
    voltage = np.vstack([data["head_mv"], data["shaft_mv"], data["soma_mv"]])
    spikes = spike_times(t, data["soma_mv"])
    error = float(np.max(abs(voltage[:, late] - voltage[:, baseline, None])))
    return bool(
        error <= p.recovery_tolerance_mv
        and data["head_mv"][:, late].max() < -30
        and not np.any(spikes >= end - p.recovery_ms)
    )


def _responses(model, orders):
    """Simulate each input order and return compact plateau measurements."""
    p = model.parameters
    recordings = _record(model)
    solver = h.CVode()
    solver.active(1)
    solver.atol(p.absolute_tolerance)
    solver.rtol(0)

    def late_step():
        """Use a 1-ms maximum step after the input sequence and plateau response."""
        solver.maxstep(1.0)

    try:
        for condition in orders:
            model.set_inputs(condition["order"])
            solver.maxstep(p.maximum_step_ms)
            h.finitialize(p.initial_voltage_mv)
            for onset in [p.seed_time_ms, *model.input_times]:
                solver.event(float(onset), solver.re_init)
            solver.event(1500.0, late_step)
            for end in (p.stop_ms, 4000.0, p.maximum_stop_ms):
                h.continuerun(end)
                data = _voltages(recordings, model)
                if _recovered(data, end, p):
                    break
            else:
                raise RuntimeError("Membrane voltages did not recover by 6000 ms.")
            measured = measure(data, p)
            yield dict(
                **condition, propagated=measured["propagated"],
                seed_plateau=measured["seed_plateau"], final_plateau=measured["final_plateau"],
                peak_shaft_response_mv=measured["peak_shaft_response_mv"],
                observation_end_ms=end, recovery_passed=True,
                nmda_onsets_ms=[head(0.5).nmda.onset for head in model.heads],
                ampa_onsets_ms=[head(0.5).ampa.onset for head in model.heads],
            )
    finally:
        for values in recordings.values():
            for vector in values if isinstance(values, list) else [values]:
                vector.play_remove()


def _sample(task):
    """Test all input orders for one set of synaptic and geometric properties."""
    definition, base, destination, code = task
    path = Path(destination) / (definition["name"] + ".json")
    placement = definition["placement"]
    head_girk, shaft_girk = GIRK_CONDUCTANCES[placement]
    p = replace(
        base, girk_head_ns=head_girk, girk_shaft_ns_per_um=shaft_girk,
    )
    settings = dict(parameters=asdict(p), sample=definition, model_code=code)
    saved = json.loads(path.read_text()) if path.exists() else dict(settings=settings, cases=[])
    if saved["settings"] != settings:
        raise ValueError("Saved sample settings differ from the requested settings.")
    completed = {case["name"] for case in saved["cases"]}
    remaining = [case for case in definition["orders"] if case["name"] not in completed]
    if remaining:
        model = build_dendrite(p, definition["factors"])
        h.celsius = p.temperature_c
        try:
            saved["model"] = _model_parameters(model)
            started = time.perf_counter()
            for case in _responses(model, remaining):
                saved["cases"].append(case)
                saved["simulation_seconds"] = saved.get("simulation_seconds", 0.0) + time.perf_counter() - started
                _write_json(path, saved)
                started = time.perf_counter()
        finally:
            model.close()
    return {**{key: value for key, value in definition.items() if key not in ("factors", "orders")},
            "cases": saved["cases"]}


def propagation_counts(rows):
    """Count propagation for ordered, swapped, shuffled and reversed inputs."""
    groups = {}
    for row in rows:
        families = FAMILIES if row["family"] == "BASELINE" else (row["family"],)
        for family in families:
            key = (row["placement"], family, row["sd"])
            groups.setdefault(key, []).append(row)
    counts = []
    for (placement, family, sd), maps in groups.items():
        cases = [case for sample in maps for case in sample["cases"]]
        for kind, per_map in (("ordered", 1), ("swap", 8), ("shuffle", SHUFFLES),
                              ("reverse", 1), ("shuffle_reverse", SHUFFLES + 1)):
            kinds = ("shuffle", "reverse") if kind == "shuffle_reverse" else (kind,)
            selected = [case for case in cases if case["kind"] in kinds]
            if len(selected) != len(maps) * per_map:
                raise ValueError("Propagation counts require all input sequences.")
            propagated = sum(case["propagated"] for case in selected)
            counts.append(dict(
                placement=placement, parameter=family,
                sd_percent=100 * sd, input_order=kind,
                propagated_cases=propagated, tested_cases=len(selected),
                percent_propagated=100 * propagated / len(selected),
            ))
    return counts


def run_variability(parameters, samples, destination, workers=1):
    """Run the variability study and save measurements and propagation percentages."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(model_hash().encode())
    digest.update((Path(__file__).parent / "variability_sampling.py").read_bytes())
    code = digest.hexdigest()
    manifest = dict(
        parameters=asdict(parameters), model_code=code, standard_deviations=list(SDS),
        distribution="mean-preserving lognormal", samples_per_nonzero_sd=SAMPLES,
        shuffles_per_sample=SHUFFLES, physical_maps=len(samples),
    )
    path = destination / "parameters.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Saved study settings differ from the requested settings.")
    _write_json(path, manifest)
    tasks = [(sample, parameters, str(destination), code) for sample in samples]
    rows = []
    started = time.monotonic()

    def completed(row):
        """Save run progress after a parameter map has finished."""
        rows.append(row)
        cases = sum(len(item["cases"]) for item in rows)
        elapsed = time.monotonic() - started
        _write_json(destination / "progress.json", dict(
            completed_maps=len(rows), total_maps=len(samples), completed_cases=cases,
            total_cases=sum(len(sample["orders"]) for sample in samples), elapsed_seconds=elapsed,
            workers=workers, latest_map=row["name"],
        ))
        print(f"{len(rows)}/{len(samples)} maps, {cases} cases, {elapsed:.1f} s", flush=True)

    if workers == 1:
        for task in tasks:
            completed(_sample(task))
    else:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_sample, task) for task in tasks]
            for future in as_completed(futures):
                completed(future.result())
    counts = propagation_counts(rows)
    with (destination / "counts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(counts[0]))
        writer.writeheader()
        writer.writerows(counts)
    return counts
