"""Sequence-evoked spiking in the reconstructed branch."""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from neuron import h

from .reconstruction import (
    ReconstructedBranch, SEEDS, SEQUENCE,
    AMPA_PEAK_NS, NMDA_AT_REVERSAL_NS, GIRK_AT_REVERSAL_NS,
)


def save_json(path, value):
    """Save numerical measurements or an instantiated model description."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def record_voltage(segments, interval):
    """Record time and segment voltages at a fixed interval."""
    return (
        h.Vector().record(h._ref_t, interval),
        [h.Vector().record(segment._ref_v, interval) for segment in segments],
    )


def run_sequence(geometry_path, order, surface=False):
    """Deliver four seed inputs and seven sequence inputs; count spikes through recovery."""
    model = ReconstructedBranch(geometry_path)
    cv = model.cv
    events = {identity: 1000.0 for identity in SEEDS}
    events.update({identity: 1020.0 + 20 * j for j, identity in enumerate(order)})
    for identity, onset in events.items():
        head = model.spines[identity][1](0.5)
        head.ampa.onset = head.nmda.onset = onset
    detector = h.NetCon(model.soma(0.5)._ref_v, None, sec=model.soma)
    detector.threshold = 0.0
    spikes = h.Vector()
    detector.record(spikes)
    h.finitialize(-65.0)
    cv.solve(999.0)
    ids = SEEDS + SEQUENCE
    time, heads = record_voltage([model.spines[i][1](0.5) for i in ids], 0.2)
    shafts = [h.Vector().record(model.shaft_at(i)._ref_v, 0.2) for i in ids]
    soma = h.Vector().record(model.soma(0.5)._ref_v, 0.2)
    global_time, nodes = record_voltage(model.nodes, 2.0)
    surface_time, shaft_surface = record_voltage(model.shaft_nodes, 1.0) if surface else (None, [])
    h.frecord_init()
    callbacks = []
    for onset in sorted(set(events.values())):
        callbacks.append(lambda: cv.re_init())
        cv.event(onset, callbacks[-1])
    callbacks.append(lambda: cv.maxstep(1.0))
    cv.event(1300.0, callbacks[-1])
    endpoint = 2700.0
    while True:
        cv.solve(endpoint)
        gt, voltage = np.asarray(global_time), np.asarray(nodes)
        baseline = voltage[:, gt < 1000][:, -1]
        tail = gt >= endpoint - 200
        error = float(np.max(abs(voltage[:, tail] - baseline[:, None])))
        recovered = error <= 5 and voltage[:, tail].max() < -30
        if recovered or endpoint >= 11500:
            break
        endpoint += 800
    spike_times = spikes.as_numpy().copy()
    row = dict(
        order=list(order),
        soma_spike_count=int(np.sum(spike_times >= 1020)),
        sequence_spikes_ms=spike_times[spike_times >= 1020].tolist(),
        seed_spikes_ms=spike_times[(spike_times >= 1000) & (spike_times < 1020)].tolist(),
        settling_spikes_ms=spike_times[spike_times < 1000].tolist(),
        recovery_passed=bool(recovered and not np.any(spike_times >= endpoint - 200)),
        recovery_max_error_mv=error,
        end_ms=endpoint,
        actual_synapses=[dict(id=i, onset_ms=t) for i, t in events.items()],
    )
    data = (
        dict(
            time_ms=np.asarray(time).copy(),
            ids=np.array(ids),
            head_v_mv=np.asarray(heads),
            shaft_v_mv=np.asarray(shafts),
            soma_v_mv=np.asarray(soma).copy(),
            surface_arc_um=model.positions,
            surface_time_ms=np.asarray(surface_time).copy() if surface else np.array([]),
            surface_v_mv=np.asarray(shaft_surface),
        )
        if surface
        else None
    )
    manifest = model.manifest()
    model.close()
    if not row["recovery_passed"]:
        raise RuntimeError("The sequence response did not recover within the observation window.")
    return row, data, manifest


def sequence_job(job):
    """Save one order's spike counts and, for a plotted example, its voltages."""
    geometry, folder, name, order = job
    row, data, manifest = run_sequence(geometry, order, name in ("ordered", "swap_3_4"))
    row["name"] = name
    save_json(folder / "orders" / (name + ".json"), row)
    if data is not None:
        np.savez_compressed(folder / "orders" / (name + ".npz"), **data)
    if name == "ordered":
        save_json(folder / "model.json", manifest)
    print(f"{name}: {row['soma_spike_count']} somatic spikes", flush=True)
    return row


def run_sequences(geometry, orders_path, folder, workers=1):
    """Simulate ordered inputs, six adjacent swaps, twenty shuffles and reversed inputs."""
    orders = json.loads(Path(orders_path).read_text())
    folder = Path(folder)
    (folder / "orders").mkdir(parents=True, exist_ok=True)
    save_json(
        folder / "sequence_protocol.json",
        dict(
            initial_voltage_mv=-65,
            seed_ids=SEEDS,
            sequence_ids=SEQUENCE,
            seed_time_ms=1000,
            input_times_ms=[1020 + 20 * j for j in range(len(SEQUENCE))],
            ampa_peak_ns=AMPA_PEAK_NS,
            nmda_at_reversal_ns=NMDA_AT_REVERSAL_NS,
            girk_at_reversal_ns=GIRK_AT_REVERSAL_NS,
            maximum_step_ms=0.05,
            maximum_step_after_1300_ms=1.0,
            initial_endpoint_ms=2700,
            recovery_duration_ms=200,
            recovery_tolerance_mv=5,
            recovery_voltage_ceiling_mv=-30,
            recovery_extension_ms=800,
            maximum_endpoint_ms=11500,
            somatic_spike_threshold_mv=0,
            spike_count_start_ms=1020,
        ),
    )
    jobs = [(geometry, folder, name, order) for name, order in orders]
    if workers == 1:
        rows = [sequence_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            rows = list(pool.map(sequence_job, jobs))
    rows.sort(key=lambda r: (r["name"] != "ordered", r["name"]))
    save_json(folder / "sequences.json", rows)
