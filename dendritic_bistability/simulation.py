"""Input sequences and observation windows."""

from dataclasses import asdict, replace

import numpy as np
from neuron import h

from .analysis import measure, recovered
from .model import build_dendrite


def run_sequence(
    parameters, order, *, pulse_ms=None, input_times=None, factors=None,
    background=None, record_surface=True,
):
    """Deliver one sequence; record the voltage surface or seed and sequence sites."""
    p = parameters
    if p.observe_recovery and not record_surface:
        raise ValueError("Recovery requires full voltage recordings.")
    model = build_dendrite(p, factors)
    model.set_inputs(order, input_times)
    if background:
        model.repeat_glutamate(background.get("sequence", []))
        for position, time in background.get("spines", []):
            model.add_glutamate_spine(position, time)
        for position, time in background.get("inhibition", []):
            model.add_inhibition(position, time)
    if pulse_ms is not None:
        model.current_pulse(pulse_ms)

    # Record spine-head, shaft and somatic voltages.
    vectors = []

    def record(reference):
        """Record a NEURON variable at the specified sampling interval."""
        vector = h.Vector().record(reference, p.record_ms)
        vectors.append(vector)
        return vector

    time = record(h._ref_t)
    count = len(model.heads) if record_surface else p.seed_count + p.sequence_count
    spine_positions = model.positions[:count]
    heads = [record(head(0.5)._ref_v) for head in model.heads[:count]]
    site_shaft = [record(model.shaft_at(float(x))._ref_v) for x in spine_positions]
    if record_surface:
        nodes = {
            float(a + seg.x * section.L): seg
            for a, section in zip(model.boundaries[:-1], model.sections)
            for seg in section.allseg()
        }
        positions = np.array(sorted(nodes))
        shaft = [record(nodes[x]._ref_v) for x in positions]
    else:
        positions, shaft = spine_positions, site_shaft
    soma = record(model.soma(0.5)._ref_v) if model.soma is not None else None

    # Initialize the adaptive solver and schedule input events.
    h.celsius = p.temperature_c
    solver = h.CVode()
    solver.active(1)
    solver.atol(p.absolute_tolerance)
    solver.rtol(0)
    solver.maxstep(p.maximum_step_ms)
    h.finitialize(p.initial_voltage_mv)
    for connection, at in model.events:
        connection.event(float(at))
    events = [*model.input_times, *([p.seed_time_ms] if p.seeded else [])]
    if pulse_ms is not None:
        events += [pulse_ms, pulse_ms + 2.0]
    events += [time for _, time in model.events]
    for onset in sorted(set(events)):
        solver.event(float(onset), solver.re_init)

    # Stop at the assay endpoint or continue through voltage recovery.
    stop = max(p.stop_ms, float(max(events)) + p.recovery_ms) if p.observe_recovery else p.stop_ms
    try:
        if stop < max(events):
            raise ValueError("The observation window must include all inputs.")
        while True:
            h.continuerun(stop)
            site_voltages = np.array(site_shaft)
            result = dict(
                time_ms=np.asarray(time).copy(),
                head_mv=np.array(heads),
                shaft_mv=np.array(shaft) if record_surface else site_voltages,
                shaft_positions_um=positions,
                site_shaft_mv=site_voltages,
                spine_positions_um=spine_positions,
                input_sites=model.input_sites,
                input_times_ms=model.input_times,
                parameters=asdict(p),
                pulse_ms=pulse_ms,
                stimulus_start_ms=float(min(events)),
                observation_end_ms=float(stop),
            )
            if soma is not None:
                result["soma_mv"] = np.asarray(soma).copy()
            result["measurements"] = measure(result, p)
            if not p.observe_recovery:
                return result
            if recovered(result, p):
                result["recovered"] = True
                return result
            if stop >= p.maximum_stop_ms:
                raise RuntimeError("Membrane voltages did not recover within the observation window.")
            stop = min(2 * stop, p.maximum_stop_ms)
    finally:
        for vector in vectors:
            vector.play_remove()
        model.close()


def compare_sequences(parameters, orders, **kwargs):
    """Run each input order through a common observation endpoint."""
    results = {name: run_sequence(parameters, order, **kwargs) for name, order in orders.items()}
    stop = max(result["observation_end_ms"] for result in results.values())
    for name, result in results.items():
        if result["observation_end_ms"] < stop:
            results[name] = run_sequence(replace(parameters, stop_ms=stop), orders[name], **kwargs)
    return results
