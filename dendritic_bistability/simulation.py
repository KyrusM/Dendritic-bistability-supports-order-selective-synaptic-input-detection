"""Input sequences and observation windows."""

from dataclasses import asdict, replace
import time

import numpy as np
from neuron import h

from .analysis import measure, recovered
from .model import SweepDendrite, build_dendrite


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


def plateau_intervals(times, mask):
    """Return intervals with at least 10 ms of continuous samples above threshold."""
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return [[float(times[a]), float(times[b])] for a, b in zip(starts, ends)
            if times[b] - times[a] >= 10 - 1e-8]


def seed_ready(trace):
    """Check that the seed initiates a plateau before sequence onset."""
    times, voltage = trace["time_ms"], trace["head_mv"]
    return bool(plateau_intervals(times, (times >= 104) & (times < 120) & (voltage[4] >= -30))
                and voltage[4, np.searchsorted(times, 120) - 1] >= -30)


def intermediate_ready(trace):
    """Check the third sequence spine plateaus before the fourth input."""
    times = trace["time_ms"]
    return bool(trace["head_mv"][7, np.searchsorted(times, 180) - 1] >= -30)


def score_propagation(trace):
    """Measure the final spine's sustained plateau after its input."""
    times, voltage = trace["time_ms"], trace["head_mv"]
    intervals = plateau_intervals(times, (times >= 224) & (times <= 300) & (voltage[10] >= -30))
    final = bool(intervals) if times[-1] >= 224 else None
    return dict(final_plateau=final, propagated=bool(final),
                final_longest_plateau_ms=max((b - a for a, b in intervals), default=0),
                observation_end_ms=float(times[-1]))


def join_recordings(prefix, tail):
    """Join recordings at a shared initialization time without repeating its sample."""
    if not np.isclose(prefix["time_ms"][-1], tail["time_ms"][0], atol=1e-8):
        raise RuntimeError("Shared recordings do not meet at the same time.")
    return dict(time_ms=np.r_[prefix["time_ms"], tail["time_ms"][1:]],
                head_mv=np.concatenate([prefix["head_mv"], tail["head_mv"][:, 1:]], axis=1),
                shaft_mv=np.concatenate([prefix["shaft_mv"], tail["shaft_mv"][:, 1:]], axis=1))


class PropagationTrial:
    """Integrate one fixed dendrite and reuse its seed and pre-swap states."""

    def __init__(self, specification, initial_ampa_ns):
        """Build the model, record spine sites every 0.25 ms and initialize CVode."""
        self.model = SweepDendrite(**specification)
        self.states = []
        self.time = h.Vector().record(h._ref_t, 0.25)
        self.heads = [h.Vector().record(head(0.5)._ref_v, 0.25) for head in self.model.heads]
        self.shaft = [h.Vector().record(self.model.shaft_at(float(x))._ref_v, 0.25)
                      for x in self.model.positions]
        self.model.set_inputs(initial_ampa_ns, "ordered")
        h.celsius = 32
        h.cvode_active(1)
        h.cvode.atol(1e-7)
        h.cvode.rtol(0)
        h.cvode.maxstep(1)
        h.finitialize(-87)
        for onset in (100, 120, 140, 160, 180, 200, 220):
            h.cvode.event(onset, h.cvode.re_init)

    def advance(self, stop_ms):
        """Advance the adaptive solver to the requested observation time."""
        h.continuerun(float(stop_ms))

    def trace(self):
        """Read the temporary spine-head and shaft recordings."""
        return dict(time_ms=np.asarray(self.time).copy(),
                    head_mv=np.array([np.asarray(vector).copy() for vector in self.heads]),
                    shaft_mv=np.array([np.asarray(vector).copy() for vector in self.shaft]))

    def snapshot(self):
        """Save membrane voltages, channel states and pending input events."""
        state = h.SaveState()
        state.save()
        self.states.append(state)
        return state

    def restore(self, state, ampa_ns, order):
        """Restore a shared state and set the conductance and input order for its continuation."""
        self.model.set_inputs(ampa_ns, order)
        state.restore()
        h.cvode.re_init()
        h.frecord_init()

    def branch(self, state, prefix, ampa_ns, order, early=True):
        """Continue one order until a final plateau is established or 300 ms elapses."""
        self.restore(state, ampa_ns, order)
        if early:
            self.advance(234)
            while not plateau_intervals(self.time.as_numpy(),
                                        (self.time.as_numpy() >= 224) & (self.heads[10].as_numpy() >= -30)) and h.t < 300 - 1e-8:
                self.advance(min(300, h.t + 2))
        else:
            self.advance(300)
        return score_propagation(join_recordings(prefix, self.trace()))

    def close(self):
        """Release recordings and model sections after the bundle finishes."""
        self.states.clear()
        for vector in [self.time, *self.heads, *self.shaft]:
            vector.play_remove()
        self.model.close()


def evaluate_propagation_bundle(condition, spacing, imbalance, requests):
    """Evaluate AMPA values at one spacing and imbalance with common seed initialization."""
    specification = dict(placement=condition["placement"], spacing_um=float(spacing),
                         imbalance=float(imbalance), girk=float(condition["girk"]),
                         neck_length_um=float(condition.get("neck_length_um", 1)),
                         spine_resistivity_ohm_cm=float(condition.get("spine_resistivity_ohm_cm", 100)))
    trial = PropagationTrial(specification, requests[0]["ampa_ns"])
    results = {}
    try:
        trial.advance(119.75)
        prefix, seed_state = trial.trace(), None
        seed_score = dict(seed_ready=seed_ready(prefix), intermediate_ready=None,
                          final_plateau=None, propagated=False, final_longest_plateau_ms=0,
                          observation_end_ms=float(prefix["time_ms"][-1]))
        extension = bool(condition.get("extension", False))
        if seed_score["seed_ready"]:
            seed_state = trial.snapshot()
        for request in requests:
            started = time.perf_counter()
            ampa, orders, paired = float(request["ampa_ns"]), {}, True
            if not seed_score["seed_ready"]:
                orders = {order: dict(seed_score, evaluated=True) for order in ("ordered", "swapped")}
            else:
                trial.restore(seed_state, ampa, "ordered")
                trial.advance(179.75)
                common = join_recordings(prefix, trial.trace())
                readiness = dict(seed_score, intermediate_ready=intermediate_ready(common),
                                 observation_end_ms=float(common["time_ms"][-1]))
                if not readiness["intermediate_ready"]:
                    orders = {order: dict(readiness, evaluated=True) for order in ("ordered", "swapped")}
                elif extension:
                    for order in ("ordered", "swapped"):
                        orders[order] = trial.branch(seed_state, prefix, ampa, order, early=False)
                        orders[order].update(seed_ready=True, intermediate_ready=True, evaluated=True)
                else:
                    state = trial.snapshot()
                    first = request.get("first_order", "ordered")
                    second = "swapped" if first == "ordered" else "ordered"
                    orders[first] = dict(trial.branch(state, common, ampa, first),
                                         seed_ready=True, intermediate_ready=True, evaluated=True)
                    success = orders[first]["propagated"]
                    infer = not request.get("force_pair", False) and (
                        first == "ordered" and not success or first == "swapped" and success
                    )
                    if infer:
                        orders[second] = dict(propagated=success, evaluated=False)
                        paired = False
                    else:
                        orders[second] = dict(trial.branch(state, common, ampa, second),
                                              seed_ready=True, intermediate_ready=True, evaluated=True)
                    trial.states.remove(state)
            category = int(orders["ordered"]["propagated"]) + 2 * int(orders["swapped"]["propagated"])
            if category == 2:
                raise RuntimeError("Swapped propagation occurred without ordered propagation.")
            results[int(request["ampa_index"])] = dict(
                ampa_index=int(request["ampa_index"]), ampa_ns=ampa, classification=category,
                both_orders_evaluated=paired, orders=orders, seconds=time.perf_counter() - started,
            )
        return results
    finally:
        trial.close()
