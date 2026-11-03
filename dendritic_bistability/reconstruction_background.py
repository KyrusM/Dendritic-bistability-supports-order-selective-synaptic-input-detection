"""Input resistance with lognormal synaptic background conductances.

Locations: Gemin et al. (2021), doi:10.1371/journal.pbio.3001375.
Rates: Rudolph and Destexhe (2003), doi:10.1523/JNEUROSCI.23-06-02466.2003.
AMPA: Beaulieu-Laroche and Harnett (2018), doi:10.1016/j.neuron.2017.11.016.
NMDA spread: Racca et al. (2000), doi:10.1523/JNEUROSCI.20-07-02512.2000.
GABA_A spread: Meye et al. (2013), doi:10.1038/mp.2012.145, Fig. 2c.
"""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from neuron import h

from .analysis import plateau_intervals, spike_times
from .reconstruction import ReconstructedBranch, SEEDS, SEQUENCE
from .reconstruction_electrical import CurrentSteps
from .reconstruction_sequences import record_voltage, save_json


TRIALS = (2, 3, 4, 5, 6)
CURRENTS = (-40, -20, 0, 20, 40)
ONSET, OFFSET, BACKGROUND_END = 1100.0, 2100.0, 2600.0
CONDUCTANCES = {
    "ampa": {"mean_ns": 2.1, "sd_ns": 2.0},
    "nmda": {"mean_ns": 1.05, "sd_ns": 1.05 * 0.67},
    "gaba": {"mean_ns": 0.5, "sd_ns": 0.1706810949943853},
}


def random_generator(identity):
    """Create a reproducible random stream for one synapse or event train."""
    seed = int(hashlib.sha256(identity.encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed)


def lognormal_conductance(channel, identity):
    """Draw a conductance from its arithmetic mean and standard deviation."""
    mean, sd = (CONDUCTANCES[channel][key] for key in ("mean_ns", "sd_ns"))
    variance = np.log1p((sd / mean) ** 2)
    return float(random_generator(identity).lognormal(np.log(mean) - variance / 2,
                                                      np.sqrt(variance)))


def sample_synapses(geometry):
    """Sample inhibitory locations and fixed conductances for all background contacts."""
    rng = random_generator("S7 inhibitory locations")
    eligible = [s["id"] for s in geometry["spines"] if s["attachment_arc_um"] >= 24]
    shaft = sorted(rng.uniform(24, geometry["length_um"],
                               round(0.21 * (geometry["length_um"] - 24))).tolist())
    heads = sorted(rng.choice(eligible, round(0.10 * len(eligible)), replace=False).tolist())
    glutamate = []
    for spine in geometry["spines"]:
        identity = spine["id"]
        selected = identity in SEEDS + SEQUENCE
        glutamate.append(dict(
            id=identity,
            ampa_peak_ns=1.4 if selected else lognormal_conductance(
                "ampa", f"S7 background AMPA {identity}"),
            nmda_at_reversal_ns=2.5 if selected else lognormal_conductance(
                "nmda", f"S7 background NMDA {identity}"),
        ))
    contacts = [dict(name=f"gaba_shaft_{i}", shaft_position_um=x)
                for i, x in enumerate(shaft)]
    contacts += [dict(name=f"gaba_spine_{identity}", spine_id=identity) for identity in heads]
    for contact in contacts:
        contact["peak_ns"] = lognormal_conductance(
            "gaba", f"S7 background GABA {contact['name']}")
    return dict(glutamate=glutamate, gaba=contacts)


def background_events(sample, trial):
    """Draw independent Poisson trains, shared across current amplitudes."""
    streams = {}
    targets = [f"glutamate_{s['id']}" for s in sample["glutamate"]]
    targets += [s["name"] for s in sample["gaba"]]
    for name in targets:
        rate = 0.5 if name.startswith("glutamate") else 2.75
        rng = random_generator(f"S7 sequence intermediate {trial} {name}")
        times, time = [], 0.0
        while True:
            time += rng.exponential(1000 / rate)
            if time >= BACKGROUND_END:
                break
            times.append(float(time))
        streams[name] = times
    return streams


class BackgroundBranch(ReconstructedBranch):
    """Add repeated glutamate inputs and GABA_A contacts to the reconstructed branch."""

    def __init__(self, geometry, sample):
        """Retain the branch's channels and replace single-input glutamate mechanisms."""
        super().__init__(geometry)
        self.receptors, self.connections = {}, {}
        for row in sample["glutamate"]:
            segment = self.spines[row["id"]][1](0.5)
            segment.ampa.g_peak = segment.nmda.g_at_e = 0
            synapse = h.FrontEvents(segment)
            synapse.ampa_ns, synapse.nmda_ns = row["ampa_peak_ns"], row["nmda_at_reversal_ns"]
            synapse.ta, synapse.cap_nmda = 0.5, 1
            self.add_receptor(f"glutamate_{row['id']}", synapse, 1)
        for row in sample["gaba"]:
            if "spine_id" in row:
                segment = self.spines[row["spine_id"]][1](0.5)
            else:
                segment = self.shaft_nodes[np.argmin(abs(self.positions - row["shaft_position_um"]))]
            synapse = h.Exp2Syn(segment)
            synapse.tau1, synapse.tau2, synapse.e = 0.3, 10, -75
            self.add_receptor(row["name"], synapse, row["peak_ns"] / 1000)

    def add_receptor(self, name, synapse, weight):
        """Retain a synaptic mechanism and its event connection."""
        connection = h.NetCon(None, synapse)
        connection.weight[0] = weight
        self.receptors[name], self.connections[name] = synapse, connection

    def schedule(self, streams):
        """Deliver the specified release times to each synapse."""
        for name, times in streams.items():
            for time in times:
                self.connections[name].event(time)

    def background_manifest(self):
        """Report instantiated cable properties, receptor parameters and contact locations."""
        result = self.manifest()
        result["glutamate_synapses"] = [dict(
            id=identity, ampa_peak_ns=self.receptors[f"glutamate_{identity}"].ampa_ns,
            nmda_at_reversal_ns=self.receptors[f"glutamate_{identity}"].nmda_ns,
            nmda_temporal_gate_cap=self.receptors[f"glutamate_{identity}"].cap_nmda,
        ) for identity in self.spines]
        result["gaba_synapses"] = [dict(
            name=name, section=synapse.get_segment().sec.name(),
            compartment=synapse.get_segment().x, rise_ms=synapse.tau1,
            decay_ms=synapse.tau2, reversal_mv=synapse.e,
            peak_ns=1000 * self.connections[name].weight[0],
        ) for name, synapse in self.receptors.items() if name.startswith("gaba")]
        return result

    def close(self):
        """Release the event connections, receptors and cable sections."""
        self.connections.clear()
        self.receptors.clear()
        super().close()


def resting_voltages(model):
    """Measure the quiet reference voltages used to check recovery."""
    time, nodes = record_voltage(model.nodes, 2)
    h.finitialize(-65)
    model.cv.solve(1000)
    tail = np.asarray(time) >= 900
    if np.ptp(np.asarray(nodes)[:, tail], axis=1).max() > 0.05:
        raise RuntimeError("The branch did not reach a quiet resting state.")
    return np.array([node.v for node in model.nodes])


def background_step(task):
    """Apply one current step with a fixed background realization and check recovery."""
    geometry, sample, streams, trial, current, reference = task
    model = BackgroundBranch(geometry, sample)
    try:
        model.cv.maxstep(0.25)
        clamp = h.IClamp(model.probe)
        clamp.delay, clamp.dur, clamp.amp = 0, 1e9, 0
        time, traces = record_voltage([model.probe, model.soma(0.5)], 0.2)
        global_time, nodes = record_voltage(model.nodes, 2)
        h.finitialize(-65)
        model.schedule(streams)
        model.cv.solve(ONSET)
        clamp.amp = current / 1000
        model.cv.re_init()
        model.cv.solve(OFFSET)
        clamp.amp = 0
        model.cv.re_init()
        endpoint = 4600
        while True:
            model.cv.solve(endpoint)
            t, voltage = np.asarray(time), np.asarray(traces)
            gt, u = np.asarray(global_time), np.asarray(nodes)
            tail = gt >= endpoint - 200
            error = float(np.max(abs(u[:, tail] - reference[:, None])))
            spikes = spike_times(t, voltage[1])
            recovered = bool(error <= 0.5 and u[:, tail].max() < -30
                             and not np.any(spikes >= endpoint - 200))
            if recovered or endpoint >= 16600:
                break
            endpoint += 2000
        before = (t >= ONSET - 200) & (t < ONSET)
        steady = (t >= OFFSET - 200) & (t < OFFSET)
        baseline_spikes = spikes[(spikes >= ONSET - 100) & (spikes < ONSET)]
        pulse_spikes = spikes[(spikes >= ONSET) & (spikes < OFFSET)]
        step = t < OFFSET
        plateaus = plateau_intervals(t[step], voltage[0, step],
                                     start=ONSET + 4, threshold=-30, duration=10)
        valid = not len(baseline_spikes) and not len(pulse_spikes) and not plateaus
        return dict(
            trial=trial, current_pa=float(current),
            baseline_local_mv=float(voltage[0, before].mean()),
            steady_local_mv=float(voltage[0, steady].mean()),
            valid_subthreshold=bool(valid), recovery_passed=recovered,
            recovery_error_mv=error, endpoint_ms=endpoint,
        )
    finally:
        model.close()


def fit_resistance(rows, current_key="current_pa"):
    """Fit steady voltage against current with a free intercept; return resistance in MΩ."""
    current = np.array([row[current_key] for row in rows])
    voltage = np.array([row["steady_local_mv"] for row in rows])
    slope, intercept = np.polyfit(current, voltage, 1)
    residual = voltage - slope * current - intercept
    return dict(resistance_mohm=float(1000 * slope), intercept_mv=float(intercept),
                r_squared=float(1 - np.sum(residual ** 2) / np.sum((voltage - voltage.mean()) ** 2)))


def run_background(geometry, folder, workers=1):
    """Measure quiet and background input resistance across five event realizations."""
    geometry = Path(geometry)
    sample = sample_synapses(json.loads(geometry.read_text()))
    events = {trial: background_events(sample, trial) for trial in TRIALS}
    model = BackgroundBranch(geometry, sample)
    try:
        model.cv.maxstep(0.25)
        reference = resting_voltages(model)
        manifest = model.background_manifest()
    finally:
        model.close()
    quiet = CurrentSteps(geometry)
    try:
        quiet.settle()
        quiet_rows = [quiet.step(current) for current in CURRENTS]
    finally:
        quiet.close()
    if not all(row["valid_subthreshold"] for row in quiet_rows):
        raise RuntimeError("A quiet current step did not remain subthreshold and recover.")
    baseline = fit_resistance(quiet_rows, "current_pA")
    destination = Path(folder) / "background"
    save_json(destination / "model.json", manifest)
    save_json(destination / "synapses.json", sample)
    save_json(destination / "protocol.json", dict(
        distribution="lognormal", conductances=CONDUCTANCES, trials=list(TRIALS),
        currents_pa=list(CURRENTS), onset_ms=ONSET, duration_ms=OFFSET - ONSET,
        baseline_average_ms=200, steady_average_ms=200, background_end_ms=BACKGROUND_END,
        glutamate_hz=0.5, gaba_a_hz=2.75, events_shared_across_currents=True,
        shaft_gaba_density_per_um=0.21, head_gaba_fraction=0.10,
        minimum_gaba_distance_um=24, initial_voltage_mv=-65,
        maximum_step_ms=0.25, record_ms=0.2, compartment_check_ms=2,
        recovery_window_ms=200, recovery_tolerance_mv=0.5,
        minimum_endpoint_ms=4600, maximum_endpoint_ms=16600,
    ))
    for trial, streams in events.items():
        save_json(destination / "events" / f"{trial}.json", streams)
    tasks = [(geometry, sample, events[trial], trial, current, reference)
             for trial in TRIALS for current in CURRENTS]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        rows = list(pool.map(background_step, tasks))
    save_json(destination / "measurements.json", dict(quiet=quiet_rows, background=rows))
    if not all(row["valid_subthreshold"] and row["recovery_passed"] for row in rows):
        raise RuntimeError("A background current step did not remain subthreshold and recover.")
    fits = [dict(trial=trial, **fit_resistance([r for r in rows if r["trial"] == trial]))
            for trial in TRIALS]
    for fit in fits:
        fit["change_percent"] = 100 * (fit["resistance_mohm"] / baseline["resistance_mohm"] - 1)
    save_json(destination / "summary.json", dict(
        quiet=baseline, background=fits,
        largest_absolute_change_percent=max(abs(f["change_percent"]) for f in fits),
    ))
