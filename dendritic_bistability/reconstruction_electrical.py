"""Current-step input-resistance measurements at the branch midpoint."""

import numpy as np
from neuron import h

from .reconstruction import ReconstructedBranch
from .reconstruction_sequences import record_voltage, save_json


class CurrentSteps:
    """Measure steady responses to independent 1-s current steps at the branch midpoint."""

    def __init__(self, geometry):
        """Build the unstimulated branch and place a current electrode at its midpoint."""
        self.model = ReconstructedBranch(geometry)
        self.probe, self.cv = self.model.probe, self.model.cv
        self.cv.maxstep(0.25)
        self.clamp = h.IClamp(self.probe)
        self.clamp.delay, self.clamp.dur, self.clamp.amp = 0.0, 1e9, 0.0
        self.detector = h.NetCon(self.model.soma(0.5)._ref_v, None, sec=self.model.soma)
        self.detector.threshold = 0.0
        self.spikes = h.Vector()
        self.detector.record(self.spikes)

    def settle(self):
        """Save a quiet resting state after all compartments settle."""
        time, nodes = record_voltage(self.model.nodes, 2.0)
        h.finitialize(-65.0)
        self.settle_end = 1000.0
        while True:
            self.cv.solve(self.settle_end)
            tail = np.asarray(time) >= self.settle_end - 100
            span = np.ptp(np.asarray(nodes)[:, tail], axis=1).max()
            if span <= 0.05 and not len(self.spikes):
                break
            if self.settle_end >= 5000:
                raise RuntimeError("The branch did not reach a quiet resting state.")
            self.settle_end += 1000
        self.baseline = np.array([s.v for s in self.model.nodes])
        self.state = h.SaveState()
        self.state.save()

    def step(self, current):
        """Return finite-step resistance, steady voltage and subthreshold/recovery checks."""
        self.clamp.amp = 0.0
        h.finitialize(-65.0)
        self.state.restore(1)
        self.cv.re_init()
        self.spikes.resize(0)
        time, traces = record_voltage([self.probe, self.model.soma(0.5)], 0.2)
        global_time, nodes = record_voltage(self.model.nodes, 2.0)
        h.frecord_init()
        onset, offset = self.settle_end + 200, self.settle_end + 1200
        self.cv.solve(onset)
        self.clamp.amp = current / 1000
        self.cv.re_init()
        self.cv.solve(offset)
        self.clamp.amp = 0.0
        self.cv.re_init()
        endpoint = offset + 500
        while True:
            self.cv.solve(endpoint)
            gt, voltage = np.asarray(global_time), np.asarray(nodes)
            tail = gt >= endpoint - 200
            error = float(np.max(abs(voltage[:, tail] - self.baseline[:, None])))
            recovered = (
                error <= 0.5
                and voltage[:, tail].max() < -30
                and not np.any(self.spikes.as_numpy() >= endpoint - 200)
            )
            if recovered or endpoint >= offset + 10000:
                break
            endpoint += 1000
        t, u = np.asarray(time), np.asarray(traces)
        before, steady = (t >= onset - 200) & (t < onset), (t >= offset - 200) & (t < offset)
        baseline, final = u[:, before].mean(axis=1), u[:, steady].mean(axis=1)
        span = float(np.ptp(u[:, steady], axis=1).max())
        during = self.spikes.as_numpy()
        during = during[(during >= onset) & (during < offset)]
        regenerative = voltage[:, (gt >= onset) & (gt < offset)].max() >= -30
        valid = not len(during) and span <= 0.2 and not regenerative and recovered
        return dict(
            current_pA=float(current),
            onset_ms=float(onset),
            offset_ms=float(offset),
            baseline_local_mv=float(baseline[0]),
            steady_local_mv=float(final[0]),
            local_R_MOhm=float(1000 * (final[0] - baseline[0]) / current) if current else None,
            valid_subthreshold=bool(valid),
            recovery_passed=bool(recovered),
            recovery_max_error_mv=error,
            steady_tail_range_mv=span,
        )

    def close(self):
        """Remove the electrode and reconstructed branch."""
        self.model.close()


def resistance_curve(geometry):
    """Measure the ten current steps used for the voltage-current fit."""
    assay = CurrentSteps(geometry)
    try:
        assay.settle()
        rows = [assay.step(current) for current in range(-40, 141, 20)]
        if not all(row["valid_subthreshold"] for row in rows):
            raise RuntimeError("A fitted current step did not remain subthreshold and recover.")
        return rows
    finally:
        assay.close()


def run_electrical(geometry, folder):
    """Save the current-step protocol and steady voltage responses."""
    save_json(
        folder / "electrical_protocol.json",
        dict(
            electrode="existing shaft compartment nearest branch midpoint",
            glutamate_inputs=False,
            initial_voltage_mv=-65,
            current_clamp=dict(
                currents_pA=list(range(-40, 141, 20)),
                step_duration_ms=1000,
                pre_step_ms=200,
                baseline_average_ms=200,
                steady_average_ms=200,
                maximum_step_ms=0.25,
                record_ms=0.2,
                steady_tail_range_mv=0.2,
                minimum_recovery_ms=500,
                recovery_window_ms=200,
                recovery_tolerance_mv=0.5,
            ),
        ),
    )
    save_json(folder / "resistance.json", resistance_curve(geometry))
