"""Plateau propagation, voltage responses and somatic spikes."""

import numpy as np


def plateau_intervals(time, voltage, start=0.0, threshold=-30.0, duration=10.0):
    """Return intervals that remain above the voltage threshold for the required duration."""
    selected = (time >= start) & (voltage >= threshold)
    edges = np.diff(np.r_[False, selected, False].astype(int))
    intervals = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1):
        if time[b] - time[a] >= duration:
            intervals.append([float(time[a]), float(time[b])])
    return intervals


def spike_times(time, voltage):
    """Interpolate the times when somatic voltage crosses 0 mV upward."""
    i = np.flatnonzero((voltage[:-1] < 0) & (voltage[1:] >= 0))
    return time[i] - voltage[i] * np.diff(time)[i] / np.diff(voltage)[i]


def measure(result, p):
    """Measure seed and final-spine plateaus, the final-site shaft response and somatic spikes."""
    t, v = result["time_ms"], result["head_mv"]
    first = float(min(result["input_times_ms"]))
    start = result.get("stimulus_start_ms", min(p.seed_time_ms, first) if p.seeded else first)
    baseline = np.searchsorted(t, start) - 1

    # Check the seed plateau immediately before the first sequence input.
    seed = p.seed_count - 1
    pre = np.searchsorted(t, first) - 1
    seed_plateaus = plateau_intervals(t[t < first], v[seed, t < first], p.seed_time_ms + 4.0)
    seed_ready = not p.seeded or bool(seed_plateaus and v[seed, pre] >= -30.0)

    # Detect plateaus after each input, excluding its initial 4-ms transient.
    onsets = dict(zip(result["input_sites"].tolist(), result["input_times_ms"].tolist()))
    plateaus = [
        plateau_intervals(t, v[p.seed_count + i], onsets[i + 1] + 4.0) for i in range(p.sequence_count)
    ]

    # Also detect plateaus at any time for the variability comparison.
    all_periods = [plateau_intervals(t, v[p.seed_count + i], start) for i in range(p.sequence_count)]
    shaft = result["site_shaft_mv"][p.seed_count + p.sequence_count - 1]
    spikes = spike_times(t, result["soma_mv"]) if "soma_mv" in result else np.array([])
    return dict(
        seed_plateau=seed_ready,
        final_plateau=bool(plateaus[-1]),
        propagated=bool(seed_ready and plateaus[-1]),
        sequence_plateaus=plateaus,
        all_sequence_spines_plateau=all(plateaus),
        final_plateau_any_time=bool(all_periods[-1]),
        all_sequence_spines_plateau_any_time=all(all_periods),
        peak_shaft_response_mv=float(shaft[t >= first].max() - shaft[baseline]),
        somatic_spikes_ms=spikes[spikes >= p.seed_time_ms + p.interval_ms].tolist(),
    )


def recovered(result, p):
    """Check that all recorded voltages return near baseline throughout the final recovery period."""
    t = result["time_ms"]
    baseline = np.searchsorted(t, result.get("stimulus_start_ms", p.seed_time_ms)) - 1
    tail = t >= t[-1] - p.recovery_ms
    voltages = np.vstack([result["head_mv"], result["shaft_mv"]])
    if "soma_mv" in result:
        voltages = np.vstack([voltages, result["soma_mv"]])
    return bool(
        np.max(abs(voltages[:, tail] - voltages[:, baseline, None])) <= p.recovery_tolerance_mv
        and np.max(result["head_mv"][:, tail]) < -30.0
    )
