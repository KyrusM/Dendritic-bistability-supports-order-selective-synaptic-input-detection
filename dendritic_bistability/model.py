"""Cylindrical dendrites, spines and somatic loads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np
from neuron import h, load_mechanisms

h.load_file("stdrun.hoc")
CHANNELS = json.loads((Path(__file__).parent / "parameters.json").read_text())
_loaded = False


def load_channels():
    """Load the compiled channel mechanisms once per process."""
    global _loaded
    if not _loaded:
        if not load_mechanisms(str(Path(__file__).parent / "mechanisms")):
            raise RuntimeError("Run python -m dendritic_bistability.compile first.")
        _loaded = True


@dataclass(frozen=True)
class Parameters:
    """Specify geometry, conductances, input timing and solver settings."""

    # Dendrite, spines and soma.
    spacing_um: float = 10.0
    diameter_um: float = 0.3
    neck_resistance_gohm: float = 1.0
    neck_length_um: float = 2.0
    resistivity_ohm_cm: float = 400.0
    spine_resistivity_ohm_cm: float = 400.0
    head_length_um: float = 0.5
    head_diameter_um: float = 0.5
    compartments_per_interval: int = 12
    seed_count: int = 4
    sequence_count: int = 9
    buffer_count: int = 0
    soma: str = "passive"
    soma_gap_um: float = 10.0
    soma_size_um: float = 25.0
    soma_leak_multiplier: float = 5.0

    # NMDA and GIRK at reversal; AMPA at peak activation.
    nmda_ns: float = 2.4
    ampa_ns: float = 0.35
    seed_ampa_ns: float = 0.35
    girk_head_ns: float = 0.0
    girk_shaft_ns_per_um: float = 0.45

    # Background channels and passive membrane.
    background_multiplier: float = 1.0
    capacitance_uf_cm2: float = 1.45
    leak_s_cm2: float = 1 / 37500
    leak_reversal_mv: float = -65.0
    temperature_c: float = 32.0
    initial_voltage_mv: float = -65.0

    # Input timing and numerical settings.
    seed_time_ms: float = 1000.0
    interval_ms: float = 20.0
    record_ms: float = 0.2
    absolute_tolerance: float = 1e-6
    maximum_step_ms: float = 0.05
    stop_ms: float = 3000.0
    maximum_stop_ms: float = 12000.0

    # Post-plateau observation.
    observe_recovery: bool = True
    recovery_ms: float = 200.0
    recovery_tolerance_mv: float = 2.0

    seeded: bool = True


def set_channels(section, region, multiplier=1.0):
    """Apply Gao channel densities, ion settings and reversal potentials to a section."""
    for mechanism, values in CHANNELS[region]["mechanisms"].items():
        section.insert(mechanism)
        for name, value in values.items():
            setattr(section, name, value * multiplier if name in CHANNELS["densities"] else value)
    for name, value in CHANNELS["ions"].items():
        setattr(section, name, value)
    for ion, style in CHANNELS["ion_styles"].items():
        h.ion_style(
            ion + "_ion",
            style % 4,
            (style // 8) % 4,
            (style // 32) % 2,
            (style // 64) % 2,
            (style // 4) % 2,
            sec=section,
        )


def passive(section, p, multiplier=1.0):
    """Set axial resistivity, membrane capacitance and passive leak on a section."""
    section.Ra, section.cm = p.resistivity_ohm_cm, p.capacitance_uf_cm2
    section.insert("pas")
    section.g_pas = p.leak_s_cm2 * multiplier
    section.e_pas = p.leak_reversal_mv


def neck_diameter(resistance_gohm, length_um, resistivity_ohm_cm):
    """Return the neck diameter in micrometers for the specified full-neck axial resistance."""
    return math.sqrt(0.04 * resistivity_ohm_cm * length_um / (math.pi * 1000 * resistance_gohm))


class Dendrite:
    """Build an aligned cylindrical shaft with explicit spines and an optional soma."""

    def __init__(self, p, factors=None):
        """Place seed, sequence and buffer spines, then build their shaft, necks, heads and soma."""
        load_channels()
        self.parameters = p
        self.sections, self.heads, self.necks, self.synapses = [], [], [], []
        self.stimuli, self.handlers, self.events = [], [], []
        self.soma = None
        self.factors = factors or {}

        # Place spines and align shaft sections to their attachment sites.
        self.seed_positions = np.linspace(0.0, p.spacing_um, p.seed_count)
        gaps = p.spacing_um * np.asarray(self.factors.get("SPACING", np.ones(p.sequence_count - 1)))
        self.sequence_positions = np.r_[2 * p.spacing_um, 2 * p.spacing_um + np.cumsum(gaps)]
        self.buffer_positions = self.sequence_positions[-1] + p.spacing_um * np.arange(1, p.buffer_count + 1)
        self.positions = np.r_[self.seed_positions, self.sequence_positions, self.buffer_positions]
        self.length = self.sequence_positions[-1] + (
            p.soma_gap_um if p.soma != "none" else (p.buffer_count + 1) * p.spacing_um
        )
        self.boundaries = np.unique(np.r_[0.0, self.positions, self.length])
        self._shaft()
        self._spines()
        self._soma()
        for name, value in CHANNELS["globals"].items():
            setattr(h, name, value)

    def _factor(self, family, index):
        """Return a site-specific multiplier, or one for a site without a specified factor."""
        values = self.factors.get(family)
        return 1.0 if values is None or index >= len(values) else float(values[index])

    def _shaft(self):
        """Build shaft sections aligned to spine positions and assign channel densities."""
        p = self.parameters
        sequence_gaps = set(zip(self.sequence_positions[:-1], self.sequence_positions[1:]))
        for a, b in zip(self.boundaries[:-1], self.boundaries[1:]):
            section = h.Section(name="shaft")
            section.L, section.diam = b - a, p.diameter_um
            section.nseg = (
                p.compartments_per_interval
                if (a, b) in sequence_gaps
                else max(1, round(p.compartments_per_interval * (b - a) / p.spacing_um))
            )
            passive(section, p, p.background_multiplier)
            set_channels(section, "shaft", p.background_multiplier)
            section.insert("girk")
            for seg in section:
                # Convert shaft conductance per micrometer to a membrane density.
                nearest = int(np.argmin(abs(self.positions - (a + seg.x * section.L))))
                seg.girk.g_at_e = (
                    p.girk_shaft_ns_per_um * self._factor("GIRK", nearest) / (10 * math.pi * p.diameter_um)
                )
            if self.sections:
                self.sections[-1].connect(section(0), 1)
            self.sections.append(section)

    def shaft_at(self, position):
        """Return the shaft segment at a distance in micrometers from the dendritic tip."""
        i = int(np.searchsorted(self.boundaries, position, side="left") - 1)
        i = int(np.clip(i, 0, len(self.sections) - 1))
        return self.sections[i]((position - self.boundaries[i]) / self.sections[i].L)

    def _spines(self):
        """Attach explicit necks and heads and assign each head its NMDA, AMPA and GIRK conductances."""
        p = self.parameters
        for i, position in enumerate(self.positions):
            neck, head = h.Section(name="neck"), h.Section(name="head")
            for section in (neck, head):
                passive(section, p, p.background_multiplier)
                section.Ra = p.spine_resistivity_ohm_cm
                section.nseg = 1
            neck.L = p.neck_length_um
            neck.diam = neck_diameter(p.neck_resistance_gohm * self._factor("NECK", i), neck.L, neck.Ra)
            head.L, head.diam = p.head_length_um, p.head_diameter_um
            neck.connect(self.shaft_at(float(position)))
            head.connect(neck(1))
            for mechanism in ("girk", "ampa", "nmda"):
                head.insert(mechanism)
            # Convert each head conductance from nS to S/cm².
            scale = 10 * h.area(0.5, sec=head)
            head(0.5).girk.g_at_e = p.girk_head_ns * self._factor("GIRK", i) / scale
            is_seed = i < p.seed_count
            driven = i < p.seed_count + p.sequence_count
            head(0.5).nmda.g_at_e = p.nmda_ns * self._factor("NMDA", i) / scale if driven else 0.0
            head(0.5).ampa.g_peak = (
                (p.seed_ampa_ns if is_seed else p.ampa_ns) * self._factor("AMPA", i) / scale
                if driven
                else 0.0
            )
            self.necks.append(neck)
            self.heads.append(head)

    def _soma(self):
        """Attach the specified passive or active one-compartment soma."""
        p = self.parameters
        if p.soma == "none":
            return
        self.soma = h.Section(name="soma")
        self.soma.L = self.soma.diam = p.soma_size_um
        self.soma.nseg = 1
        passive(self.soma, p, p.soma_leak_multiplier)
        if p.soma == "active":
            set_channels(self.soma, "soma")
        self.sections[-1].connect(self.soma(0.5), 1)

    def set_inputs(self, order, times=None):
        """Schedule simultaneous seed inputs and assign the sequence inputs to their time slots."""
        p = self.parameters
        onsets = np.full(len(self.heads), 1e9)
        if p.seeded:
            onsets[: p.seed_count] = p.seed_time_ms
        times = (
            p.seed_time_ms + p.interval_ms * np.arange(1, p.sequence_count + 1)
            if times is None
            else np.asarray(times)
        )
        for site, time in zip(order, times):
            onsets[p.seed_count + site - 1] = time
        for head, onset in zip(self.heads, onsets):
            head(0.5).ampa.onset = float(onset)
            head(0.5).nmda.onset = float(onset)
        self.input_sites, self.input_times = np.asarray(order), times

    def current_pulse(self, time_ms, amplitude_na=0.5, duration_ms=2.0):
        """Add a somatic current pulse with the specified onset, amplitude and duration."""
        stimulus = h.IClamp(self.soma(0.5))
        stimulus.delay, stimulus.dur, stimulus.amp = time_ms, duration_ms, amplitude_na
        self.stimuli.append(stimulus)

    def repeat_glutamate(self, events):
        """Add repeated glutamate inputs at sequence spines, summing AMPA and capping NMDA activation."""
        grouped = {}
        for site, time in events:
            grouped.setdefault(site, []).append(time)
        for site, extra_times in grouped.items():
            head = self.heads[self.parameters.seed_count + site - 1]
            seg, area = head(0.5), 10 * h.area(0.5, sec=head)
            synapse = h.FrontEvents(seg)
            synapse.nmda_ns = seg.nmda.g_at_e * area
            synapse.ampa_ns = seg.ampa.g_peak * area
            times = [seg.ampa.onset, *extra_times]
            seg.ampa.g_peak = seg.nmda.g_at_e = 0.0
            connection = h.NetCon(None, synapse)
            connection.weight[0] = 1.0
            self.synapses.extend([synapse, connection])
            self.events.extend((connection, time) for time in times)

    def add_glutamate_spine(self, position_um, time_ms):
        """Attach an additional spine and deliver a glutamate input at its specified time."""
        p = self.parameters
        neck, head = h.Section(name="neck"), h.Section(name="head")
        for section in (neck, head):
            passive(section, p, p.background_multiplier)
            section.Ra, section.nseg = p.spine_resistivity_ohm_cm, 1
        neck.L = p.neck_length_um
        neck.diam = neck_diameter(p.neck_resistance_gohm, neck.L, neck.Ra)
        head.L, head.diam = p.head_length_um, p.head_diameter_um
        neck.connect(self.shaft_at(position_um))
        head.connect(neck(1))
        synapse = h.FrontEvents(head(0.5))
        synapse.nmda_ns, synapse.ampa_ns = p.nmda_ns, p.ampa_ns
        connection = h.NetCon(None, synapse)
        connection.weight[0] = 1.0
        self.necks.append(neck)
        self.heads.append(head)
        self.positions = np.r_[self.positions, position_um]
        self.synapses.extend([synapse, connection])
        self.events.append((connection, time_ms))

    def add_inhibition(self, position_um, time_ms):
        """Add a 0.5-nS GABA_A input on the shaft with 0.3-ms rise and 10-ms decay."""
        synapse = h.Exp2Syn(self.shaft_at(position_um))
        synapse.tau1, synapse.tau2, synapse.e = 0.3, 10.0, -75.0
        connection = h.NetCon(None, synapse)
        connection.weight[0] = 0.0005
        self.synapses.extend([synapse, connection])
        self.events.append((connection, time_ms))

    def close(self):
        """Release stimuli and synapses and delete the model sections."""
        self.handlers.clear()
        self.stimuli.clear()
        self.events.clear()
        self.synapses.clear()
        for section in [
            *self.heads,
            *self.necks,
            *reversed(self.sections),
            *([self.soma] if self.soma else []),
        ]:
            h.delete_section(sec=section)


def build_dendrite(parameters=Parameters(), factors=None):
    """Build a dendrite from model parameters and optional site-specific variability factors."""
    return Dendrite(parameters, factors)
