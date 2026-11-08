"""An EM-derived dendritic branch with explicit spines and an active soma.

Geometry: MICrONS Consortium (2025), doi:10.1038/s41586-025-08790-w.
Background channels: Gao et al. (2021), ModelDB 249705.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from neuron import h

from .model import CHANNELS, Parameters, load_channels, passive, set_channels


SEEDS = [121, 118, 114, 108]
SEQUENCE = [101, 87, 72, 63, 45, 31, 7]
AMPA_PEAK_NS = 1.4
NMDA_AT_REVERSAL_NS = 2.5
GIRK_AT_REVERSAL_NS = 4.9


class ReconstructedBranch:
    """Build the S7 branch, its 124 spines and its one-compartment soma."""

    def __init__(self, geometry_path):
        """Load the cable geometry and assign membrane and channel parameters."""
        load_channels()
        self.geometry = json.loads(Path(geometry_path).read_text())
        self.geometry_hash = hashlib.sha256(Path(geometry_path).read_bytes()).hexdigest()
        self.sections, self.shaft, self.spines = [], [], {}
        self.parameters = Parameters()
        self.soma = self.make_section("soma", self.geometry["soma"])
        previous = self.soma
        for index, geometry in enumerate(self.geometry["shaft"]):
            section = self.make_section(f"shaft_{index}", geometry, previous)
            set_channels(section, "shaft")
            self.shaft.append(section)
            previous = section
        self.boundaries = np.r_[
            [q["arc_start_um"] for q in self.geometry["shaft"]],
            self.geometry["length_um"],
        ]
        self.attachments = {}
        for geometry in self.geometry["spines"]:
            identity = geometry["id"]
            self.attachments[identity] = geometry["attachment_arc_um"]
            index = np.argmin(abs(self.boundaries - geometry["attachment_arc_um"]))
            neck = self.make_section(f"neck_{identity}", geometry["neck"], self.shaft[index - 1])
            head = self.make_section(f"head_{identity}", geometry["head"], neck)
            for mechanism in ("nmda", "ampa", "girk"):
                head.insert(mechanism)
            segment = head(0.5)
            scale = 10 * segment.area()
            stimulated = identity in SEEDS + SEQUENCE
            segment.girk.g_at_e = GIRK_AT_REVERSAL_NS / scale if stimulated else 0.0
            segment.nmda.g_at_e = NMDA_AT_REVERSAL_NS / scale if stimulated else 0.0
            segment.ampa.g_peak = AMPA_PEAK_NS / scale if stimulated else 0.0
            segment.ampa.tau = 0.5
            segment.ampa.onset = segment.nmda.onset = 1e9
            self.spines[identity] = neck, head
        set_channels(self.soma, "soma")
        self.soma.gbar_kv = 60.0
        self.soma.g_pas = 5 / 37500
        for name, value in CHANNELS["globals"].items():
            setattr(h, name, value)
        h.celsius, h.cainf_cad, h.taur_cad = 32.0, 0.0001, 20 / 1.2
        self.cv = h.CVode()
        self.cv.active(1)
        self.cv.atol(1e-6)
        self.cv.maxstep(0.05)
        self.nodes = [segment for section in self.sections for segment in section]
        self.shaft_nodes = [segment for section in self.shaft for segment in section]
        self.positions = np.array(
            [
                start + segment.x * section.L
                for start, section in zip(self.boundaries, self.shaft)
                for segment in section
            ]
        )
        self.length = self.geometry["length_um"]
        self.probe = self.shaft_nodes[np.argmin(abs(self.positions - self.length / 2))]

    def make_section(self, name, geometry, parent=None):
        """Create a tapered cable section with its specified mesh and passive membrane."""
        section = h.Section(name=name)
        h.pt3dclear(sec=section)
        for point, radius in zip(geometry["points"], geometry["radii"]):
            h.pt3dadd(*point, 2 * radius, sec=section)
        section.nseg = geometry["nseg"]
        passive(section, self.parameters)
        if parent is not None:
            section.connect(parent(1))
        self.sections.append(section)
        return section

    def shaft_at(self, identity):
        """Return the shaft attachment of a spine."""
        index = np.argmin(abs(self.boundaries - self.attachments[identity]))
        return self.shaft[index - 1](1)

    def manifest(self):
        """Report instantiated geometry, passive properties and channel densities."""
        return dict(
            source=self.geometry["source"],
            geometry_sha256=self.geometry_hash,
            temperature_c=float(h.celsius),
            calcium_rest_mm=float(h.cainf_cad),
            calcium_decay_ms=float(h.taur_cad),
            absolute_tolerance=float(self.cv.atol()),
            synaptic_parameters={
                name: {
                    field: float(getattr(getattr(self.spines[SEEDS[0]][1](0.5), name), field))
                    for field in fields
                }
                for name, fields in {
                    "ampa": ["e", "tau"],
                    "nmda": [
                        "e",
                        "vhalf",
                        "vslope",
                        "tau_rise",
                        "tau_decay_fast",
                        "tau_decay_mid",
                        "tau_decay_slow",
                        "fast_weight",
                        "normalization",
                    ],
                    "girk": ["e", "vhalf", "vslope"],
                }.items()
            },
            shaft_length_um=self.length,
            shaft_compartments=len(self.shaft_nodes),
            total_compartments=len(self.nodes),
            sections=[
                dict(
                    name=section.name(),
                    length_um=float(section.L),
                    nseg=int(section.nseg),
                    resistivity_ohm_cm=float(section.Ra),
                    area_um2=float(sum(s.area() for s in section)),
                    capacitance_uf_cm2=float(section.cm),
                    leak_s_cm2=float(section.g_pas),
                    leak_reversal_mv=float(section.e_pas),
                )
                for section in self.sections
            ],
            necks=[
                dict(
                    id=identity,
                    length_um=float(neck.L),
                    diameter_um=float(neck(0.5).diam),
                    resistance_mohm=float(h.ri(0.5, sec=neck) + h.ri(1, sec=neck)),
                    head_area_um2=float(head(0.5).area()),
                    attachment_arc_um=self.attachments[identity],
                    ampa_peak_ns=float(10 * head(0.5).area() * head(0.5).ampa.g_peak),
                    nmda_at_reversal_ns=float(10 * head(0.5).area() * head(0.5).nmda.g_at_e),
                    girk_at_reversal_ns=float(10 * head(0.5).area() * head(0.5).girk.g_at_e),
                )
                for identity, (neck, head) in self.spines.items()
            ],
            channel_parameters={
                region: {
                    mechanism: {name: float(getattr(section(0.5), name)) for name in fields}
                    for mechanism, fields in CHANNELS[region]["mechanisms"].items()
                }
                for region, section in (("shaft", self.shaft[0]), ("soma", self.soma))
            },
            ion_parameters={
                region: {name: float(getattr(section(0.5), name)) for name in CHANNELS["ions"]}
                for region, section in (("shaft", self.shaft[0]), ("soma", self.soma))
            },
        )

    def close(self):
        """Remove the branch and its spines from NEURON."""
        for section in self.sections:
            h.delete_section(sec=section)
