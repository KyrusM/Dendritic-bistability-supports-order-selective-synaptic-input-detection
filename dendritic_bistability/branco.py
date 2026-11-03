"""Published Branco dendrite and receptor mechanisms."""

from pathlib import Path
import os

import numpy as np
from neuron import h, load_mechanisms

from .results import save_result


class BrancoDendrite:
    """Load and stimulate the published Branco reconstruction and receptor mechanisms."""

    def __init__(self):
        """Load the reconstruction, channel parameters and nine sequence synapses."""
        folder = Path(__file__).parent / "branco"
        load_mechanisms(str(folder))
        previous = Path.cwd()
        try:
            os.chdir(folder)
            h.load_file("stdrun.hoc")
            for file in ("rc19.hoc", "init_biophys.hoc", "init_synapses.hoc"):
                h.xopen(file)
            h.init_params()
            h.init_passive()
            h.init_syn_params()
            h.init_syns(9)
        finally:
            os.chdir(previous)
        self.sites = [h.cAMPA[i].get_segment() for i in range(9)]
        self.sections = [s for s in h.allsec() if not s.name().startswith("PRE")]

    def run(self, trial, destination=None):
        """Deliver a sequence and additional inputs, recording the final-site shaft response through recovery."""
        order, times = trial["order"], trial["input_times_ms"]
        for i in range(9):
            h.cAMPA[i].onset = h.PRE[i].del_rel = 1e9
        for site, at in zip(order, times):
            h.cAMPA[site - 1].onset = h.PRE[site - 1].del_rel = at

        # Add repeated glutamate inputs, intervening inputs and GABA_A inputs.
        background = trial.get("background", {})
        objects, sections, events = [], [], []
        grouped = {}
        for site, at in background.get("sequence", []):
            grouped.setdefault(site, []).append(at)
        for site, extra in grouped.items():
            i = site - 1
            release = h.SequenceRelease(self.sites[i])
            connection = h.NetCon(None, release)
            connection.weight[0] = 1.0
            h.setpointer(release._ref_T, "C", h.cNMDA[i])
            for at in [h.cAMPA[i].onset, *extra]:
                ampa = h.AlphaSynapse(self.sites[i])
                ampa.tau, ampa.gmax, ampa.e, ampa.onset = 2.0, 0.0005, 0.0, at
                objects.append(ampa)
                events.append((connection, at))
            h.cAMPA[i].onset = h.PRE[i].del_rel = 1e9
            objects.extend([release, connection])
        for position, at in background.get("spines", []):
            segment = self.sites[0].sec(self.sites[0].x + (self.sites[-1].x - self.sites[0].x) * position)
            pre = h.Section(name="transmitter")
            pre.L = pre.diam = 1.0
            pre.insert("rel")
            pre.dur_rel, pre.amp_rel, pre.del_rel = 0.5, 2.0, at
            ampa = h.AlphaSynapse(segment)
            ampa.tau, ampa.gmax, ampa.e, ampa.onset = 2.0, 0.0005, 0.0, at
            nmda = h.NMDA_Mg_T(segment)
            nmda.gmax = 8000.0
            h.setpointer(pre(0.5)._ref_T_rel, "C", nmda)
            objects.extend([ampa, nmda])
            sections.append(pre)
        for position, at in background.get("inhibition", []):
            x = self.sites[0].x + (self.sites[-1].x - self.sites[0].x) * position
            synapse = h.Exp2Syn(self.sites[0].sec(x))
            synapse.tau1, synapse.tau2, synapse.e = 0.3, 10.0, -75.0
            connection = h.NetCon(None, synapse)
            connection.weight[0] = 0.0005
            objects.extend([synapse, connection])
            events.append((connection, at))

        # Record the response and continue until the dendrite recovers.
        time = h.Vector().record(h._ref_t, 0.1)
        voltage = h.Vector().record(self.sites[-1]._ref_v, 0.1)
        soma = h.Vector().record(h.soma(0.5)._ref_v, 0.1)
        all_time = h.Vector().record(h._ref_t, 2.0)
        all_voltage = [h.Vector().record(seg._ref_v, 2.0) for s in self.sections for seg in s]
        h.CVode().active(0)
        h.dt, h.steps_per_ms = 0.025, 40
        h.finitialize(-75.0)
        for connection, at in events:
            connection.event(at)
        stop = max(1500.0, max(times) + 500.0)
        while True:
            h.continuerun(stop)
            t, v = np.asarray(all_time), np.array(all_voltage)
            baseline = np.searchsorted(t, min(times)) - 1
            tail = t >= stop - 200.0
            if np.max(abs(v[:, tail] - v[:, baseline, None])) <= 5.0:
                break
            stop += 800.0
            if stop > 6300.0:
                raise RuntimeError("The Branco dendrite did not recover.")

        # Measure the final-site response and restore the sequence synapses.
        t, v = np.asarray(time).copy(), np.asarray(voltage).copy()
        baseline = np.searchsorted(t, min(times)) - 1
        row = {k: v for k, v in trial.items() if k not in ("background", "parameters")}
        row["peak_shaft_response_mv"] = float(v[t >= min(times)].max() - v[baseline])
        if destination is not None:
            save_result(
                destination, dict(time_ms=t, voltage_mv=v, soma_mv=np.asarray(soma).copy(), condition=row)
            )
        for i in grouped:
            h.setpointer(h.PRE[i - 1](0.5)._ref_T_rel, "C", h.cNMDA[i - 1])
        objects.clear()
        for section in sections:
            h.delete_section(sec=section)
        return row
