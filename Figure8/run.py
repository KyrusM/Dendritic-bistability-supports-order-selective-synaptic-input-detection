"""Sequence-selective spiking and the effects of a somatic spike."""

from pathlib import Path
import json
from dendritic_bistability import Parameters, load_result
from dendritic_bistability.commands import options, results_folder
from dendritic_bistability.experiments import run_trials
from dendritic_bistability.plotting.voltage import plot_examples
from dendritic_bistability.plotting.spiking import plot_spiking


RERUN_SIMULATIONS = True


def main():
    """Simulate sequence-selective spiking and imposed somatic spikes, then plot counts and voltage examples."""
    folder = Path(__file__).resolve().parent
    args = options(__doc__, simulate=RERUN_SIMULATIONS)
    data = results_folder(folder)

    panels = {
        "A": ("gap10_ordered", "gap10_swap_4_5"),
        "B": ("swap_4_5_spike_target80ms", "swap_7_8_spike_target140ms"),
    }

    # Vary the soma gap and imposed-spike timing; retain the plotted examples.
    if args.simulate:
        inputs = json.loads((folder / "inputs.json").read_text())
        model = Parameters(
            nmda_ns=2.4,
            ampa_ns=0.35,
            seed_ampa_ns=0.35,
            girk_shaft_ns_per_um=0.45,
            soma="active",
            soma_size_um=25.0,
            soma_leak_multiplier=5.0,
        )
        recordings = [name for names in panels.values() for name in names]
        run_trials(model, inputs, data, args.workers, args.limit, recordings, plot_end_ms=450)

    # Plot spike counts and the corresponding voltage examples.
    plot_spiking(data / "responses.json", folder / "plots")
    for panel, names in panels.items():
        examples = {
            name: load_result(data / (name + ".npz")) for name in names if (data / (name + ".npz")).exists()
        }
        if examples:
            plot_examples(examples, folder / "plots" / panel, window=(-20, 450))


if __name__ == "__main__":
    main()
