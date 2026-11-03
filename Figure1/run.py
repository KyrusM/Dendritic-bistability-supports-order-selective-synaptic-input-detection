"""Ordered and disrupted sequences with a passive soma."""

from pathlib import Path
import json
from dendritic_bistability import Parameters, load_result
from dendritic_bistability.commands import options, results_folder
from dendritic_bistability.experiments import run_trials
from dendritic_bistability.plotting.voltage import plot_examples
from dendritic_bistability.plotting.responses import plot_responses


RERUN_SIMULATIONS = True


def main():
    """Simulate input orders with a passive soma and plot voltage examples and response amplitudes."""
    folder = Path(__file__).resolve().parent
    args = options(__doc__, simulate=RERUN_SIMULATIONS)
    data = results_folder(folder)
    inputs = json.loads((folder / "inputs.json").read_text())
    examples = {"impedance": ("ordered", "reversed"), "front": ("ordered", "swap_4_5")}

    # Run all input orders and retain the plotted voltage examples.
    if args.simulate:
        model = Parameters(
            nmda_ns=2.4,
            ampa_ns=0.35,
            seed_ampa_ns=0.35,
            girk_shaft_ns_per_um=0.45,
            soma="passive",
            soma_size_um=25.0,
            soma_leak_multiplier=5.0,
        )
        recordings = [model + "_" + name for model, orders in examples.items() for name in orders]
        run_trials(model, inputs, data, args.workers, args.limit, recordings, plot_end_ms=280)

    # Plot the voltage examples and summarize all sequence responses.
    for panel, (model, orders) in zip(("A", "B"), examples.items()):
        names = [model + "_" + name for name in orders]
        results = {
            name: load_result(data / (name + ".npz")) for name in names if (data / (name + ".npz")).exists()
        }
        if results:
            plot_examples(results, folder / "plots" / panel, window=(-20, 280))
    plot_responses(data / "responses.json", folder / "plots")


if __name__ == "__main__":
    main()
