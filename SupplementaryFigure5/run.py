"""Front propagation under timing jitter and additional synaptic inputs."""

import json
from pathlib import Path

from dendritic_bistability import Parameters
from dendritic_bistability.experiments import run_background_comparison
from dendritic_bistability.plotting.input_variability import plot_input_variability


RERUN_SIMULATIONS = True
WORKERS = 1


def main():
    """Simulate input perturbations and plot propagation for the three GIRK placements."""
    folder = Path(__file__).resolve().parent
    data = folder / "results"
    if RERUN_SIMULATIONS:
        inputs = json.loads((folder / "inputs.json").read_text())
        run_background_comparison(Parameters(seed_ampa_ns=2.0), inputs, data, WORKERS)
    plot_input_variability(data / "responses.json", folder / "plots" / "FigureS5")


if __name__ == "__main__":
    main()
