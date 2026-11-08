"""Input timing, jitter and additional excitation and inhibition."""

from pathlib import Path
import json
from dendritic_bistability import Parameters
from dendritic_bistability.commands import options, results_folder
from dendritic_bistability.experiments import run_background_comparison
from dendritic_bistability.plotting.background import plot_background_comparison


RERUN_SIMULATIONS = True


def main():
    """Compare timing, jitter and additional inputs across head GIRK, shaft GIRK and control models."""
    folder = Path(__file__).resolve().parent
    args = options(__doc__, simulate=RERUN_SIMULATIONS)
    data = results_folder(folder)
    if args.simulate:
        inputs = json.loads((folder / "inputs.json").read_text())
        model = Parameters(ampa_ns=0.35, seed_ampa_ns=2.0)
        run_background_comparison(model, inputs, data, args.workers, args.limit)
    plot_background_comparison(data / "responses.json", folder / "plots" / "FigureS1")


if __name__ == "__main__":
    main()
