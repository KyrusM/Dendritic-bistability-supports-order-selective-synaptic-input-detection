"""Background membrane and AMPA conductances."""

from pathlib import Path
from itertools import product
from dendritic_bistability.commands import options, results_folder
from dendritic_bistability.sweeps import PropagationSweep, adaptive_sweep
from dendritic_bistability.plotting.sections import plot_conductance_ranges


RERUN_SIMULATIONS = True


def main():
    """Map propagation across background conductance densities and large AMPA conductances."""
    folder = Path(__file__).resolve().parent
    args = options(__doc__, sweep=True, simulate=RERUN_SIMULATIONS)
    data = results_folder(folder)
    for placement, ampa, scale in product(("shaft", "head"), (0.175, 1.4), (0, 1, 5)):
        path = data / (f"{placement}_{scale}_{ampa:g}.npz")
        if args.simulate:
            sweep = PropagationSweep(placement, ampa_ns=ampa, background_multiplier=scale)
            adaptive_sweep(
                sweep,
                [[0.5, 21.682771382138135], [-0.9, 0.55]],
                path,
                depth=args.depth,
                workers=args.workers,
            )
    for placement, ampa in product(("shaft", "head"), (5.0, 10.0)):
        path = data / "large_ampa" / f"{placement}_{ampa:g}.npz"
        if args.simulate:
            sweep = PropagationSweep(placement, ampa_ns=ampa)
            adaptive_sweep(
                sweep,
                [[0.5, 21.682771382138135], [-0.9, 0.55]],
                path,
                depth=args.depth,
                workers=args.workers,
            )
    plot_conductance_ranges(data, folder / "plots" / "FigureS3")


if __name__ == "__main__":
    main()
