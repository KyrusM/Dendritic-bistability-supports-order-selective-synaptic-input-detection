"""Biophysical conductance and spacing ranges for order-selective front propagation."""

from pathlib import Path
from dendritic_bistability.commands import options, results_folder
from dendritic_bistability.sweeps import PropagationSweep, adaptive_sweep
from dendritic_bistability.plotting.sweeps import plot_surfaces, plot_spacing_minima


RERUN_SIMULATIONS = True


def main():
    """Map and plot the biophysical propagation ranges and minimum spacings in Figure 6B."""
    folder = Path(__file__).resolve().parent
    args = options(__doc__, sweep=True, simulate=RERUN_SIMULATIONS)
    data = results_folder(folder)
    files = [data / ("B_" + placement + ".npz") for placement in ("shaft", "head")]

    # Sweep spacing, NMDA–GIRK imbalance and AMPA conductance.
    if args.simulate:
        for placement, path in zip(("shaft", "head"), files):
            adaptive_sweep(
                PropagationSweep(placement, gc_ns=4.075),
                [[0.5, 21.682771382138135], [-0.9, 0.55], [0.0, 1.4]],
                path,
                depth=args.depth,
                workers=args.workers,
            )

    # Plot propagation boundaries and the minimum spacing at 0.1 imbalance tolerance.
    plot_surfaces(files, folder / "plots" / "B")
    plot_spacing_minima(files, folder / "plots" / "Biii", width=0.1)


if __name__ == "__main__":
    main()
