"""Single-swap tolerance and rejection of a two-away swap."""

from pathlib import Path
from dendritic_bistability.commands import options, results_folder
from dendritic_bistability.sweeps import PropagationSweep, adaptive_sweep
from dendritic_bistability.plotting.sections import plot_swap_tolerance


RERUN_SIMULATIONS = True


def main():
    """Compare ordered, adjacent-swap and two-away-swap propagation across conductances and spacing."""
    folder = Path(__file__).resolve().parent
    args = options(__doc__, sweep=True, simulate=RERUN_SIMULATIONS)
    data = results_folder(folder)
    for placement in ("shaft", "head"):
        for ampa in (0.175, 1.4):
            path = data / (placement + f"_{ampa:g}.npz")
            if args.simulate:
                sweep = PropagationSweep(
                    placement,
                    ampa_ns=ampa,
                    orders=((1, 2, 3, 4, 5, 6), (1, 2, 3, 5, 4, 6), (1, 4, 5, 2, 3, 6)),
                )
                adaptive_sweep(
                    sweep,
                    [[0.5, 21.682771382138135], [-0.9, 0.55]],
                    path,
                    depth=args.depth,
                    workers=args.workers,
                )
    plot_swap_tolerance(data, folder / "plots" / "FigureS4")


if __name__ == "__main__":
    main()
