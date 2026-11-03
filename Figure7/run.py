"""Minimum spacing at specified GIRK conductances."""

from pathlib import Path
import json
import numpy as np
from dendritic_bistability.commands import options, results_folder
from dendritic_bistability.sweeps import PropagationSweep, adaptive_sweep, minimum_spacing
from dendritic_bistability.plotting.sweeps import plot_minima
from dendritic_bistability.plotting.projections import project_minima


RERUN_SIMULATIONS = True


def main():
    """Map propagation at fixed GIRK conductances and plot minimum spacing and imbalance ranges."""
    folder = Path(__file__).resolve().parent
    args = options(__doc__, sweep=True, simulate=RERUN_SIMULATIONS)

    # Sweep spacing, imbalance and AMPA at each fixed GIRK conductance.
    if args.simulate:
        curves = {}
        for placement, reference in (("shaft", 1.15), ("head", 5.7)):
            for name, fraction in (("full", 1.0), ("half", 0.5), ("quarter", 0.25)):
                path = folder / "results" / (placement + "_" + name + ".npz")
                sweep = PropagationSweep(placement, fixed_girk=reference * fraction, nmda_limit_ns=9.6)
                maximum_spacing = (
                    42.86554276427627 if placement == "head" and fraction < 1 else 21.682771382138135
                )
                adaptive_sweep(
                    sweep,
                    [[0.5, maximum_spacing], [-0.9, 0.55], [0.0, 1.4]],
                    path,
                    depth=args.depth,
                    workers=args.workers,
                )

                # Extract minimum spacing at the two imbalance tolerances.
                with np.load(path, allow_pickle=False) as data:
                    for width in (0.05, 0.1):
                        curves[f"{placement}_{name}_width_{width}"] = minimum_spacing(
                            data["classes"], data["bounds"], width
                        )
        (folder / "results" / "curves.json").write_text(json.dumps(curves, indent=2))
        (folder / "results" / "projections.json").write_text(json.dumps(project_minima(curves)))

    # Plot minimum-spacing curves with their imbalance projections.
    path = results_folder(folder) / "curves.json"
    plot_minima(json.loads(path.read_text()), folder / "plots" / "Figure7")


if __name__ == "__main__":
    main()
