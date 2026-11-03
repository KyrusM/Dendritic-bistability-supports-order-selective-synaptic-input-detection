"""Run Figure S6 with lognormal synaptic and geometric variability."""

from pathlib import Path

from dendritic_bistability import Parameters
from dendritic_bistability.variability import run_variability
from dendritic_bistability.variability_sampling import variability_samples
from dendritic_bistability.plotting.variability import plot_variability


RERUN_SIMULATIONS = True
WORKERS = 1


def main():
    """Simulate the sampled dendrites and plot propagation by input order and GIRK placement."""
    folder = Path(__file__).resolve().parent
    if RERUN_SIMULATIONS:
        run_variability(
            Parameters(seed_ampa_ns=2.0, recovery_tolerance_mv=5.0,
                       stop_ms=2500.0, maximum_stop_ms=6000.0),
            variability_samples(), folder / "results", WORKERS,
        )
    plot_variability(folder / "results/counts.csv", folder / "plots/FigureS6")


if __name__ == "__main__":
    main()
