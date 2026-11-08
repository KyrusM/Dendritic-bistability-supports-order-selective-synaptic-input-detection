"""Sequence-selective spiking and input resistance in the MICrONS branch."""

from pathlib import Path

from dendritic_bistability.reconstruction_sequences import run_sequences
from dendritic_bistability.reconstruction_electrical import run_electrical
from dendritic_bistability.reconstruction_background import run_background
from dendritic_bistability.plotting.reconstruction import plot_reconstruction


RERUN_SIMULATIONS = True
WORKERS = 1


def main():
    """Simulate the input orders and current steps, then assemble Figure S7."""
    folder = Path(__file__).resolve().parent
    geometry, results = folder / "geometry.json", folder / "results"
    if RERUN_SIMULATIONS:
        run_sequences(geometry, folder / "orders.json", results, WORKERS)
        run_electrical(geometry, results)
        run_background(geometry, results, WORKERS)
    plot_reconstruction(results, geometry, folder / "plots")


if __name__ == "__main__":
    main()
