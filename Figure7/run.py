"""Figure 7: minimum spacing at six fixed GIRK conductances."""

from pathlib import Path

from dendritic_bistability.commands import options
from dendritic_bistability.sweeps import ConductanceCondition, run_fixed_girk_sweep
from dendritic_bistability.plotting.sweeps import save_minimum_spacing_figure


RERUN_SIMULATIONS = True

# Bounds: spacing (µm), conductance imbalance and peak AMPA (nS).
BASE_BOUNDS = ((0.5, 21.682771382138135), (-0.9, 0.55), (0.0, 1.4))
EXTENSION_BOUNDS = ((21.682771382138135, 42.86554276427627), (-0.9, 0.55), (0.0, 1.4))


def main():
    """Sweep spacing, conductance imbalance and AMPA, then plot the minimum-spacing curves."""
    folder = Path(__file__).resolve().parent
    args = options(__doc__, sweep=True, simulate=RERUN_SIMULATIONS, fixed_depth=8)
    results = folder / "results"
    settings = dict(bounds=BASE_BOUNDS, depth=args.depth, workers=args.workers, rerun=args.simulate)
    curves = {}

    curves.update(run_fixed_girk_sweep(
        ConductanceCondition("shaft_full", "shaft", 1.15), results, **settings,
    ))

    curves.update(run_fixed_girk_sweep(
        ConductanceCondition("shaft_half", "shaft", 0.575), results, **settings,
    ))

    curves.update(run_fixed_girk_sweep(
        ConductanceCondition("shaft_quarter", "shaft", 0.2875), results, **settings,
    ))

    curves.update(run_fixed_girk_sweep(
        ConductanceCondition("head_full", "head", 5.7), results, **settings,
    ))

    curves.update(run_fixed_girk_sweep(
        ConductanceCondition("head_half", "head", 2.85), results,
        extension_bounds=EXTENSION_BOUNDS, extension_depth=7, **settings,
    ))

    curves.update(run_fixed_girk_sweep(
        ConductanceCondition("head_quarter", "head", 1.425), results,
        extension_bounds=EXTENSION_BOUNDS, extension_depth=7, **settings,
    ))

    save_minimum_spacing_figure(curves, folder, "Figure7")


if __name__ == "__main__":
    main()
