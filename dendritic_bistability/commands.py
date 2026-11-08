"""Figure command-line options."""

import argparse


def results_folder(folder):
    """Locate a figure's numerical results."""
    return folder / "results"


def options(description, sweep=False, simulate=False, *, fixed_depth=None):
    """Read optional simulation, worker-count and sweep-depth or condition-limit arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--simulate", action="store_true", default=simulate, help="Run simulations before plotting."
    )
    parser.add_argument("--workers", type=int, default=1)
    if sweep:
        if fixed_depth is None:
            parser.add_argument("--depth", type=int, default=8, help="Adaptive sweep depth.")
        else:
            parser.set_defaults(depth=fixed_depth)
    else:
        parser.add_argument("--limit", type=int, help="Run only this many simulation conditions.")
    return parser.parse_args()
