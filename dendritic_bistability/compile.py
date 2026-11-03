"""Compile the channel mechanisms."""

from pathlib import Path
import shutil
import subprocess
import sys


def main():
    """Compile the cylindrical-model and published Branco channel mechanisms."""
    nearby = Path(sys.executable).parent / "nrnivmodl"
    compiler = str(nearby) if nearby.exists() else shutil.which("nrnivmodl")
    if compiler is None:
        raise RuntimeError("Install NEURON and its mechanism compiler first.")
    folder = Path(__file__).resolve().parent / "mechanisms"
    subprocess.run([compiler], cwd=folder, check=True)
    subprocess.run([compiler, "mechanisms"], cwd=folder.parent / "branco", check=True)


if __name__ == "__main__":
    main()
