"""Reconstruct the Figure S7 cable from the MICrONS source morphology."""

from pathlib import Path
import json

from reconstruction import prepare_geometry


def main():
    """Regenerate the geometry JSON used by Figure S7."""
    folder = Path(__file__).resolve().parent
    geometry = prepare_geometry(folder / "inputs")
    output = folder.parent / "geometry.json"
    output.write_text(json.dumps(geometry, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
