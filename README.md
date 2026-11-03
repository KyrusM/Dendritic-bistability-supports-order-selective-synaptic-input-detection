# Dendritic order-selective front propagation

Code for the theoretical analysis, biophysical simulations and figures in the manuscript.

## Contents

- `Mathematica/PaperFigures.nb`: theoretical calculations and figure formatting.
- `Figure1/`, `Figure6/`, `Figure7/` and `Figure8/`: Python scripts for the biophysical figures.
- `SupplementaryFigure1/` and `SupplementaryFigure3/`–`SupplementaryFigure7/`: Python scripts for the supplementary figures. Figure S7 includes the reconstructed branch geometry and input orders.
- [`SupplementaryFigure7/Reconstruction/`](SupplementaryFigure7/Reconstruction/README.md): geometry preparation, source morphology and setup instructions.
- `dendritic_bistability/`: shared model, simulation, analysis and plotting functions, with channel mechanisms.
- `SOURCES.md`: source papers, code versions and reuse notices.

## Python

Use Python 3.9–3.11 and a C/C++ compiler. On macOS, install the Xcode command-line tools; on Windows, use WSL. NEURON's compiler requires an environment path without spaces. From the package folder:

```sh
python -m venv "$HOME/.venvs/dendritic-bistability"
source "$HOME/.venvs/dendritic-bistability/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e .
python -m dendritic_bistability.compile
```

Select this Python environment in your editor, open the figure's `run.py` and press Run. Alternatively, run the script from the package folder:

```sh
python Figure1/run.py
```

Each script runs its simulations and saves numerical results in `results/` and figures in `plots/` within that figure's folder. Set `RERUN_SIMULATIONS = False` to redraw saved results without rerunning simulations. Set `WORKERS` in the Figure S5, S6 or S7 script to choose the number of processes. The other scripts accept optional command-line arguments (`--help`).

Install Arial to match the figure lettering. Figure S7 also uses Poppler's `pdftoppm` for PNG export: `brew install poppler` on macOS or `sudo apt install poppler-utils` on Ubuntu.

## Mathematica

Open `Mathematica/PaperFigures.nb` in Mathematica (tested with 14.2), evaluate the initialization cells, then evaluate the cells in the desired figure section in order. Calculations, formatting and saved outputs remain together in the notebook. Figure cells export panels to the corresponding `plots/` folder.

The notebook generates Figures 2–5, 6A and S2, and formats Python results for Figures 6B and 7. Figure 6A is under “Spines” → “Checking drift.” Run the corresponding Python scripts before formatting Figures 6B and 7.

## Sources

Channel mechanisms and parameters derive from Gao et al. (2021; [ModelDB 249705](https://modeldb.science/249705)). Figure S1 also includes the published model of Branco, Clark & Häusser (2010; [ModelDB 140828](https://modeldb.science/140828)). Figure S7 uses geometry derived from the MICrONS reconstruction (MICrONS Consortium, 2025).

See [SOURCES.md](SOURCES.md) for the full source list and file-level attributions. Original author notices are retained in the reused files.

