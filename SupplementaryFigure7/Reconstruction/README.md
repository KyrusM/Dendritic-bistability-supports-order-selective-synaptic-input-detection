# MICrONS reconstruction

Geometry preparation for Figure S7 of *Dendritic bistability supports order-selective synaptic input detection*.

- `prepare_geometry.py`: runnable entry point.
- `reconstruction/`: surface measurements and cable reduction.
- `inputs/`: the released cell SWC, LOD0 membrane mesh and fixed figure projection.
- `SOURCES.md`: data sources and versions.

Use a separate Python 3.9–3.11 environment. From this folder:

```sh
python3 -m venv "$HOME/.venvs/microns-reconstruction"
source "$HOME/.venvs/microns-reconstruction/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Select that environment in your editor, open `prepare_geometry.py` and press Run. Alternatively:

```sh
python prepare_geometry.py
```

The script overwrites `../geometry.json`, the geometry read by the Figure S7 simulation script. It measures the supplied morphology and applies the geometric reductions described in Supplementary Methods. The camera and marker directions are fixed figure-formatting inputs. No membrane simulations run, and no network connection is required.
