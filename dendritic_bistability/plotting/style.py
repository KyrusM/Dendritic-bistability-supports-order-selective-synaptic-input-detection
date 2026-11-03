"""Shared figure style."""

from pathlib import Path
from unittest.mock import patch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.transforms import Bbox
from matplotlib.backends.backend_svg import RendererSVG

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "lines.linewidth": 1.0,
    }
)


SVG_DRAW_TEXT = RendererSVG.draw_text


def _svg_text(renderer, gc, x, y, text, prop, angle, ismath=False, mtext=None):
    """Outline Computer Modern equations while keeping other SVG labels editable."""
    fonttype = "path" if ismath and prop.get_math_fontfamily() == "cm" else "none"
    with matplotlib.rc_context({"svg.fonttype": fonttype}):
        return SVG_DRAW_TEXT(renderer, gc, x, y, text, prop, angle, ismath=ismath, mtext=mtext)


def save(figure, destination, *, bounds=None, dpi=300, padding=0):
    """Apply tick notation and save SVG, PDF and PNG plots with room for axis labels."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.canvas.draw()
    for ax in figure.axes:
        for axis in (ax.xaxis, ax.yaxis):
            labels = [label.get_text() for label in axis.get_ticklabels()]
            replacements = {"0.4": ".4", "−0.4": "−.4", "-0.4": "−.4"}
            if any(label in replacements for label in labels):
                axis.set_major_locator(FixedLocator(axis.get_majorticklocs()))
                axis.set_major_formatter(FixedFormatter([replacements.get(label, label) for label in labels]))
    labels = [
        axis.label
        for ax in figure.axes
        for axis in (ax.xaxis, ax.yaxis, *([ax.zaxis] if hasattr(ax, "zaxis") else []))
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    if bounds is None:
        extent = Bbox.union([figure.bbox, *[label.get_window_extent(renderer) for label in labels]])
        bounds = extent.transformed(figure.dpi_scale_trans.inverted()).padded(0.15)
    for extension in ("svg", "pdf", "png"):
        with patch.object(RendererSVG, "draw_text", _svg_text):
            figure.savefig(path.with_suffix("." + extension), dpi=dpi, bbox_inches=bounds, pad_inches=padding)
    plt.close(figure)
    print(path.with_suffix(".png"))
