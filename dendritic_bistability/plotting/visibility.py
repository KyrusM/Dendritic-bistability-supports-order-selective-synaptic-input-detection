"""Visibility of voltage traces on an orthographic surface."""

import numpy as np


def interpolate_surface(x, y, voltage, px, py):
    """Interpolate surface voltages at the specified horizontal coordinates."""
    ix = np.clip(np.searchsorted(x, px) - 1, 0, len(x) - 2)
    iy = np.clip(np.searchsorted(y, py) - 1, 0, len(y) - 2)
    u = np.clip((px - x[ix]) / (x[ix + 1] - x[ix]), 0, 1)
    v = np.clip((py - y[iy]) / (y[iy + 1] - y[iy]), 0, 1)
    return (
        (1 - u) * (1 - v) * voltage[ix, iy]
        + u * (1 - v) * voltage[ix + 1, iy]
        + (1 - u) * v * voltage[ix, iy + 1]
        + u * v * voltage[ix + 1, iy + 1]
    )


def visibility(points, x, y, voltage, direction, on_surface=False):
    """Identify trace points visible above the voltage surface from the camera direction."""
    fractions = np.linspace(1 / 768, 1, 768)
    visible = np.ones(len(points), dtype=bool)
    for start in range(0, len(points), 128):
        batch = points[start : start + 128]
        anchor = batch[:, 2]
        if on_surface:
            anchor = np.maximum(anchor, interpolate_surface(x, y, voltage, batch[:, 0], batch[:, 1]))
        end = np.minimum((x[-1] - batch[:, 0]) / direction[0], (y[0] - batch[:, 1]) / direction[1])
        distance = np.maximum(end, 0)[:, None] * fractions
        rx = batch[:, 0, None] + direction[0] * distance
        ry = batch[:, 1, None] + direction[1] * distance
        rz = anchor[:, None] + direction[2] * distance
        foreground = interpolate_surface(x, y, voltage, rx, ry)
        visible[start : start + len(batch)] = ~np.any(foreground > rz + 0.03, axis=1)
    return visible


def densify(original):
    """Add trace points to resolve changes in position, time and voltage before testing visibility."""
    change = np.abs(np.diff(original, axis=0))
    steps = np.maximum(1, np.ceil(np.max(change / [0.25, 0.2, 1.5], axis=1))).astype(int)
    return np.concatenate(
        [a + (b - a) * np.arange(n)[:, None] / n for (a, b, n) in zip(original[:-1], original[1:], steps)]
        + [original[-1:]]
    )
