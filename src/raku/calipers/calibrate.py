"""Calibration error on multi-label outputs.

Ref: Sec. V.I — ECE and MCE over 10 equal-width probability bins; every class
prediction is treated as one binary calibration event.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def reliability(probs: Array, labels: Array, bins: int = 10) -> tuple[Array, Array, Array]:
    p = probs.reshape(-1)
    y = labels.reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    conf = np.zeros(bins)
    acc = np.zeros(bins)
    weight = np.zeros(bins)
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (p > lo) & (p <= hi) if b > 0 else (p >= lo) & (p <= hi)
        if sel.any():
            conf[b] = float(p[sel].mean())
            acc[b] = float(y[sel].mean())
            weight[b] = float(sel.mean())
    return conf, acc, weight


def expected_calibration_error(probs: Array, labels: Array, bins: int = 10) -> float:
    conf, acc, weight = reliability(probs, labels, bins)
    return float((weight * np.abs(acc - conf)).sum())


def maximum_calibration_error(probs: Array, labels: Array, bins: int = 10) -> float:
    conf, acc, weight = reliability(probs, labels, bins)
    gaps = np.abs(acc - conf)[weight > 0]
    return float(gaps.max()) if gaps.size else 0.0
