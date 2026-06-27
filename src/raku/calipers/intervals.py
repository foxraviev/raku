"""Resampling intervals, DeLong AUC comparison and Fleiss' kappa.

Ref: Sec. V.H/I — bootstrap CIs (10000 iterations), DeLong test for paired AUC
differences, Fleiss' kappa for multi-rater agreement.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from raku.calipers.scores import _rankdata

Array = NDArray[np.float64]
Metric = Callable[[Array, Array], float]


def bootstrap_ci(
    metric: Metric,
    probs: Array,
    labels: Array,
    iterations: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = probs.shape[0]
    point = metric(probs, labels)
    draws = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        draws[i] = metric(probs[idx], labels[idx])
    finite = draws[np.isfinite(draws)]
    lo = float(np.quantile(finite, alpha / 2))
    hi = float(np.quantile(finite, 1 - alpha / 2))
    return point, lo, hi


def _structural(scores: Array, pos: NDArray[np.bool_]) -> tuple[Array, Array]:
    x = scores[pos]
    y = scores[~pos]
    m, n = len(x), len(y)
    tx = _rankdata(x)
    ty = _rankdata(y)
    tz = _rankdata(scores)
    v10 = (tz[pos] - tx) / n
    v01 = 1.0 - (tz[~pos] - ty) / m
    return v10, v01


def delong_variance(scores: Array, labels: Array) -> tuple[float, float]:
    pos = labels > 0.5
    m = int(pos.sum())
    n = int((~pos).sum())
    if m == 0 or n == 0:
        return float("nan"), float("nan")
    v10, v01 = _structural(scores, pos)
    auc = float(v10.mean())
    var = float(v10.var(ddof=1) / m + v01.var(ddof=1) / n)
    return auc, var


def delong_test(scores_a: Array, scores_b: Array, labels: Array) -> tuple[float, float]:
    pos = labels > 0.5
    m = int(pos.sum())
    n = int((~pos).sum())
    a10, a01 = _structural(scores_a, pos)
    b10, b01 = _structural(scores_b, pos)
    auc_a, auc_b = float(a10.mean()), float(b10.mean())
    cov10 = np.cov(np.stack([a10, b10]))
    cov01 = np.cov(np.stack([a01, b01]))
    cov = cov10 / m + cov01 / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return auc_a - auc_b, float("nan")
    z = (auc_a - auc_b) / np.sqrt(var)
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return auc_a - auc_b, float(p)


def fleiss_kappa(counts: Array) -> float:
    items, categories = counts.shape
    raters = counts.sum(axis=1)
    if not np.all(raters == raters[0]):
        raise ValueError("Fleiss' kappa expects a fixed number of raters per item")
    r = float(raters[0])
    p_j = counts.sum(axis=0) / (items * r)
    p_i = (np.square(counts).sum(axis=1) - r) / (r * (r - 1))
    p_bar = float(p_i.mean())
    p_e = float(np.square(p_j).sum())
    if p_e >= 1.0:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def _normal_cdf(x: float) -> float:
    return float(0.5 * (1.0 + _erf(x / np.sqrt(2.0))))


def _erf(x: float) -> float:
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    y = 1.0 - (
        ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592
    ) * t * np.exp(-x * x)
    return float(np.sign(x) * y)
