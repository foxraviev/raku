from __future__ import annotations

import numpy as np

from raku.calipers.calibrate import expected_calibration_error, maximum_calibration_error
from raku.calipers.intervals import bootstrap_ci, delong_test, delong_variance, fleiss_kappa
from raku.calipers.scores import binary_auc, macro_auc, macro_f1, macro_kappa


def test_auc_separation_and_ties() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert binary_auc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert binary_auc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert binary_auc(np.array([0.5, 0.5, 0.5, 0.5]), y) == 0.5


def test_auc_known_value() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0])
    scores = np.array([0.2, 0.3, 0.4, 0.1])
    assert abs(binary_auc(scores, y) - 0.25) < 1e-9


def test_macro_metrics_on_perfect_predictions() -> None:
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    labels = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert macro_auc(probs, labels) == 1.0
    assert macro_f1(probs, labels, 0.5) == 1.0
    assert macro_kappa(probs, labels, 0.5) == 1.0


def test_ece_zero_for_calibrated_stream() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=20000)
    y = (rng.uniform(0, 1, size=20000) < p).astype(np.float64)
    assert expected_calibration_error(p[:, None], y[:, None], 10) < 0.02
    assert maximum_calibration_error(p[:, None], y[:, None], 10) < 0.05


def test_bootstrap_contains_point_estimate() -> None:
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, size=(200, 3))
    labels = (rng.uniform(0, 1, size=(200, 3)) < probs).astype(np.float64)
    point, lo, hi = bootstrap_ci(macro_auc, probs, labels, iterations=300, seed=0)
    assert lo <= point <= hi


def test_delong_self_difference_is_insignificant() -> None:
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, size=300)
    labels = (rng.uniform(0, 1, size=300) < 0.5).astype(np.float64)
    diff, p = delong_test(scores, scores, labels)
    assert abs(diff) < 1e-9
    auc, var = delong_variance(scores, labels)
    assert 0.0 <= auc <= 1.0 and var >= 0.0


def test_fleiss_kappa_bounds() -> None:
    perfect = np.array([[5, 0], [0, 5], [5, 0]], dtype=np.float64)
    assert abs(fleiss_kappa(perfect) - 1.0) < 1e-9
    split = np.array([[3, 2], [2, 3], [3, 2]], dtype=np.float64)
    assert fleiss_kappa(split) < 0.5
