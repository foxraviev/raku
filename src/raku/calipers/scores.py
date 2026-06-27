"""Multi-label classification metrics.

Ref: Sec. IV.A — macro AUC, macro F1, sample-based accuracy, macro sensitivity
and specificity, Cohen's kappa. AUC is the rank statistic; no external metric
library is used so numbers are reproducible across environments.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class Report(TypedDict):
    auc: float
    f1: float
    accuracy: float
    sensitivity: float
    specificity: float
    kappa: float


def _rankdata(values: Array) -> Array:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def binary_auc(scores: Array, labels: Array) -> float:
    pos = labels > 0.5
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def macro_auc(probs: Array, labels: Array) -> float:
    per_class = [binary_auc(probs[:, c], labels[:, c]) for c in range(labels.shape[1])]
    valid = [a for a in per_class if not np.isnan(a)]
    return float(np.mean(valid)) if valid else float("nan")


def _counts(pred: Array, labels: Array) -> tuple[float, float, float, float]:
    tp = float(((pred == 1) & (labels == 1)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    tn = float(((pred == 0) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    return tp, fp, tn, fn


def macro_f1(probs: Array, labels: Array, threshold: float) -> float:
    pred = (probs >= threshold).astype(np.float64)
    scores = []
    for c in range(labels.shape[1]):
        tp, fp, _, fn = _counts(pred[:, c], labels[:, c])
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


def sample_accuracy(probs: Array, labels: Array, threshold: float) -> float:
    pred = (probs >= threshold).astype(np.float64)
    return float((pred == labels).mean(axis=1).mean())


def macro_sensitivity(probs: Array, labels: Array, threshold: float) -> float:
    pred = (probs >= threshold).astype(np.float64)
    vals = []
    for c in range(labels.shape[1]):
        tp, _, _, fn = _counts(pred[:, c], labels[:, c])
        if tp + fn > 0:
            vals.append(tp / (tp + fn))
    return float(np.mean(vals)) if vals else float("nan")


def macro_specificity(probs: Array, labels: Array, threshold: float) -> float:
    pred = (probs >= threshold).astype(np.float64)
    vals = []
    for c in range(labels.shape[1]):
        _, fp, tn, _ = _counts(pred[:, c], labels[:, c])
        if tn + fp > 0:
            vals.append(tn / (tn + fp))
    return float(np.mean(vals)) if vals else float("nan")


def binary_kappa(pred: Array, labels: Array) -> float:
    tp, fp, tn, fn = _counts(pred, labels)
    total = tp + fp + tn + fn
    if total == 0:
        return float("nan")
    observed = (tp + tn) / total
    p_pred_pos = (tp + fp) / total
    p_true_pos = (tp + fn) / total
    expected = p_pred_pos * p_true_pos + (1 - p_pred_pos) * (1 - p_true_pos)
    if expected >= 1.0:
        return 0.0
    return float((observed - expected) / (1 - expected))


def macro_kappa(probs: Array, labels: Array, threshold: float) -> float:
    pred = (probs >= threshold).astype(np.float64)
    vals = [binary_kappa(pred[:, c], labels[:, c]) for c in range(labels.shape[1])]
    clean = [v for v in vals if not np.isnan(v)]
    return float(np.mean(clean)) if clean else float("nan")


def classification_report(probs: Array, labels: Array, threshold: float = 0.5) -> Report:
    return {
        "auc": macro_auc(probs, labels),
        "f1": macro_f1(probs, labels, threshold),
        "accuracy": sample_accuracy(probs, labels, threshold),
        "sensitivity": macro_sensitivity(probs, labels, threshold),
        "specificity": macro_specificity(probs, labels, threshold),
        "kappa": macro_kappa(probs, labels, threshold),
    }
