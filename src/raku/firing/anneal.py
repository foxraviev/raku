"""Validation-set calibration of the blend weight lambda.

Ref: Alg. 1 line 14 — lambda is fixed after training by minimising calibration
error of the blended confidence on held-out data.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from raku.calipers.calibrate import expected_calibration_error
from raku.wedging.plates import Sample
from raku.wheel.net import RareEyeVLM


@torch.no_grad()
def calibrate_lambda(
    model: RareEyeVLM, loader: DataLoader[Sample], device: torch.device, grid: int = 21
) -> float:
    model.eval()
    probs: list[np.ndarray] = []
    concept: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        out = model(batch["view"].to(device))
        probs.append(out["prob"].cpu().numpy())
        concept.append(out["concept_conf"].cpu().numpy())
        labels.append(batch["label"].numpy())
    p = np.concatenate(probs).astype(np.float64)
    cc = np.concatenate(concept).astype(np.float64)
    y = np.concatenate(labels).astype(np.float64)
    best_lambda, best_ece = 1.0, float("inf")
    for value in np.linspace(0.0, 1.0, grid):
        blended = value * p + (1.0 - value) * cc
        ece = expected_calibration_error(blended, y)
        if ece < best_ece:
            best_ece, best_lambda = ece, float(value)
    model.head.set_lambda(best_lambda)
    return best_lambda
