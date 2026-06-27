"""Evaluation pass collecting predictions and aggregate metrics."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

from raku.calipers.calibrate import expected_calibration_error, maximum_calibration_error
from raku.calipers.scores import Report, classification_report
from raku.wedging.plates import Sample
from raku.wheel.net import RareEyeVLM


class Gathered(TypedDict):
    prob: np.ndarray
    conf: np.ndarray
    label: np.ndarray


class EvalResult(TypedDict):
    report: Report
    ece: float
    mce: float


@torch.no_grad()
def collect(model: RareEyeVLM, loader: DataLoader[Sample], device: torch.device) -> Gathered:
    model.eval()
    probs: list[np.ndarray] = []
    confs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        out = model(batch["view"].to(device))
        probs.append(out["prob"].cpu().numpy())
        confs.append(out["conf"].cpu().numpy())
        labels.append(batch["label"].numpy())
    return {
        "prob": np.concatenate(probs).astype(np.float64),
        "conf": np.concatenate(confs).astype(np.float64),
        "label": np.concatenate(labels).astype(np.float64),
    }


def evaluate(
    model: RareEyeVLM, loader: DataLoader[Sample], device: torch.device, threshold: float = 0.5
) -> EvalResult:
    data = collect(model, loader, device)
    report = classification_report(data["prob"], data["label"], threshold)
    return {
        "report": report,
        "ece": expected_calibration_error(data["conf"], data["label"]),
        "mce": maximum_calibration_error(data["conf"], data["label"]),
    }
