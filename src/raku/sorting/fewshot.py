"""Few-shot evaluation across the five reported seeds.

Ref: Sec. IV.A — for each K-shot setting the protocol re-samples support sets
for seeds {0, 1, 2, 42, 2024} and reports mean and standard deviation.
"""

from __future__ import annotations

import dataclasses
from typing import TypedDict

import numpy as np
import torch

from raku.bench.glaze_io import kiln_log, set_seed
from raku.bench.settings import Settings
from raku.firing.trainer import Trainer
from raku.sorting.runner import evaluate
from raku.wedging.feed import build_loader
from raku.wheel.net import build_model

_LOG = kiln_log("raku.sorting")


class ShotSummary(TypedDict):
    shots: int
    auc_mean: float
    auc_std: float
    f1_mean: float
    f1_std: float


def run_shots(settings: Settings, shots: int, device: torch.device) -> ShotSummary:
    aucs: list[float] = []
    f1s: list[float] = []
    for seed in settings.station.eval_seeds:
        set_seed(seed)
        run = dataclasses.replace(
            settings,
            seam=dataclasses.replace(settings.seam, shots=shots),
            kiln=dataclasses.replace(settings.kiln, seed=seed),
        )
        model = build_model(run.body, run.seam.dataset, run.seam.num_classes)
        trainer = Trainer(model, run, device)
        trainer.fit(
            build_loader(run, "train", train=True), build_loader(run, "val", train=False)
        )
        result = evaluate(
            model, build_loader(run, "test", train=False), device, run.station.threshold
        )
        aucs.append(result["report"]["auc"])
        f1s.append(result["report"]["f1"])
        _LOG.info("shots %d seed %d auc %.4f", shots, seed, result["report"]["auc"])
    return {
        "shots": shots,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
    }
