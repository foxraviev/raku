"""Determinism, logging and atomic checkpoint writes."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

_LOG_READY = False


def kiln_log(name: str) -> logging.Logger:
    global _LOG_READY
    if not _LOG_READY:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _LOG_READY = True
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_kiln(path: str | Path, payload: dict[str, Any]) -> None:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    torch.save(payload, tmp)
    os.replace(tmp, dst)


def load_kiln(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)
