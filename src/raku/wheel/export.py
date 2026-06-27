"""ONNX export of the prediction path.

Ref: Sec. V.G — the trained head is portable; the export emits per-class
probability and calibrated confidence for a single fundus image.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import torch
from torch import Tensor, nn

from raku.wheel.net import RareEyeVLM


class PredictionPath(nn.Module):
    def __init__(self, model: RareEyeVLM) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        out = self.model(image)
        return out["prob"], out["conf"]


def export_onnx(model: RareEyeVLM, path: str | Path, image_size: int) -> Path:
    wrapper = PredictionPath(model).eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    target = Path(path)
    axes = {"image": {0: "batch"}, "prob": {0: "batch"}, "conf": {0: "batch"}}
    legacy = "dynamo" in inspect.signature(torch.onnx.export).parameters
    if legacy:
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(target),
            input_names=["image"],
            output_names=["prob", "conf"],
            opset_version=17,
            dynamic_axes=axes,
            dynamo=False,
        )
    else:
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(target),
            input_names=["image"],
            output_names=["prob", "conf"],
            opset_version=17,
            dynamic_axes=axes,
        )
    return target
