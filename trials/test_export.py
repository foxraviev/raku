from __future__ import annotations

import warnings
from pathlib import Path

import torch

from raku.bench.settings import Settings
from raku.wheel.export import export_onnx
from raku.wheel.net import build_model


def test_onnx_export_runs(tiny_settings: Settings, tmp_path: Path) -> None:
    model = build_model(tiny_settings.body, "odir5k", 8)
    target = tmp_path / "head.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = export_onnx(model, target, tiny_settings.body.image_size)
    assert path.exists() and path.stat().st_size > 0


def test_onnx_parity_when_runtime_available(tiny_settings: Settings, tmp_path: Path) -> None:
    ort = __import__("importlib").util.find_spec("onnxruntime")
    if ort is None:
        return
    import onnxruntime as rt  # noqa: F401

    model = build_model(tiny_settings.body, "odir5k", 8).eval()
    target = tmp_path / "head.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_onnx(model, target, tiny_settings.body.image_size)
    image = torch.randn(1, 3, tiny_settings.body.image_size, tiny_settings.body.image_size)
    with torch.no_grad():
        reference = model(image)["prob"].numpy()
    session = rt.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    got = session.run(["prob"], {"image": image.numpy()})[0]
    assert abs(reference - got).max() < 1e-3
