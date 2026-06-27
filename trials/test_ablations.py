from __future__ import annotations

import dataclasses

import torch

from raku.bench.glaze_io import set_seed
from raku.bench.settings import Settings
from raku.wheel.net import build_model


def _build(settings: Settings, **body: object):
    set_seed(0)
    cfg = dataclasses.replace(settings.body, **body)
    return build_model(cfg, "odir5k", 8)


def test_disabling_ipc_collapses_confidence_to_probability(
    tiny_settings: Settings, batch: torch.Tensor
) -> None:
    model = _build(tiny_settings, use_ipc=False).eval()
    out = model(batch)
    assert torch.allclose(out["conf"], out["prob"], atol=1e-6)


def test_hcdm_toggle_changes_predictions(tiny_settings: Settings, batch: torch.Tensor) -> None:
    full = _build(tiny_settings, use_hcdm=True).eval()(batch)["logit"]
    flat = _build(tiny_settings, use_hcdm=False).eval()(batch)["logit"]
    assert not torch.allclose(full, flat)


def test_apsn_toggle_uses_static_prompt(tiny_settings: Settings, batch: torch.Tensor) -> None:
    model = _build(tiny_settings, use_apsn=False).eval()
    out = model(batch)
    assert out["logit"].shape == (4, 8)
    assert torch.isfinite(out["logit"]).all()


def test_baseline_differs_from_full_model(tiny_settings: Settings, batch: torch.Tensor) -> None:
    full = _build(tiny_settings).eval()(batch)["conf"]
    base = _build(tiny_settings, use_hcdm=False, use_apsn=False, use_ipc=False).eval()(batch)[
        "conf"
    ]
    assert not torch.allclose(full, base)
