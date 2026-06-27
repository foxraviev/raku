from __future__ import annotations

import torch

from raku.bench.glaze_io import set_seed
from raku.bench.settings import Settings
from raku.wheel.net import RareEyeVLM, build_model


def test_forward_output_shapes(tiny_model: RareEyeVLM, batch: torch.Tensor) -> None:
    out = tiny_model(batch)
    assert out["logit"].shape == (4, 8)
    assert out["prob"].shape == (4, 8)
    assert out["conf"].shape == (4, 8)
    assert out["activation"].shape == (4, 66)
    assert (out["prob"] >= 0).all() and (out["prob"] <= 1).all()


def test_backbone_is_frozen_and_head_trains(
    tiny_model: RareEyeVLM, batch: torch.Tensor
) -> None:
    frozen = [p for n, p in tiny_model.named_parameters() if n.startswith("encoders")]
    assert frozen and all(not p.requires_grad for p in frozen)
    tiny_model(batch)["prob"].sum().backward()
    grads = [p.grad is not None for p in (tiny_model.prompts.context, tiny_model.head.tau)]
    assert all(grads)


def test_forward_is_deterministic_in_eval(tiny_settings: Settings, batch: torch.Tensor) -> None:
    set_seed(0)
    model = build_model(tiny_settings.body, "odir5k", 8).eval()
    a = model(batch)["logit"]
    b = model(batch)["logit"]
    assert torch.allclose(a, b)


def test_concept_activation_matches_forward(
    tiny_model: RareEyeVLM, batch: torch.Tensor
) -> None:
    tiny_model.eval()
    direct = tiny_model.concept_activation(batch)
    full = tiny_model(batch)["activation"]
    assert torch.allclose(direct, full, atol=1e-6)
