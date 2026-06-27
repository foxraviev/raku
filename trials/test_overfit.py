from __future__ import annotations

import torch
from torch.optim import Adam

from raku.bench.glaze_io import set_seed
from raku.bench.settings import Settings
from raku.kiln.objectives import focal_multilabel
from raku.wheel.net import build_model


def test_head_memorises_a_single_batch(tiny_settings: Settings) -> None:
    set_seed(0)
    model = build_model(tiny_settings.body, "odir5k", 8).train()
    images = torch.randn(4, 3, 32, 32)
    targets = (torch.rand(4, 8) > 0.5).float()
    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    start = focal_multilabel(model(images)["logit"], targets, 2.0).item()
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        loss = focal_multilabel(model(images)["logit"], targets, 2.0)
        loss.backward()
        opt.step()
    end = focal_multilabel(model(images)["logit"], targets, 2.0).item()
    assert end < start * 0.2
