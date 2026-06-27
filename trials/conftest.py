from __future__ import annotations

import pytest
import torch

from raku.bench.glaze_io import set_seed
from raku.bench.settings import BodyCfg, KilnCfg, SeamCfg, Settings, StationCfg
from raku.wheel.net import RareEyeVLM, build_model


@pytest.fixture
def tiny_settings() -> Settings:
    return Settings(
        body=BodyCfg(image_size=32, patch=16, ffn_hidden=64, offline_backbone=True),
        seam=SeamCfg(dataset="odir5k", num_classes=8, batch_size=4, num_workers=0),
        kiln=KilnCfg(epochs=1, warmup_epochs=0, seed=0),
        station=StationCfg(world_size=1, log_every=1, ckpt_every=1, eval_seeds=(0,)),
    )


@pytest.fixture
def tiny_model(tiny_settings: Settings) -> RareEyeVLM:
    set_seed(0)
    return build_model(
        tiny_settings.body, tiny_settings.seam.dataset, tiny_settings.seam.num_classes
    )


@pytest.fixture
def batch() -> torch.Tensor:
    set_seed(1)
    return torch.randn(4, 3, 32, 32)
