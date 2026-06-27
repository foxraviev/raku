from __future__ import annotations

from pathlib import Path

import torch

from raku.bench.glaze_io import load_kiln, save_kiln
from raku.bench.settings import Settings
from raku.firing.trainer import Trainer
from raku.wheel.net import build_model


def test_atomic_save_leaves_no_partial(tmp_path: Path) -> None:
    target = tmp_path / "nest" / "ckpt.pt"
    save_kiln(target, {"a": torch.zeros(3)})
    assert target.exists()
    assert not target.with_suffix(".pt.part").exists()


def test_trainer_roundtrip_restores_weights(tiny_settings: Settings, tmp_path: Path) -> None:
    model = build_model(tiny_settings.body, "odir5k", 8)
    trainer = Trainer(model, tiny_settings, torch.device("cpu"))
    path = tmp_path / "ckpt.pt"
    trainer.save(path, epoch=7)
    blob = load_kiln(path)
    assert blob["epoch"] == 7
    assert blob["seed"] == tiny_settings.kiln.seed

    fresh = build_model(tiny_settings.body, "odir5k", 8)
    restored = Trainer(fresh, tiny_settings, torch.device("cpu"))
    assert restored.load(path) == 7
    a = dict(model.named_parameters())["prompts.context"]
    b = dict(fresh.named_parameters())["prompts.context"]
    assert torch.allclose(a, b)
