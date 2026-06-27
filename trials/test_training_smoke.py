from __future__ import annotations

import torch

from raku.bench.glaze_io import set_seed
from raku.bench.settings import load_settings
from raku.firing.trainer import Trainer
from raku.wedging.feed import build_loader
from raku.wheel.net import build_model


def test_short_training_run_drives_loss_down() -> None:
    set_seed(0)
    settings = load_settings("benchtops/kiln/_bench.toml")
    device = torch.device("cpu")
    model = build_model(settings.body, settings.seam.dataset, settings.seam.num_classes)
    trainer = Trainer(model, settings, device)
    loader = build_loader(settings, "train", train=True)
    batch = next(iter(loader))
    first = trainer._step(batch)["total"].item()
    for _ in range(30):
        trainer.optimizer.zero_grad(set_to_none=True)
        loss = trainer._step(batch)["total"]
        loss.backward()
        trainer.optimizer.step()
    later = trainer._step(batch)["total"].item()
    assert later < first
