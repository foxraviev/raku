"""Training loop.

Ref: Alg. 1 — per batch: extract features, decompose concepts, synthesise
prompts, score classes, estimate activations and minimise the composite loss.
The two augmented views share one forward pass so distributed reduction sees
every parameter exactly once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from raku.bench import spinning
from raku.bench.glaze_io import kiln_log, save_kiln
from raku.bench.settings import Settings, as_mapping
from raku.firing.anneal import calibrate_lambda
from raku.firing.ema import WeightEma
from raku.firing.schedule import build_optimizer, build_scheduler
from raku.kiln.objectives import LossParts, Objective
from raku.slip import lexicon
from raku.wedging.plates import Sample
from raku.wheel.net import RareEyeVLM

_LOG = kiln_log("raku.firing")


def build_objective(model: RareEyeVLM, settings: Settings) -> Objective:
    indicator = model.head.indicator.detach().clone()
    weights = torch.tensor(lexicon.class_weights(model.dataset), dtype=torch.float32)
    return Objective(
        indicator,
        weights,
        settings.body.tau_alpha,
        settings.kiln.focal_gamma,
        settings.kiln.beta_align,
        settings.kiln.beta_consist,
    )


class Trainer:
    def __init__(self, model: RareEyeVLM, settings: Settings, device: torch.device) -> None:
        self.core = model.to(device)
        self.settings = settings
        self.device = device
        self.objective = build_objective(model, settings).to(device)
        self.optimizer = build_optimizer(self.core, settings.kiln)
        self.ema = WeightEma(self.core)
        self.use_amp = settings.kiln.amp and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.model: torch.nn.Module = self.core
        if spinning.is_distributed() and device.type == "cuda":
            self.model = DistributedDataParallel(self.core, device_ids=[device.index])

    def _step(self, batch: Sample) -> LossParts:
        labels = batch["label"].to(self.device)
        images = torch.cat(
            [batch["view"].to(self.device), batch["pair"].to(self.device)], dim=0
        )
        n = labels.shape[0]
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            out = self.model(images)
            parts = self.objective(
                out["logit"][:n], labels, out["activation"][:n], out["activation"][n:]
            )
        return parts

    def fit(
        self,
        train_loader: DataLoader[Sample],
        val_loader: DataLoader[Sample] | None = None,
        max_steps: int | None = None,
    ) -> None:
        steps_per_epoch = len(train_loader)
        total = max_steps or self.settings.kiln.epochs * steps_per_epoch
        warmup = self.settings.kiln.warmup_epochs * steps_per_epoch
        scheduler = build_scheduler(self.optimizer, warmup, total)
        clip = self.settings.kiln.grad_clip
        step = 0
        for epoch in range(self.settings.kiln.epochs):
            self.model.train()
            for batch in train_loader:
                self.optimizer.zero_grad(set_to_none=True)
                parts = self._step(batch)
                self.scaler.scale(parts["total"]).backward()
                if clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.core.parameters(), clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scheduler.step()
                self.ema.update(self.core)
                if step % self.settings.station.log_every == 0 and spinning.is_primary():
                    _LOG.info(
                        "epoch %d step %d loss %.4f cls %.4f align %.4f consist %.4f",
                        epoch,
                        step,
                        parts["total"].item(),
                        parts["cls"].item(),
                        parts["align"].item(),
                        parts["consist"].item(),
                    )
                step += 1
                if max_steps is not None and step >= max_steps:
                    break
            if (epoch + 1) % self.settings.station.ckpt_every == 0 and spinning.is_primary():
                self.save(
                    Path(self.settings.station.out_dir) / f"epoch_{epoch + 1}.pt", epoch + 1
                )
            if max_steps is not None and step >= max_steps:
                break
        calibrate = self.settings.station.calibrate_lambda and self.settings.body.use_ipc
        if val_loader is not None and calibrate and spinning.is_primary():
            chosen = calibrate_lambda(self.core, val_loader, self.device)
            _LOG.info("calibrated lambda = %.3f", chosen)

    def save(self, path: str | Path, epoch: int) -> None:
        payload: dict[str, Any] = {
            "model": self.core.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "epoch": epoch,
            "seed": self.settings.kiln.seed,
            "settings": as_mapping(self.settings),
        }
        save_kiln(path, payload)

    def load(self, path: str | Path) -> int:
        blob = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.core.load_state_dict(blob["model"])
        self.optimizer.load_state_dict(blob["optimizer"])
        return int(blob.get("epoch", 0))
