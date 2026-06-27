"""Optimizer and warmup-cosine schedule.

Ref: Appendix A.B (Table 12) — AdamW, lr 1e-4, weight decay 0.01, cosine
annealing with a 5-epoch warmup over 100 epochs.
"""

from __future__ import annotations

import math

from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from raku.bench.settings import KilnCfg


def build_optimizer(model: nn.Module, cfg: KilnCfg) -> Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))


def build_scheduler(optimizer: Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    floor = max(total_steps, warmup_steps + 1)

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, floor - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, factor)
