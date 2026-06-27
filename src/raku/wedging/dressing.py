"""Augmentation pipeline.

Ref: Appendix C — RandomResizedCrop(0.8-1.0, 0.9-1.1), h-flip 0.5,
ColorJitter(0.2/0.2/0.2/0.1), RandAugment(N=2, M=9), ImageNet normalisation.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from PIL.Image import Image
from torchvision import transforms

from raku.bench.settings import SeamCfg

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

Transform = Callable[[Image], torch.Tensor]


def train_transform(cfg: SeamCfg, size: int) -> Transform:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size,
                scale=(cfg.crop_scale_low, cfg.crop_scale_high),
                ratio=(0.9, 1.1),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(cfg.jitter, cfg.jitter, cfg.jitter, cfg.hue),
            transforms.RandAugment(num_ops=cfg.randaug_n, magnitude=cfg.randaug_m),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ]
    )


def eval_transform(size: int) -> Transform:
    return transforms.Compose(
        [
            transforms.Resize(int(size * 1.14), antialias=True),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ]
    )
