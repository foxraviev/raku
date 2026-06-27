"""Dataset assembly and DataLoader construction."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from raku.bench import spinning
from raku.bench.settings import Settings
from raku.wedging.dressing import eval_transform, train_transform
from raku.wedging.plates import ManifestFundus, Sample, StandInFundus
from raku.wedging.portion import few_shot_indices

_STANDIN_LENGTH = {"train": 96, "val": 48, "test": 48}


def _labels_of(dataset: Dataset[Sample]) -> torch.Tensor:
    if isinstance(dataset, (ManifestFundus, StandInFundus)):
        return dataset.labels()
    raise TypeError("dataset does not expose labels()")


def build_dataset(settings: Settings, split: str, train: bool) -> Dataset[Sample]:
    seam = settings.seam
    size = settings.body.image_size
    primary = train_transform(seam, size) if train else eval_transform(size)
    secondary = train_transform(seam, size) if train else None
    manifest = Path(seam.root) / f"{split}.csv"
    base: Dataset[Sample]
    if manifest.exists():
        base = ManifestFundus(manifest, seam.root, seam.num_classes, primary, secondary)
    else:
        base = StandInFundus(
            _STANDIN_LENGTH.get(split, 48),
            seam.num_classes,
            size,
            primary,
            secondary,
            seed=settings.kiln.seed + hash(split) % 97,
        )
    if train and seam.shots > 0:
        keep = few_shot_indices(_labels_of(base), seam.shots, settings.kiln.seed)
        base = Subset(base, keep)
    return base


def build_loader(settings: Settings, split: str, train: bool) -> DataLoader[Sample]:
    dataset = build_dataset(settings, split, train)
    sampler: DistributedSampler[Sample] | None = None
    if spinning.is_distributed():
        sampler = DistributedSampler(dataset, shuffle=train, drop_last=train)
    return DataLoader(
        dataset,
        batch_size=settings.seam.batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=settings.seam.num_workers,
        drop_last=train and sampler is None,
        pin_memory=torch.cuda.is_available(),
    )
