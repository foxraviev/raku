"""Multi-label fundus datasets: on-disk manifest and a deterministic stand-in."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TypedDict

import torch
from PIL import Image
from torch.utils.data import Dataset

from raku.wedging.dressing import Transform


class Sample(TypedDict):
    view: torch.Tensor
    pair: torch.Tensor
    label: torch.Tensor


def _parse_bits(token: str, num_classes: int) -> torch.Tensor:
    token = token.strip()
    if len(token) == num_classes and set(token) <= {"0", "1"}:
        return torch.tensor([float(c) for c in token])
    out = torch.zeros(num_classes)
    for piece in token.replace("|", " ").split():
        out[int(piece)] = 1.0
    return out


class ManifestFundus(Dataset[Sample]):
    def __init__(
        self,
        manifest: str | Path,
        root: str | Path,
        num_classes: int,
        primary: Transform,
        secondary: Transform | None = None,
    ) -> None:
        self._root = Path(root)
        self._num_classes = num_classes
        self._primary = primary
        self._secondary = secondary if secondary is not None else primary
        self._rows: list[tuple[str, torch.Tensor]] = []
        with Path(manifest).open(newline="", encoding="utf-8") as handle:
            for line in csv.reader(handle):
                if not line or line[0].startswith("#"):
                    continue
                self._rows.append((line[0], _parse_bits(line[1], num_classes)))

    def __len__(self) -> int:
        return len(self._rows)

    def labels(self) -> torch.Tensor:
        return torch.stack([label for _, label in self._rows])

    def __getitem__(self, index: int) -> Sample:
        name, label = self._rows[index]
        with Image.open(self._root / name) as raw:
            image = raw.convert("RGB")
        return {"view": self._primary(image), "pair": self._secondary(image), "label": label}


class StandInFundus(Dataset[Sample]):
    def __init__(
        self,
        length: int,
        num_classes: int,
        size: int,
        primary: Transform,
        secondary: Transform | None = None,
        seed: int = 0,
    ) -> None:
        self._length = length
        self._num_classes = num_classes
        self._size = size
        self._primary = primary
        self._secondary = secondary if secondary is not None else primary
        gen = torch.Generator().manual_seed(seed)
        self._labels = (torch.rand(length, num_classes, generator=gen) > 0.6).float()
        self._labels[:, 0] = (self._labels[:, 1:].sum(dim=1) == 0).float()
        self._tint = torch.rand(num_classes, 3, generator=gen)

    def __len__(self) -> int:
        return self._length

    def labels(self) -> torch.Tensor:
        return self._labels

    def _paint(self, index: int) -> Image.Image:
        gen = torch.Generator().manual_seed(1000 + index)
        canvas = torch.rand(3, self._size, self._size, generator=gen) * 0.2
        active = self._labels[index].nonzero(as_tuple=False).flatten().tolist()
        band = max(1, self._size // max(1, self._num_classes))
        for slot, cls in enumerate(active):
            lo = (slot * band) % self._size
            canvas[:, lo : lo + band, :] += self._tint[cls].view(3, 1, 1)
        pixels = (canvas.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
        return Image.fromarray(pixels, mode="RGB")

    def __getitem__(self, index: int) -> Sample:
        image = self._paint(index)
        return {
            "view": self._primary(image),
            "pair": self._secondary(image),
            "label": self._labels[index],
        }
