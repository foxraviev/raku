"""Typed run configuration loaded from a TOML bench sheet.

Ref: Appendix A.B (Table 12) — every default mirrors the reported run.
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class BodyCfg:
    embed_dim: int = 512
    prompt_len: int = 16
    n_anatomical: int = 12
    n_pathological: int = 45
    n_severity: int = 9
    gat_layers: int = 2
    gat_heads: int = 4
    xmodal_heads: int = 8
    ffn_hidden: int = 2048
    dropout: float = 0.1
    tau_learn_init: float = 14.3
    tau_alpha: float = 0.1
    lambda_init: float = 0.7
    vision_id: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    text_id: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    image_size: int = 224
    patch: int = 16
    offline_backbone: bool = True
    use_hcdm: bool = True
    use_apsn: bool = True
    use_ipc: bool = True


@dataclass(frozen=True, slots=True)
class SeamCfg:
    dataset: str = "odir5k"
    root: str = "data/odir5k"
    num_classes: int = 8
    shots: int = 0
    batch_size: int = 32
    num_workers: int = 8
    crop_scale_low: float = 0.8
    crop_scale_high: float = 1.0
    jitter: float = 0.2
    hue: float = 0.1
    randaug_n: int = 2
    randaug_m: int = 9


@dataclass(frozen=True, slots=True)
class KilnCfg:
    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    epochs: int = 100
    grad_clip: float = 1.0
    focal_gamma: float = 2.0
    beta_align: float = 0.5
    beta_consist: float = 0.1
    amp: bool = False
    seed: int = 0


@dataclass(frozen=True, slots=True)
class StationCfg:
    world_size: int = 4
    out_dir: str = "firings/main"
    log_every: int = 20
    ckpt_every: int = 5
    eval_seeds: tuple[int, ...] = (0, 1, 2, 42, 2024)
    threshold: float = 0.5
    calibrate_lambda: bool = True


@dataclass(frozen=True, slots=True)
class Settings:
    body: BodyCfg = field(default_factory=BodyCfg)
    seam: SeamCfg = field(default_factory=SeamCfg)
    kiln: KilnCfg = field(default_factory=KilnCfg)
    station: StationCfg = field(default_factory=StationCfg)

    @property
    def effective_batch(self) -> int:
        return self.seam.batch_size * self.station.world_size


def _coerce(cls: type[_T], blob: Mapping[str, Any]) -> _T:
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    for key, value in blob.items():
        if key not in known:
            raise KeyError(f"unknown field '{key}' for {cls.__name__}")
        target = known[key].type
        if isinstance(value, list) and "tuple" in str(target):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_settings(path: str | Path) -> Settings:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    sections: dict[str, Any] = {}
    for name, sub in (
        ("body", BodyCfg),
        ("seam", SeamCfg),
        ("kiln", KilnCfg),
        ("station", StationCfg),
    ):
        blob = raw.get(name, {})
        if not isinstance(blob, dict):
            raise TypeError(f"section [{name}] must be a table")
        sections[name] = _coerce(sub, blob)
    return Settings(**sections)


def as_mapping(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: as_mapping(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return list(obj)
    return cast(Any, obj)
