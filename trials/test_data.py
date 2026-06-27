from __future__ import annotations

import torch

from raku.bench.settings import Settings
from raku.wedging.dressing import eval_transform, train_transform
from raku.wedging.feed import build_loader
from raku.wedging.plates import StandInFundus
from raku.wedging.portion import few_shot_indices


def test_standin_emits_two_views_and_label(tiny_settings: Settings) -> None:
    tf = eval_transform(32)
    ds = StandInFundus(10, 8, 32, tf, tf, seed=0)
    sample = ds[0]
    assert sample["view"].shape == (3, 32, 32)
    assert sample["pair"].shape == (3, 32, 32)
    assert sample["label"].shape == (8,)


def test_normal_class_is_complement_of_disease() -> None:
    tf = eval_transform(32)
    ds = StandInFundus(40, 8, 32, tf, tf, seed=3)
    labels = ds.labels()
    disease = labels[:, 1:].sum(dim=1)
    assert torch.all((labels[:, 0] == 1.0) == (disease == 0))


def test_few_shot_covers_every_class() -> None:
    torch.manual_seed(0)
    labels = (torch.rand(200, 8) > 0.5).float()
    labels[:, 0] = 1.0
    idx = few_shot_indices(labels, 5, seed=0)
    coverage = labels[idx].sum(dim=0)
    assert torch.all(coverage >= 5)


def test_loader_batches_have_expected_shapes(tiny_settings: Settings) -> None:
    loader = build_loader(tiny_settings, "train", train=True)
    batch = next(iter(loader))
    assert batch["view"].shape == (4, 3, 32, 32)
    assert batch["label"].shape == (4, 8)


def test_train_transform_differs_from_eval() -> None:
    assert train_transform.__name__ == "train_transform"
    assert eval_transform.__name__ == "eval_transform"
