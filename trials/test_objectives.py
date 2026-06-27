from __future__ import annotations

import torch
import torch.nn.functional as F

from raku.kiln.objectives import Alignment, consistency, focal_multilabel


def test_focal_reduces_to_bce_when_gamma_zero() -> None:
    torch.manual_seed(0)
    logits = torch.randn(5, 8)
    targets = (torch.rand(5, 8) > 0.5).float()
    focal = focal_multilabel(logits, targets, 0.0)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(focal, bce, atol=1e-6)


def test_focal_down_weights_easy_examples() -> None:
    logits = torch.tensor([[4.0]])
    targets = torch.tensor([[1.0]])
    assert focal_multilabel(logits, targets, 2.0) < focal_multilabel(logits, targets, 0.0)


def test_focal_gradient_matches_finite_difference() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
    targets = (torch.rand(3, 4) > 0.5).double()
    focal_multilabel(logits, targets, 2.0).backward()
    analytic = logits.grad.clone()
    eps = 1e-6
    numeric = torch.zeros_like(logits)
    with torch.no_grad():
        flat = logits.view(-1)
        for i in range(flat.numel()):
            flat[i] += eps
            hi = focal_multilabel(logits, targets, 2.0)
            flat[i] -= 2 * eps
            lo = focal_multilabel(logits, targets, 2.0)
            flat[i] += eps
            numeric.view(-1)[i] = (hi - lo) / (2 * eps)
    assert torch.allclose(analytic, numeric, atol=1e-5)


def test_alignment_target_uses_class_weight() -> None:
    indicator = torch.zeros(2, 66)
    indicator[0, 5] = 1.0
    indicator[1, 5] = 1.0
    weights = torch.tensor([1.0, 0.5])
    align = Alignment(indicator, weights, 0.1)
    labels = torch.tensor([[0.0, 1.0]])
    target = align.targets(labels)
    assert abs(float(target[0, 5]) - 0.5) < 1e-6


def test_consistency_is_zero_for_identical_views() -> None:
    activation = torch.rand(4, 66)
    assert float(consistency(activation, activation)) == 0.0
    assert float(consistency(activation, activation + 0.1)) > 0.0
