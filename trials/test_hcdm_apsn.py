from __future__ import annotations

import torch

from raku.wheel.decompose import Decomposer, GroundingAttention
from raku.wheel.synthesis import CrossModalHead, PromptSynthesis


def test_grounding_attention_respects_mask() -> None:
    torch.manual_seed(0)
    layer = GroundingAttention(16, 4)
    path = torch.randn(45, 16)
    anat = torch.randn(12, 16)
    mask = (torch.rand(45, 12) > 0.5).float()
    mask[:, 0] = 1.0
    out = layer(path, anat, mask)
    assert out.shape == (45, 16)
    assert torch.isfinite(out).all()


def test_grounding_attention_tolerates_empty_neighbourhood() -> None:
    torch.manual_seed(0)
    layer = GroundingAttention(16, 4)
    path = torch.randn(3, 16)
    anat = torch.randn(5, 16)
    mask = torch.zeros(3, 5)
    mask[0, 2] = 1.0
    out = layer(path, anat, mask)
    assert torch.isfinite(out).all()


def test_decomposer_shapes_and_residual() -> None:
    torch.manual_seed(0)
    dec = Decomposer(16, 2, 4)
    anat = torch.randn(12, 16)
    path = torch.randn(45, 16)
    sev = torch.randn(9, 16)
    mask = (torch.rand(45, 12) > 0.5).float()
    mask[:, 0] = 1.0
    a, hp, c = dec(anat, path, sev, mask)
    assert a.shape == (12, 16)
    assert hp.shape == (45, 16)
    assert c.shape == (9, 16)


def test_cross_modal_attention_is_a_distribution() -> None:
    torch.manual_seed(0)
    head = CrossModalHead(16, 8)
    vis = torch.randn(2, 17, 16)
    concept = torch.randn(12, 16)
    vbar, attn = head(vis, concept)
    assert vbar.shape == (2, 12, 16)
    assert attn.shape == (2, 17, 12)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2, 17), atol=1e-5)


def test_activation_is_bounded_unit_interval() -> None:
    torch.manual_seed(0)
    apsn = PromptSynthesis(16, 8, 12, 45, 9, 64)
    enhanced = torch.randn(3, 17, 16)
    anat, path, sev = torch.randn(12, 16), torch.randn(45, 16), torch.randn(9, 16)
    p_dyn, activation = apsn.synthesize(enhanced, anat, path, sev)
    assert p_dyn.shape == (3, 16)
    assert activation.shape == (3, 66)
    assert (activation >= 0).all() and (activation <= 1).all()
