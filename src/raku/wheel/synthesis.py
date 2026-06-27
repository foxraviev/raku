"""Adaptive Prompt Synthesis Network.

Ref: Sec. III.C, Eq. (8)-(18). Cross-modal attention grounds visual tokens in
concept embeddings, concept-attended features are pooled per hierarchy level and
fused through a sigmoid gate into one instance-specific prompt token.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FeatureEnhancer(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, feats: Tensor) -> Tensor:
        return feats + self.ffn(self.norm(feats))


class CrossModalHead(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("embedding dim must divide cross-modal heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)

    def forward(self, vis: Tensor, concept: Tensor) -> tuple[Tensor, Tensor]:
        b, length, _ = vis.shape
        m = concept.shape[0]
        q = self.q(vis).view(b, length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(concept).view(m, self.heads, self.head_dim).permute(1, 0, 2)
        v = vis.view(b, length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        score = torch.einsum("bhld,hmd->bhlm", q, k) * self.scale
        attn = torch.softmax(score, dim=-1)
        vbar = torch.einsum("bhlm,bhld->bhmd", attn, v).permute(0, 2, 1, 3).reshape(b, m, -1)
        return vbar, attn.mean(dim=1)


class PromptSynthesis(nn.Module):
    def __init__(
        self, dim: int, heads: int, n_anat: int, n_path: int, n_sev: int, ffn_hidden: int
    ) -> None:
        super().__init__()
        self.enhance = FeatureEnhancer(dim, ffn_hidden)
        self.anat_head = CrossModalHead(dim, heads)
        self.path_head = CrossModalHead(dim, heads)
        self.sev_head = CrossModalHead(dim, heads)
        self.pool_anat = nn.Parameter(torch.randn(n_anat) * 0.02)
        self.pool_path = nn.Parameter(torch.randn(n_path) * 0.02)
        self.pool_sev = nn.Parameter(torch.randn(n_sev) * 0.02)
        self.gate = nn.Linear(3 * dim, dim)
        self.fuse_anat = nn.Linear(dim, dim, bias=False)
        self.fuse_path = nn.Linear(dim, dim, bias=False)
        self.fuse_sev = nn.Linear(dim, dim, bias=False)

    @staticmethod
    def _pool(vbar: Tensor, weight: Tensor) -> Tensor:
        coeff = torch.softmax(weight, dim=0).view(1, -1, 1)
        return (coeff * vbar).sum(dim=1)

    def synthesize(
        self, enhanced: Tensor, anat: Tensor, path: Tensor, sev: Tensor
    ) -> tuple[Tensor, Tensor]:
        va_bar, a_attn = self.anat_head(enhanced, anat)
        vp_bar, p_attn = self.path_head(enhanced, path)
        vs_bar, s_attn = self.sev_head(enhanced, sev)
        v_anat = self._pool(va_bar, self.pool_anat)
        v_path = self._pool(vp_bar, self.pool_path)
        v_sev = self._pool(vs_bar, self.pool_sev)
        gate = torch.sigmoid(self.gate(torch.cat([v_anat, v_path, v_sev], dim=-1)))
        p_dyn = (
            gate * self.fuse_anat(v_anat)
            + (1.0 - gate) * self.fuse_path(v_path)
            + self.fuse_sev(v_sev)
        )
        activation = torch.cat(
            [a_attn.max(dim=1).values, p_attn.max(dim=1).values, s_attn.max(dim=1).values],
            dim=-1,
        )
        return p_dyn, activation
