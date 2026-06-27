"""Hierarchical Concept Decomposition Module.

Ref: Sec. III.B, Eq. (1)-(7). Anatomy embeddings ground pathology embeddings
through graph attention; severity embeddings are enriched by cross-attention
over the concatenated anatomical-pathological context.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GroundingAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("embedding dim must divide the number of GAT heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.proj = nn.Linear(dim, dim, bias=False)
        self.src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.dst = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.src)
        nn.init.xavier_uniform_(self.dst)

    def forward(self, path: Tensor, anat: Tensor, mask: Tensor) -> Tensor:
        mp, ma = path.shape[0], anat.shape[0]
        wp = self.proj(path).view(mp, self.heads, self.head_dim).transpose(0, 1)
        wa = self.proj(anat).view(ma, self.heads, self.head_dim).transpose(0, 1)
        score = (wp * self.src.unsqueeze(1)).sum(-1, keepdim=True) + (
            wa * self.dst.unsqueeze(1)
        ).sum(-1).unsqueeze(1)
        score = F.leaky_relu(score, 0.2)
        score = score.masked_fill(~mask.bool().unsqueeze(0), float("-inf"))
        alpha = torch.nan_to_num(torch.softmax(score, dim=-1))
        out = torch.einsum("hkj,hjd->hkd", alpha, wa)
        return out.transpose(0, 1).reshape(mp, -1)


class Decomposer(nn.Module):
    def __init__(self, dim: int, gat_layers: int, gat_heads: int) -> None:
        super().__init__()
        self.scale = math.sqrt(dim)
        self.layers = nn.ModuleList(
            GroundingAttention(dim, gat_heads) for _ in range(gat_layers)
        )
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

    def forward(
        self, anat: Tensor, path: Tensor, sev: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        hp = path
        for layer in self.layers:
            hp = F.gelu(layer(hp, anat, mask)) + hp
        context = torch.cat([anat, hp], dim=0)
        attn = torch.softmax(self.q(sev) @ self.k(context).t() / self.scale, dim=-1)
        enriched = attn @ self.v(context)
        return anat, hp, enriched
