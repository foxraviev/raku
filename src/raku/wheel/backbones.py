"""Frozen vision/text encoders.

Ref: Sec. III.A, Eq. (1), (8) — BiomedCLIP ViT-B/16 vision tower and a
PubMedBERT text tower, both frozen. A deterministic stand-in keeps the same
interface (token grid, pooled phrase embeddings, prompt-sequence encoding) so
the trainable head can run without network access.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@runtime_checkable
class Encoders(Protocol):
    embed_dim: int
    num_tokens: int

    def vision(self, images: Tensor) -> Tensor: ...

    def encode_phrases(self, phrases: tuple[str, ...]) -> Tensor: ...

    def word_embeddings(self, name: str) -> Tensor: ...

    def encode_sequence(self, embeds: Tensor, mask: Tensor) -> Tensor: ...


def _seed_of(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _hash_vector(text: str, width: int) -> Tensor:
    gen = torch.Generator().manual_seed(_seed_of(text) % (2**63))
    return torch.randn(width, generator=gen)


class FrozenStack(nn.Module):
    patch_proj: Tensor
    cls: Tensor
    pos: Tensor
    text_proj: Tensor

    def __init__(
        self, embed_dim: int, image_size: int, patch: int, hash_width: int = 256
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        grid = image_size // patch
        self.num_tokens = grid * grid + 1
        self._patch = patch
        self._hash_width = hash_width
        gen = torch.Generator().manual_seed(20240517)
        scale = embed_dim**-0.5
        self.register_buffer(
            "patch_proj", torch.randn(embed_dim, 3 * patch * patch, generator=gen) * scale
        )
        self.register_buffer("cls", torch.randn(embed_dim, generator=gen) * scale)
        self.register_buffer(
            "pos", torch.randn(self.num_tokens, embed_dim, generator=gen) * scale
        )
        self.register_buffer(
            "text_proj", torch.randn(embed_dim, hash_width, generator=gen) * scale
        )
        layer = nn.TransformerEncoderLayer(
            embed_dim, nhead=8, dim_feedforward=embed_dim * 2, batch_first=True, dropout=0.0
        )
        self.text_layers = nn.TransformerEncoder(layer, num_layers=2)
        for param in self.text_layers.parameters():
            param.requires_grad_(False)
        self.text_layers.eval()

    def vision(self, images: Tensor) -> Tensor:
        b = images.shape[0]
        patches = F.unfold(images, kernel_size=self._patch, stride=self._patch)
        patches = patches.transpose(1, 2)
        proj = patches @ self.patch_proj.t()
        cls = self.cls.expand(b, 1, self.embed_dim)
        tokens = torch.cat([cls, proj], dim=1)
        tokens = tokens + self.pos.unsqueeze(0)
        return F.gelu(F.layer_norm(tokens, (self.embed_dim,)))

    def _phrase_vec(self, phrase: str) -> Tensor:
        raw = _hash_vector(phrase, self._hash_width).to(self.text_proj.device)
        return F.normalize(raw @ self.text_proj.t(), dim=-1)

    def encode_phrases(self, phrases: tuple[str, ...]) -> Tensor:
        return torch.stack([self._phrase_vec(p) for p in phrases])

    def word_embeddings(self, name: str) -> Tensor:
        words = name.split() or [name]
        return torch.stack([self._phrase_vec(w) for w in words])

    def encode_sequence(self, embeds: Tensor, mask: Tensor) -> Tensor:
        pad = ~mask.bool()
        hidden = self.text_layers(embeds, src_key_padding_mask=pad)
        weight = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        return pooled


def build_encoders(
    embed_dim: int,
    image_size: int,
    patch: int,
    offline: bool,
    vision_id: str,
    text_id: str,
) -> Encoders:
    if offline:
        return FrozenStack(embed_dim, image_size, patch)
    from raku.wheel.biomed import BiomedStack

    return BiomedStack(embed_dim, image_size, patch, vision_id, text_id)
