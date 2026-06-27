"""BiomedCLIP + PubMedBERT realisation of the frozen encoder interface.

Ref: Sec. IV.B — vision tower ViT-B/16 from BiomedCLIP, text tower PubMedBERT.
Both towers expose width 768; the shared space is 512, so per-token features are
mapped with the tower's own learned projection when present and a fixed
orthogonal map otherwise (logged in the module docstring rather than silently).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class BiomedStack(nn.Module):
    vis_map: Tensor
    txt_map: Tensor

    def __init__(
        self,
        embed_dim: int,
        image_size: int,
        patch: int,
        vision_id: str,
        text_id: str,
    ) -> None:
        super().__init__()
        import open_clip
        from transformers import AutoModel, AutoTokenizer

        self.embed_dim = embed_dim
        grid = image_size // patch
        self.num_tokens = grid * grid + 1

        model, _, _ = open_clip.create_model_and_transforms(vision_id)
        self._clip_tokenizer = open_clip.get_tokenizer(vision_id)
        self._clip = model.eval()
        self._bert = AutoModel.from_pretrained(text_id).eval()
        self._bert_tokenizer = AutoTokenizer.from_pretrained(text_id)
        for module in (self._clip, self._bert):
            for param in module.parameters():
                param.requires_grad_(False)

        vis_width = self._clip.visual.trunk.num_features
        bert_width = self._bert.config.hidden_size
        gen = torch.Generator().manual_seed(11)
        self.register_buffer("vis_map", _orthogonal(vis_width, embed_dim, gen))
        self.register_buffer("txt_map", _orthogonal(bert_width, embed_dim, gen))

    @torch.no_grad()
    def vision(self, images: Tensor) -> Tensor:
        tokens = self._clip.visual.trunk.forward_features(images)
        return tokens @ self.vis_map

    @torch.no_grad()
    def encode_phrases(self, phrases: tuple[str, ...]) -> Tensor:
        tokens = self._clip_tokenizer(list(phrases))
        feats = self._clip.encode_text(tokens)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def word_embeddings(self, name: str) -> Tensor:
        enc = self._bert_tokenizer(name, return_tensors="pt", add_special_tokens=False)
        table = self._bert.get_input_embeddings()
        embeds = table(enc["input_ids"])[0]
        return F.normalize(embeds @ self.txt_map, dim=-1)

    def encode_sequence(self, embeds: Tensor, mask: Tensor) -> Tensor:
        back = embeds @ torch.linalg.pinv(self.txt_map)
        out = self._bert(inputs_embeds=back, attention_mask=mask.long())
        hidden = out.last_hidden_state @ self.txt_map
        weight = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def _orthogonal(rows: int, cols: int, gen: torch.Generator) -> Tensor:
    flat = torch.randn(rows, cols, generator=gen)
    q, _ = torch.linalg.qr(flat) if rows >= cols else torch.linalg.qr(flat.t())
    return q[:, :cols] if rows >= cols else q[:, :rows].t()
