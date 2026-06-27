"""CoOp-style prompt assembly with an instance-specific dynamic token.

Ref: Sec. III.C.4, Eq. (19)-(20). Shared learnable context tokens are
concatenated with the per-image dynamic prompt and the frozen class-name word
embeddings before the frozen text layers consume the sequence.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PromptAssembler(nn.Module):
    class_words: Tensor
    class_lens: Tensor

    def __init__(self, dim: int, prompt_len: int, class_words: list[Tensor]) -> None:
        super().__init__()
        self.dim = dim
        self.prompt_len = prompt_len
        self.num_classes = len(class_words)
        self.context = nn.Parameter(torch.empty(prompt_len, dim))
        nn.init.normal_(self.context, std=0.02)
        max_words = max(t.shape[0] for t in class_words)
        words = torch.zeros(self.num_classes, max_words, dim)
        lens = torch.zeros(self.num_classes, dtype=torch.long)
        for i, tensor in enumerate(class_words):
            words[i, : tensor.shape[0]] = tensor
            lens[i] = tensor.shape[0]
        self.register_buffer("class_words", words)
        self.register_buffer("class_lens", lens)
        self._seq_len = prompt_len + 1 + max_words

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def build(self, p_dyn: Tensor) -> tuple[Tensor, Tensor]:
        b = p_dyn.shape[0]
        c = self.num_classes
        seq = torch.zeros(b, c, self._seq_len, self.dim, device=p_dyn.device, dtype=p_dyn.dtype)
        mask = torch.zeros(b, c, self._seq_len, device=p_dyn.device)
        ctx = self.context.view(1, 1, self.prompt_len, self.dim)
        seq[:, :, : self.prompt_len] = ctx
        mask[:, :, : self.prompt_len] = 1.0
        seq[:, :, self.prompt_len] = p_dyn.unsqueeze(1)
        mask[:, :, self.prompt_len] = 1.0
        words = self.class_words.to(p_dyn.dtype).unsqueeze(0).expand(b, -1, -1, -1)
        seq[:, :, self.prompt_len + 1 :] = words
        offset = self.prompt_len + 1
        arange = torch.arange(self.class_words.shape[1], device=p_dyn.device)
        valid = arange.view(1, -1) < self.class_lens.view(-1, 1)
        mask[:, :, offset:] = valid.unsqueeze(0).to(mask.dtype)
        return seq, mask
