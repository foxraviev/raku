"""Interpretable Prediction Calibration.

Ref: Sec. III.D, Eq. (21)-(26). Cosine logits with a learnable temperature feed
a sigmoid head; concept activations supply a class-consistency confidence that
is blended with the predicted probability.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_EPS = 1e-8


class CalibratedHead(nn.Module):
    tau: Tensor
    indicator: Tensor
    lam: Tensor

    def __init__(self, indicator: Tensor, tau_learn_init: float, lambda_init: float) -> None:
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(float(tau_learn_init)))
        self.register_buffer("indicator", indicator)
        self.register_buffer("lam", torch.tensor(float(lambda_init)))

    def logits(self, f_cls: Tensor, text: Tensor) -> Tensor:
        f = F.normalize(f_cls, dim=-1).unsqueeze(1)
        t = F.normalize(text, dim=-1)
        cos = (f * t).sum(dim=-1)
        return self.tau * cos

    def concept_confidence(self, activation: Tensor) -> Tensor:
        num = activation @ self.indicator.t()
        den = activation.sum(dim=-1, keepdim=True) + _EPS
        return num / den

    def forward(self, f_cls: Tensor, text: Tensor, activation: Tensor) -> dict[str, Tensor]:
        z = self.logits(f_cls, text)
        prob = torch.sigmoid(z)
        concept_conf = self.concept_confidence(activation)
        conf = self.lam * prob + (1.0 - self.lam) * concept_conf
        return {"logit": z, "prob": prob, "concept_conf": concept_conf, "conf": conf}

    def set_lambda(self, value: float) -> None:
        self.lam.fill_(float(value))
