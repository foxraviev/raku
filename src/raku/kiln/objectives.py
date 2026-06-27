"""Training objective: focal classification + concept alignment + consistency.

Ref: Sec. III.E, Eq. (27)-(30).
"""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LossParts(TypedDict):
    total: Tensor
    cls: Tensor
    align: Tensor
    consist: Tensor


def focal_multilabel(logits: Tensor, targets: Tensor, gamma: float) -> Tensor:
    prob = torch.sigmoid(logits)
    log_pos = F.logsigmoid(logits)
    log_neg = F.logsigmoid(-logits)
    pos = targets * (1.0 - prob).pow(gamma) * log_pos
    neg = (1.0 - targets) * prob.pow(gamma) * log_neg
    return -(pos + neg).mean()


class Alignment(nn.Module):
    weighted: Tensor

    def __init__(self, indicator: Tensor, class_weights: Tensor, tau_alpha: float) -> None:
        super().__init__()
        self.tau_alpha = tau_alpha
        self.register_buffer("weighted", indicator * class_weights.unsqueeze(-1))

    def targets(self, labels: Tensor) -> Tensor:
        weighted = self.weighted.to(labels.dtype)
        return (labels.unsqueeze(-1) * weighted.unsqueeze(0)).amax(dim=1).clamp(0.0, 1.0)

    def forward(self, activation: Tensor, labels: Tensor) -> Tensor:
        target = self.targets(labels)
        scaled = torch.sigmoid(activation / self.tau_alpha)
        return F.binary_cross_entropy(scaled, target, reduction="mean")


def consistency(activation: Tensor, activation_pair: Tensor) -> Tensor:
    return (activation - activation_pair).pow(2).sum(dim=-1).mean()


class Objective(nn.Module):
    def __init__(
        self,
        indicator: Tensor,
        class_weights: Tensor,
        tau_alpha: float,
        gamma: float,
        beta_align: float,
        beta_consist: float,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.beta_align = beta_align
        self.beta_consist = beta_consist
        self.align = Alignment(indicator, class_weights, tau_alpha)

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        activation: Tensor,
        activation_pair: Tensor,
    ) -> LossParts:
        cls = focal_multilabel(logits, labels, self.gamma)
        align = self.align(activation, labels)
        consist = consistency(activation, activation_pair)
        total = cls + self.beta_align * align + self.beta_consist * consist
        return {"total": total, "cls": cls, "align": align, "consist": consist}
