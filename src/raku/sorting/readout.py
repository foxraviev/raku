"""Single-image diagnosis with concept-level explanation.

Ref: Alg. 2 and Sec. III.D.4 — predicted classes above threshold are reported
with calibrated confidence and the top activated concepts per hierarchy level.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from raku.slip import lexicon
from raku.wheel.net import RareEyeVLM


@dataclass(frozen=True, slots=True)
class Diagnosis:
    disease: str
    confidence: float
    anatomical: tuple[str, ...]
    pathological: tuple[str, ...]
    severity: tuple[str, ...]


def _top(activation: torch.Tensor, names: tuple[str, ...], k: int) -> tuple[str, ...]:
    k = min(k, len(names))
    idx = torch.topk(activation, k).indices.tolist()
    return tuple(names[i] for i in idx)


@torch.no_grad()
def diagnose(
    model: RareEyeVLM,
    image: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
    top_k: int = 3,
) -> list[Diagnosis]:
    model.eval()
    out = model(image.unsqueeze(0).to(device))
    activation = out["activation"][0]
    n_a = len(lexicon.ANATOMICAL)
    n_p = len(lexicon.PATHOLOGICAL)
    anat = _top(activation[:n_a], lexicon.ANATOMICAL, top_k)
    path = _top(activation[n_a : n_a + n_p], lexicon.PATHOLOGICAL, top_k)
    sev = _top(activation[n_a + n_p :], lexicon.SEVERITY, top_k)
    names = lexicon.class_names(model.dataset)
    found: list[Diagnosis] = []
    for c, prob in enumerate(out["prob"][0].tolist()):
        if prob > threshold:
            found.append(Diagnosis(names[c], float(out["conf"][0, c].item()), anat, path, sev))
    return found
