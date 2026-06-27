"""K-shot index selection for the few-shot protocol.

Ref: Sec. IV.B — for multi-label images, every positive sample is counted for
each of its positive classes; K in {1, 2, 5, 10}.
"""

from __future__ import annotations

import torch


def few_shot_indices(labels: torch.Tensor, k: int, seed: int) -> list[int]:
    if k <= 0:
        return list(range(labels.shape[0]))
    num_classes = labels.shape[1]
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(labels.shape[0], generator=gen).tolist()
    quota = [k] * num_classes
    chosen: list[int] = []
    taken: set[int] = set()
    for idx in order:
        positives = labels[idx].nonzero(as_tuple=False).flatten().tolist()
        if not positives:
            continue
        if any(quota[c] > 0 for c in positives):
            chosen.append(idx)
            taken.add(idx)
            for c in positives:
                quota[c] = max(0, quota[c] - 1)
        if all(q == 0 for q in quota):
            break
    for idx in order:
        if all(q == 0 for q in quota):
            break
        if idx in taken:
            continue
        positives = labels[idx].nonzero(as_tuple=False).flatten().tolist()
        for c in positives:
            if quota[c] > 0:
                chosen.append(idx)
                taken.add(idx)
                quota[c] = max(0, quota[c] - 1)
                break
    return sorted(chosen)
