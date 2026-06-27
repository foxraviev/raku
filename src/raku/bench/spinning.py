"""Distributed-process helpers (DDP-ready, single-process safe)."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_primary() -> bool:
    return rank() == 0


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        local = int(os.environ.get("LOCAL_RANK", "0"))
        return torch.device(f"cuda:{local}")
    return torch.device("cpu")


def maybe_init() -> None:
    if dist.is_available() and "RANK" in os.environ and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def shutdown() -> None:
    if is_distributed():
        dist.destroy_process_group()
