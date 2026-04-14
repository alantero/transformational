from __future__ import annotations

import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_autocast_dtype(device: torch.device, precision: str) -> torch.dtype | None:
    if device.type != "cuda":
        return None

    if precision == "none":
        return None
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if precision != "auto":
        raise ValueError(f"Unsupported precision mode: {precision}")

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context(device: torch.device, autocast_dtype: torch.dtype | None):
    if device.type == "cuda" and autocast_dtype is not None:
        return torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
    return torch.amp.autocast(device_type="cpu", enabled=False)


def create_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_json(path: str | os.PathLike[str], payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: str | os.PathLike[str]) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def list_checkpoints(output_dir: str) -> list[Path]:
    base = Path(output_dir)
    checkpoints = [path for path in base.glob("checkpoint-*") if path.is_dir()]
    return sorted(checkpoints, key=lambda path: int(path.name.split("-")[-1]))


def prune_checkpoints(output_dir: str, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = list_checkpoints(output_dir)
    for path in checkpoints[:-keep_last]:
        shutil.rmtree(path, ignore_errors=True)
