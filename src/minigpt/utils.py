"""Small reproducibility and hardware helpers shared by all CLIs."""

import logging
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and all available PyTorch device generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def select_device(requested: str = "auto") -> torch.device:
    """Prefer CUDA, then Apple Metal, then CPU; reject unavailable explicit devices."""
    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    if requested == "auto":
        return torch.device(next((name for name, ready in available.items() if ready), "cpu"))
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unknown device: {requested}")
    if requested in available and not available[requested]:
        raise ValueError(f"requested device is unavailable: {requested}")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    """Wait for outstanding accelerator work before reading a wall clock."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def count_parameters(model: nn.Module) -> int:
    """Count unique trainable parameters (tied tensors count once)."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def git_hash() -> str:
    """Return the package checkout's commit, or 'unknown' outside Git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment(device: torch.device) -> dict[str, Any]:
    """Record hardware and software details needed to interpret timing results."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": str(device),
        "num_threads": torch.get_num_threads(),
    }


def setup_logging() -> None:
    """Configure concise console logging for command-line programs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
