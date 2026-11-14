"""Full validation metrics, training curves, and traceable baseline comparisons."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.nn import functional as F

from minigpt.baselines import bigram_loss, loss_metrics, uniform_loss
from minigpt.config import Config
from minigpt.data import load_corpus
from minigpt.model import GPT
from minigpt.train import load_model
from minigpt.utils import environment, git_hash, select_device, setup_logging


@torch.no_grad()
def full_validation_loss(model: GPT, tokens: Tensor, batch_size: int = 16) -> float:
    """Score every validation target once, resetting context at non-overlapping block boundaries.

    Each window contains up to block_size inputs and their shifted targets; the
    next window starts with the preceding target as its first input. There is no
    train/validation overlap. Short windows are token-weighted, not batch-weighted.
    """
    if batch_size <= 0 or len(tokens) < 2:
        raise ValueError("positive batch_size and at least two validation tokens required")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    block_size = model.config.block_size
    windows = (len(tokens) - 1) // block_size
    total_loss = 0.0
    try:
        for start in range(0, windows, batch_size):
            count = min(batch_size, windows - start)
            positions = torch.arange(start, start + count)[:, None] * block_size
            positions = positions + torch.arange(block_size)
            x, y = tokens[positions].to(device), tokens[positions + 1].to(device)
            logits, _ = model(x)
            total_loss += F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum").item()
        tail = windows * block_size
        if tail < len(tokens) - 1:
            x, y = tokens[None, tail:-1].to(device), tokens[None, tail + 1 :].to(device)
            logits, _ = model(x)
            total_loss += F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum").item()
    finally:
        model.train(was_training)
    return total_loss / (len(tokens) - 1)


def plot_losses(history: list[dict[str, Any]], path: str | Path) -> None:
    """Save a training/validation curve using only recorded evaluation points."""
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.plot(
        [row["step"] for row in history], [row["train_loss"] for row in history], label="Train"
    )
    axis.plot(
        [row["step"] for row in history], [row["val_loss"] for row in history], label="Validation"
    )
    axis.set(
        xlabel="Optimizer steps",
        ylabel="Cross entropy (nats / character)",
        title="Tiny Shakespeare · sampled evaluation batches",
    )
    axis.legend()
    axis.grid(alpha=0.2)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def evaluate_checkpoint(checkpoint_path: str | Path, device: torch.device) -> dict[str, Any]:
    """Score a checkpoint against its original corpus and include complete provenance."""
    model, checkpoint = load_model(checkpoint_path, device)
    config = Config.from_dict(checkpoint["config"])
    torch.set_num_threads(config.train.num_threads)
    corpus = load_corpus(config.train.data_path, config.train.train_fraction)
    if corpus.sha256 != checkpoint["data_sha256"]:
        raise ValueError("evaluation corpus differs from checkpoint")
    value = full_validation_loss(model, corpus.val, config.train.batch_size)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint["step"],
        "config": checkpoint["config"],
        "training_git_commit": checkpoint["git_commit"],
        "evaluation_git_commit": git_hash(),
        "environment": environment(device),
        "data_sha256": corpus.sha256,
        "validation_targets": len(corpus.val) - 1,
        "protocol": "all validation targets once, context reset each block_size inputs",
        "model": loss_metrics(value),
        "uniform": loss_metrics(uniform_loss(corpus.tokenizer.vocab_size)),
        "bigram": loss_metrics(bigram_loss(corpus.train, corpus.val, corpus.tokenizer.vocab_size)),
        "history": checkpoint["history"],
    }


def main() -> None:
    """Write full validation results and a loss curve for a trusted local checkpoint."""
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/small/best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="reports/metrics.json")
    parser.add_argument("--plot", default="reports/loss_curve.png")
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, select_device(args.device))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_losses(result["history"], args.plot)
    logging.info("Full validation: %s", result["model"])


if __name__ == "__main__":
    main()
