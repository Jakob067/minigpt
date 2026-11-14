"""Readable single-device training with complete checkpoint and RNG restoration."""

import argparse
import csv
import json
import logging
import math
import random
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from minigpt.config import Config, TrainConfig, load_config, save_config
from minigpt.data import Corpus, download_shakespeare, get_batch, load_corpus
from minigpt.model import GPT
from minigpt.utils import (
    count_parameters,
    environment,
    git_hash,
    seed_everything,
    select_device,
    setup_logging,
    synchronize,
)

LOGGER = logging.getLogger(__name__)


def learning_rate(step: int, config: TrainConfig) -> float:
    """Warm up linearly, then decay to min_lr on the final scheduled update."""
    if step < config.warmup_iters:
        return config.learning_rate * (step + 1) / config.warmup_iters
    denominator = max(1, config.max_iters - 1 - config.warmup_iters)
    progress = min(1.0, max(0.0, (step - config.warmup_iters) / denominator))
    return config.min_lr + 0.5 * (config.learning_rate - config.min_lr) * (
        1 + math.cos(math.pi * progress)
    )


def make_optimizer(model: GPT, config: TrainConfig) -> torch.optim.AdamW:
    """Decay only matrix parameters; exclude all biases and LayerNorm scales."""
    parameters = list(model.parameters())
    groups = [
        {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": config.weight_decay},
        {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.learning_rate, betas=(config.beta1, config.beta2))


def precision_context(config: TrainConfig, device: torch.device) -> AbstractContextManager:
    """Use optional BF16 only on CUDA hardware that explicitly supports it."""
    if config.precision == "float32":
        return nullcontext()
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise ValueError("bfloat16 requires a CUDA device with BF16 support; use float32 here")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


@torch.no_grad()
def estimate_losses(model: GPT, corpus: Corpus, config: TrainConfig) -> dict[str, float]:
    """Average reproducible random batches without consuming the training RNG stream."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(config.seed + 1)
    result = {}
    try:
        for name, tokens in (("train", corpus.train), ("val", corpus.val)):
            total = 0.0
            for _ in range(config.eval_batches):
                x, y = get_batch(
                    tokens, config.batch_size, model.config.block_size, device, generator
                )
                with precision_context(config, device):
                    _, loss = model(x, y)
                assert loss is not None
                total += loss.item()
            result[name] = total / config.eval_batches
    finally:
        model.train(was_training)
    return result


def rng_state(generator: torch.Generator, device: torch.device) -> dict[str, Any]:
    """Capture Python, NumPy, PyTorch CPU, sampler, and active accelerator states."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "batch": generator.get_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state: dict[str, Any], generator: torch.Generator, device: torch.device) -> None:
    """Restore states after constructing/loading the model and optimizer."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["batch"].cpu())
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    elif device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"].cpu())


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    """Save to a temporary sibling first, preserving the previous checkpoint on failure."""
    destination = Path(path)
    temporary = destination.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a trusted local checkpoint (contains Python/NumPy RNG state via pickle)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model(path: str | Path, device: torch.device) -> tuple[GPT, dict[str, Any]]:
    """Restore an evaluation model with its resolved configuration and provenance."""
    checkpoint = load_checkpoint(path)
    config = Config.from_dict(checkpoint["config"])
    model = GPT(config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


def _write_logs(out_dir: Path, metadata: dict[str, Any], history: list[dict[str, Any]]) -> None:
    (out_dir / "metrics.json").write_text(
        json.dumps({**metadata, "history": history}, indent=2),
        encoding="utf-8",
    )
    if history:
        with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)


def run_training(
    config: Config,
    resume: str | Path | None = None,
    stop_after: int | None = None,
) -> dict[str, Any]:
    """Train up to max_iters (or an absolute stop_after), retaining the original LR horizon.

    A resumed run must keep every setting except out_dir unchanged. CPU float32
    continuation is tested bit-for-bit, including dropout and optimizer moments.
    Accelerator kernels may differ across hardware/software and are not claimed exact.
    """
    torch.set_num_threads(config.train.num_threads)
    seed_everything(config.train.seed)
    device = select_device(config.train.device)
    with precision_context(config.train, device):
        pass
    path = Path(config.train.data_path)
    if not path.exists() and path == Path("data/input.txt"):
        download_shakespeare(path)
    corpus = load_corpus(path, config.train.train_fraction)
    if config.model.vocab_size != corpus.tokenizer.vocab_size:
        raise ValueError(f"config vocab_size must match corpus: {corpus.tokenizer.vocab_size}")
    if min(len(corpus.train), len(corpus.val)) <= config.model.block_size:
        raise ValueError("both corpus splits must be longer than block_size")
    out_dir = Path(config.train.out_dir)
    if not resume and (out_dir / "last.pt").exists():
        raise FileExistsError(
            f"{out_dir} already has a checkpoint; use --resume or a new --out-dir"
        )
    model = GPT(config.model).to(device)
    optimizer = make_optimizer(model, config.train)
    generator = torch.Generator().manual_seed(config.train.seed)
    step, best_loss, elapsed_before = 0, float("inf"), 0.0
    history: list[dict[str, Any]] = []
    metadata = {
        "config": config.to_dict(),
        "git_commit": git_hash(),
        "environment": environment(device),
        "data_sha256": corpus.sha256,
        "parameters": count_parameters(model),
        "command": __import__("sys").argv,
    }
    if resume:
        checkpoint = load_checkpoint(resume)
        original = Config.from_dict(checkpoint["config"])
        original.train.out_dir = config.train.out_dir
        if original.to_dict() != config.to_dict():
            raise ValueError("resume requires the original config and LR horizon (except out_dir)")
        if checkpoint["data_sha256"] != corpus.sha256:
            raise ValueError("corpus changed since checkpoint")
        if checkpoint["environment"]["device"] != str(device):
            raise ValueError("resume requires the original device for RNG continuity")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng"], generator, device)
        step, best_loss = checkpoint["step"], checkpoint["best_loss"]
        history = checkpoint["history"]
        elapsed_before = history[-1]["elapsed_seconds"] if history else 0.0
        metadata["resumed_from"] = str(resume)
        metadata["origin_git_commit"] = checkpoint["git_commit"]
    end_step = (
        config.train.max_iters if stop_after is None else min(stop_after, config.train.max_iters)
    )
    if end_step <= step:
        raise ValueError("stop_after/max_iters must exceed the checkpoint step")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir / "config.yaml")
    corpus.tokenizer.save(out_dir / "vocab.json")
    synchronize(device)
    start_time = time.perf_counter()
    interval_start = start_time
    interval_step = step
    tokens_per_step = config.train.batch_size * config.model.block_size

    def record() -> None:
        nonlocal best_loss, interval_start, interval_step
        synchronize(device)
        now = time.perf_counter()
        tokens_per_second = (
            (step - interval_step) * tokens_per_step / max(now - interval_start, 1e-9)
        )
        losses = estimate_losses(model, corpus, config.train)
        synchronize(device)
        elapsed = elapsed_before + time.perf_counter() - start_time
        row = {
            "step": step,
            "train_loss": losses["train"],
            "val_loss": losses["val"],
            "lr": learning_rate(max(0, step - 1), config.train),
            "tokens_per_second": tokens_per_second,
            "elapsed_seconds": elapsed,
            "git_commit": metadata["git_commit"],
        }
        history.append(row)
        improved = losses["val"] < best_loss
        best_loss = min(best_loss, losses["val"])
        payload = {
            **metadata,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": rng_state(generator, device),
            "step": step,
            "best_loss": best_loss,
            "history": history,
            "vocab": corpus.tokenizer.chars,
        }
        save_checkpoint(payload, out_dir / "last.pt")
        if improved:
            save_checkpoint(payload, out_dir / "best.pt")
        _write_logs(out_dir, metadata, history)
        LOGGER.info(
            "step %d/%d train %.4f val %.4f %.0f tokens/s",
            step,
            config.train.max_iters,
            losses["train"],
            losses["val"],
            tokens_per_second,
        )
        interval_step, interval_start = step, time.perf_counter()

    if step == 0:
        record()
    model.train()
    while step < end_step:
        x, y = get_batch(
            corpus.train, config.train.batch_size, config.model.block_size, device, generator
        )
        lr = learning_rate(step, config.train)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with precision_context(config.train, device):
            _, loss = model(x, y)
        assert loss is not None
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"nonfinite training loss at step {step}")
        loss.backward()
        if config.train.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        optimizer.step()
        step += 1
        if step % config.train.eval_interval == 0 or step == end_step:
            record()
    return {**metadata, "history": history, "best_loss": best_loss, "step": step}


def main() -> None:
    """Train from YAML, optionally resuming or stopping without changing the LR schedule."""
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--out-dir")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.out_dir:
        config.train = replace(config.train, out_dir=args.out_dir)
    run_training(config, args.resume, args.stop_after)


if __name__ == "__main__":
    main()
