"""Measure synchronized end-to-end training throughput before selecting run budgets."""

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from minigpt.config import Config, load_config
from minigpt.data import download_shakespeare, get_batch, load_corpus
from minigpt.model import GPT
from minigpt.train import estimate_losses, make_optimizer, precision_context
from minigpt.utils import (
    environment,
    git_hash,
    seed_everything,
    select_device,
    setup_logging,
    synchronize,
)


def benchmark(config: Config, warmup: int = 5, steps: int = 20) -> dict[str, Any]:
    """Time complete batch/forward/backward/AdamW steps after untimed device warmup."""
    if warmup < 0 or steps <= 0:
        raise ValueError("warmup must be nonnegative and steps positive")
    torch.set_num_threads(config.train.num_threads)
    seed_everything(config.train.seed)
    device = select_device(config.train.device)
    corpus = load_corpus(download_shakespeare(config.train.data_path), config.train.train_fraction)
    model = GPT(config.model).to(device).train()
    optimizer = make_optimizer(model, config.train)
    generator = torch.Generator().manual_seed(config.train.seed)

    def update() -> None:
        x, y = get_batch(
            corpus.train, config.train.batch_size, config.model.block_size, device, generator
        )
        optimizer.zero_grad(set_to_none=True)
        with precision_context(config.train, device):
            _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        optimizer.step()

    for _ in range(warmup):
        update()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(steps):
        update()
    synchronize(device)
    seconds = time.perf_counter() - start
    start = time.perf_counter()
    estimate_losses(model, corpus, config.train)
    synchronize(device)
    eval_seconds = time.perf_counter() - start
    return {
        "config": config.to_dict(),
        "git_commit": git_hash(),
        "environment": environment(device),
        "data_sha256": corpus.sha256,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "seconds": seconds,
        "seconds_per_step": seconds / steps,
        "tokens_per_second": steps * config.train.batch_size * config.model.block_size / seconds,
        "eval_seconds": eval_seconds,
        "estimated_run_seconds": config.train.max_iters * seconds / steps
        + (
            1
            + (config.train.max_iters + config.train.eval_interval - 1)
            // config.train.eval_interval
        )
        * eval_seconds,
        "excludes": "checkpoint writes, process startup; thermal throttling may change throughput",
    }


def main() -> None:
    """Benchmark one or more YAML configurations and write complete timing provenance."""
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=["configs/tiny.yaml", "configs/small.yaml"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", default="reports/benchmark.json")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    results = []
    for path in args.configs:
        config = load_config(path)
        if args.device:
            config.train = replace(config.train, device=args.device)
        result = benchmark(config, args.warmup, args.steps)
        result["config_path"] = path
        results.append(result)
        logging.info(
            "%s: %.0f tokens/s, %.3f s/step, estimated %.1f min",
            path,
            result["tokens_per_second"],
            result["seconds_per_step"],
            result["estimated_run_seconds"] / 60,
        )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
