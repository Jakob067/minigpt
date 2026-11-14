"""Equal-token, two-seed ablations with shared baseline runs and honest uncertainty."""

import argparse
import csv
import json
import logging
import math
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from minigpt.baselines import loss_metrics
from minigpt.config import Config, load_config
from minigpt.data import load_corpus
from minigpt.evaluate import full_validation_loss
from minigpt.train import load_checkpoint, load_model, run_training
from minigpt.utils import git_hash, select_device, setup_logging

GROUPS = {
    "n_layer": [("2", "layers_2"), ("4", "layers_4"), ("6", "baseline")],
    "block_size": [("64", "context_64"), ("128", "context_128"), ("256", "baseline")],
    "positional_embeddings": [("on", "baseline"), ("off", "positions_off")],
}
VARIANTS = {
    "baseline": {},
    "layers_2": {"n_layer": 2},
    "layers_4": {"n_layer": 4},
    "context_64": {"block_size": 64},
    "context_128": {"block_size": 128},
    "positions_off": {"positional_embeddings": False},
}


def load_plan(path: str | Path) -> dict[str, Any]:
    """Read a strict experiment grid with at least two distinct seeds."""
    plan = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {
        "base_config",
        "seeds",
        "token_budget",
        "warmup_fraction",
        "evaluations",
        "eval_batches",
        "out_dir",
        "report_dir",
    }
    if not isinstance(plan, dict) or set(plan) != expected:
        raise ValueError(f"grid must contain exactly {sorted(expected)}")
    if len(plan["seeds"]) < 2 or len(set(plan["seeds"])) != len(plan["seeds"]):
        raise ValueError("at least two distinct seeds required")
    if any(type(seed) is not int or seed < 0 for seed in plan["seeds"]):
        raise ValueError("seeds must be nonnegative integers")
    for field in ("token_budget", "evaluations", "eval_batches"):
        if type(plan[field]) is not int or plan[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if not 0 <= plan["warmup_fraction"] < 1:
        raise ValueError("warmup_fraction must be in [0, 1)")
    return plan


def build_runs(base: Config, plan: dict[str, Any]) -> list[tuple[str, Config]]:
    """Keep batch size fixed and scale schedule/eval intervals to equal token budgets.

    Context ablations necessarily change tokens per optimizer step. The normalized
    LR schedule is held fixed in token space; shorter context gets more updates.
    Baseline is reused across the three one-factor groups, never retrained three times.
    """
    runs = []
    for name, changes in VARIANTS.items():
        model = replace(base.model, **changes)
        tokens_per_step = base.train.batch_size * model.block_size
        if plan["token_budget"] % tokens_per_step:
            raise ValueError("token_budget must be divisible by every variant's tokens per step")
        steps = plan["token_budget"] // tokens_per_step
        for seed in plan["seeds"]:
            train = replace(
                base.train,
                seed=seed,
                max_iters=steps,
                warmup_iters=int(steps * plan["warmup_fraction"]),
                eval_interval=max(1, math.ceil(steps / plan["evaluations"])),
                eval_batches=plan["eval_batches"],
                out_dir=f"{plan['out_dir']}/{name}/seed_{seed}",
            )
            runs.append((name, Config(model, train)))
    return runs


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report arithmetic mean and sample standard deviation across seeds for every metric."""
    result = []
    for group, variants in GROUPS.items():
        for value, variant in variants:
            selected = [row for row in rows if row["variant"] == variant]
            if len(selected) < 2:
                raise ValueError(f"missing seeds for {variant}")
            summary = {"factor": group, "value": value, "variant": variant, "seeds": len(selected)}
            for metric in ("loss", "perplexity", "bits_per_char"):
                values = [row[metric] for row in selected]
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_std"] = statistics.stdev(values)
            result.append(summary)
    return result


def write_reports(plan: dict[str, Any], rows: list[dict[str, Any]], elapsed: float) -> None:
    """Publish CSV, provenance, and a Markdown table after the entire grid finishes."""
    summaries = summarize(rows)
    destination = Path(plan["report_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "ablations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    payload = {
        "plan": plan,
        "git_commit": git_hash(),
        "seconds": elapsed,
        "runs": rows,
        "summary": summaries,
        "protocol": "last checkpoint, full validation, sample std (ddof=1), equal tokens",
    }
    (destination / "ablations.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "| Factor | Value | Loss (nats) | Perplexity | Bits/char |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        cells = [
            f"{row[key + '_mean']:.4f} ± {row[key + '_std']:.4f}"
            for key in ("loss", "perplexity", "bits_per_char")
        ]
        lines.append(f"| {row['factor']} | {row['value']} | " + " | ".join(cells) + " |")
    (destination / "ablations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_grid(plan: dict[str, Any], reuse_completed: bool = False) -> list[dict[str, Any]]:
    """Execute unique variants sequentially; optionally reuse only verified completed runs."""
    runs = build_runs(load_config(plan["base_config"]), plan)
    rows = []
    start = time.perf_counter()
    for variant, config in runs:
        checkpoint_path = Path(config.train.out_dir) / "last.pt"
        if reuse_completed and checkpoint_path.exists():
            checkpoint = load_checkpoint(checkpoint_path)
            if (
                checkpoint["config"] != config.to_dict()
                or checkpoint["step"] != config.train.max_iters
            ):
                raise ValueError(
                    f"incomplete/different run: {checkpoint_path}; resume it explicitly"
                )
        else:
            run_training(config)
        device = select_device(config.train.device)
        model, checkpoint = load_model(checkpoint_path, device)
        corpus = load_corpus(config.train.data_path, config.train.train_fraction)
        if corpus.sha256 != checkpoint["data_sha256"]:
            raise ValueError("ablation corpus differs from checkpoint")
        metrics = loss_metrics(full_validation_loss(model, corpus.val, config.train.batch_size))
        row = {
            "variant": variant,
            "seed": config.train.seed,
            "steps": checkpoint["step"],
            "training_tokens": config.train.max_iters
            * config.train.batch_size
            * config.model.block_size,
            **metrics,
            "config": config.to_dict(),
            "git_commit": checkpoint["git_commit"],
            "environment": checkpoint["environment"],
            "data_sha256": corpus.sha256,
            "training_seconds": checkpoint["history"][-1]["elapsed_seconds"],
            "history": checkpoint["history"],
        }
        rows.append(row)
        logging.info("Completed %s seed=%d: loss %.4f", variant, config.train.seed, metrics["loss"])
        del model, checkpoint
    write_reports(plan, rows, time.perf_counter() - start)
    return rows


def main() -> None:
    """Inspect or execute the equal-token grid defined by a YAML plan."""
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ablation.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-completed", action="store_true")
    args = parser.parse_args()
    plan = load_plan(args.config)
    if args.dry_run:
        for name, config in build_runs(load_config(plan["base_config"]), plan):
            logging.info(
                "%s seed=%d: %d steps, %d tokens",
                name,
                config.train.seed,
                config.train.max_iters,
                plan["token_budget"],
            )
    else:
        run_grid(plan, args.reuse_completed)


if __name__ == "__main__":
    main()
