"""Regenerate portfolio artifacts from completed checkpoints; never fabricate examples."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from minigpt.evaluate import evaluate_checkpoint, plot_losses
from minigpt.generate import generate
from minigpt.tokenizer import CharTokenizer
from minigpt.train import load_model
from minigpt.utils import environment, git_hash, select_device, setup_logging


def write_samples(
    checkpoint_path: str | Path,
    destination: Path,
    device: torch.device,
    max_new_tokens: int = 160,
    seed: int = 1337,
) -> dict[str, Any]:
    """Save all fixed-seed outputs, including cooler/hotter and top-k/top-p examples."""
    model, checkpoint = load_model(checkpoint_path, device)
    tokenizer = CharTokenizer(checkpoint["vocab"])
    torch.set_num_threads(checkpoint["config"]["train"]["num_threads"])
    requests = [
        {"prompt": prompt, "temperature": temperature, "top_k": None, "top_p": 1.0}
        for prompt in ("ROMEO:\n", "First Citizen:\n")
        for temperature in (0.5, 0.8, 1.2)
    ]
    requests += [
        {"prompt": "JULIET:\n", "temperature": 0.8, "top_k": 20, "top_p": 1.0},
        {"prompt": "JULIET:\n", "temperature": 0.8, "top_k": None, "top_p": 0.9},
    ]
    samples = []
    markdown = [
        "# Recorded samples",
        "",
        f"Checkpoint: `{checkpoint_path}`; seed: {seed}; "
        f"new characters: {max_new_tokens}; device: {device}.",
        "",
    ]
    for request in requests:
        tokens = torch.tensor(
            [tokenizer.encode(request["prompt"])], device=device, dtype=torch.long
        )
        result = generate(
            model,
            tokens,
            max_new_tokens=max_new_tokens,
            seed=seed,
            temperature=request["temperature"],
            top_k=request["top_k"],
            top_p=request["top_p"],
        )
        text = tokenizer.decode(result[0].tolist())
        samples.append({**request, "seed": seed, "max_new_tokens": max_new_tokens, "text": text})
        markdown.extend(
            [
                f"## Prompt {request['prompt']!r}, temperature {request['temperature']}, "
                f"top-k {request['top_k']}, top-p {request['top_p']}",
                "",
                "```text",
                text,
                "```",
                "",
            ]
        )
    payload = {
        "checkpoint": str(checkpoint_path),
        "step": checkpoint["step"],
        "config": checkpoint["config"],
        "training_git_commit": checkpoint["git_commit"],
        "generation_git_commit": git_hash(),
        "environment": environment(device),
        "samples": samples,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "samples.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (destination / "samples.md").write_text("\n".join(markdown), encoding="utf-8")
    return payload


def main() -> None:
    """Rebuild full metrics, curves, log copies, and unedited samples from finished local runs."""
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/small/best.pt")
    parser.add_argument("--tiny-checkpoint", default="runs/tiny/best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--sample-tokens", type=int, default=160)
    args = parser.parse_args()
    device = select_device(args.device)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, path in (("small", args.checkpoint), ("tiny", args.tiny_checkpoint)):
        result = evaluate_checkpoint(path, device)
        filename = "metrics.json" if name == "small" else "tiny_metrics.json"
        (destination / filename).write_text(json.dumps(result, indent=2), encoding="utf-8")
        for suffix in ("csv", "json"):
            shutil.copyfile(
                Path(path).parent / f"metrics.{suffix}", destination / f"{name}_train.{suffix}"
            )
        plot_losses(result["history"], destination / f"{name}_loss_curve.png")
    write_samples(args.checkpoint, destination, device, args.sample_tokens)


if __name__ == "__main__":
    main()
