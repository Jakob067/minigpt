"""Autoregressive sampling with temperature, top-k, and nucleus filtering."""

import argparse
import logging
import math
from pathlib import Path

import torch
from torch import Tensor

from minigpt.model import GPT
from minigpt.tokenizer import CharTokenizer
from minigpt.train import load_model
from minigpt.utils import select_device, setup_logging


def filter_logits(logits: Tensor, top_k: int | None = None, top_p: float = 1.0) -> Tensor:
    """Keep top-k candidates, then the smallest sorted nucleus reaching probability p."""
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    result = logits.clone()
    if top_k is not None:
        threshold = torch.topk(result, min(top_k, result.size(-1))).values[..., -1:]
        result = result.masked_fill(result < threshold, float("-inf"))
    if top_p < 1:
        ordered, indices = torch.sort(result, descending=True, dim=-1)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        mask = torch.zeros_like(remove).scatter(-1, indices, remove)
        result = result.masked_fill(mask, float("-inf"))
    return result


@torch.no_grad()
def generate(
    model: GPT,
    idx: Tensor,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float = 1.0,
    seed: int = 1337,
) -> Tensor:
    """Append sampled tokens using cropped context and a dedicated CPU sampler RNG.

    Temperature zero is greedy; top_k=1 follows the same argmax path even on ties.
    Probabilities are copied to CPU for a seeded sampler supported on every device.
    Learned position indices restart at zero when cropping, as in nanoGPT.
    """
    if max_new_tokens < 0 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("max_new_tokens and finite temperature must be nonnegative")
    if idx.ndim != 2 or idx.size(1) == 0:
        raise ValueError("a nonempty (batch, time) prompt is required")
    filter_logits(torch.zeros(1, model.config.vocab_size), top_k, top_p)
    generator = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            logits = model(idx[:, -model.config.block_size :])[0][:, -1, :]
            if temperature == 0 or top_k == 1:
                token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = filter_logits(logits / temperature, top_k, top_p)
                probabilities = torch.softmax(logits, dim=-1).cpu()
                token = torch.multinomial(probabilities, 1, generator=generator).to(idx.device)
            idx = torch.cat((idx, token), dim=1)
    finally:
        model.train(was_training)
    return idx


def main() -> None:
    """Generate text from a trusted checkpoint, saving or logging the result."""
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/tiny/best.pt")
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    args = parser.parse_args()
    device = select_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    tokenizer = CharTokenizer(checkpoint["vocab"])
    tokens = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    result = generate(
        model, tokens, args.max_new_tokens, args.temperature, args.top_k, args.top_p, args.seed
    )
    text = tokenizer.decode(result[0].tolist())
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        logging.info("Generated text:\n%s", text)


if __name__ == "__main__":
    main()
