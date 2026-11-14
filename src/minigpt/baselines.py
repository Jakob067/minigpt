"""Reference losses measured on the same contiguous character validation split."""

import math

import torch
from torch import Tensor


def loss_metrics(loss: float) -> dict[str, float]:
    """Convert cross entropy in nats into perplexity and bits per character."""
    return {"loss": loss, "perplexity": math.exp(loss), "bits_per_char": loss / math.log(2)}


def uniform_loss(vocab_size: int) -> float:
    """Return the cross entropy of assigning equal probability to every token."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    return math.log(vocab_size)


def bigram_loss(train: Tensor, val: Tensor, vocab_size: int, smoothing: float = 1.0) -> float:
    """Fit add-alpha-smoothed bigram counts on training pairs and score validation pairs."""
    if smoothing <= 0 or vocab_size <= 0 or min(len(train), len(val)) < 2:
        raise ValueError("positive smoothing/vocabulary and at least two tokens per split required")
    pair_ids = train[:-1] * vocab_size + train[1:]
    counts = torch.bincount(pair_ids, minlength=vocab_size**2).reshape(vocab_size, vocab_size)
    counts = counts.to(torch.float64) + smoothing
    probabilities = counts / counts.sum(dim=-1, keepdim=True)
    return -probabilities[val[:-1], val[1:]].log().mean().item()
