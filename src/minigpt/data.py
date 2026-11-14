"""Contiguous data splitting and random autoregressive batch sampling."""

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from minigpt.tokenizer import CharTokenizer

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


@dataclass
class Corpus:
    """Keep train/validation token streams and a fingerprint of the source text."""

    train: Tensor
    val: Tensor
    tokenizer: CharTokenizer
    sha256: str


def download_shakespeare(path: str | Path) -> Path:
    """Download tiny Shakespeare only when absent; replace the file atomically."""
    destination = Path(path)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SHAKESPEARE_URL, timeout=60) as response:
            content = response.read()
        content.decode("utf-8")
        temporary = destination.with_suffix(".download")
        temporary.write_bytes(content)
        temporary.replace(destination)
    return destination


def load_corpus(path: str | Path, train_fraction: float = 0.9) -> Corpus:
    """Tokenize a UTF-8 file and split contiguously without overlapping targets.

    The vocabulary is fitted on the full corpus, as in nanoGPT's character dataset;
    only the character inventory sees validation, never counts or training examples.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    content = Path(path).read_bytes()
    text = content.decode("utf-8")
    tokenizer = CharTokenizer(text)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    boundary = int(len(tokens) * train_fraction)
    if boundary < 2 or len(tokens) - boundary < 2:
        raise ValueError("both splits must contain at least two characters")
    return Corpus(
        tokens[:boundary], tokens[boundary:], tokenizer, hashlib.sha256(content).hexdigest()
    )


def get_batch(
    tokens: Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample input windows and one-token-shifted targets using a CPU generator."""
    if batch_size <= 0 or block_size <= 0 or len(tokens) <= block_size:
        raise ValueError("positive batch/context sizes and at least block_size + 1 tokens required")
    starts = torch.randint(len(tokens) - block_size, (batch_size,), generator=generator)
    indices = starts[:, None] + torch.arange(block_size)
    return tokens[indices].to(device), tokens[indices + 1].to(device)
