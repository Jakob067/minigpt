"""A transparent, deterministic character vocabulary with no external tokenizer."""

import json
from collections.abc import Iterable
from pathlib import Path


class CharTokenizer:
    """Map sorted Unicode characters to consecutive integer token IDs."""

    def __init__(self, chars: Iterable[str]) -> None:
        self.chars = sorted(set(chars))
        if not self.chars or any(len(char) != 1 for char in self.chars):
            raise ValueError("vocabulary must contain individual characters")
        self.stoi = {char: index for index, char in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        """Return the number of unique characters."""
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        """Encode text, raising ValueError for characters outside the vocabulary."""
        try:
            return [self.stoi[char] for char in text]
        except KeyError as error:
            raise ValueError(f"unknown character: {error.args[0]!r}") from error

    def decode(self, tokens: Iterable[int]) -> str:
        """Decode token IDs, rejecting negative and out-of-range indices."""
        output = []
        for token in tokens:
            if not 0 <= token < self.vocab_size:
                raise ValueError(f"invalid token ID: {token}")
            output.append(self.chars[token])
        return "".join(output)

    def save(self, path: str | Path) -> None:
        """Persist the ordered vocabulary as readable UTF-8 JSON."""
        Path(path).write_text(json.dumps(self.chars, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Restore a vocabulary saved by save()."""
        chars = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(chars, list) or any(not isinstance(char, str) for char in chars):
            raise ValueError("vocabulary must be a JSON array of characters")
        tokenizer = cls(chars)
        if tokenizer.chars != chars:
            raise ValueError("vocabulary must be unique and sorted")
        return tokenizer
