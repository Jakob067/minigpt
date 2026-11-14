"""Validated, serializable experiment settings; all defaults live here."""

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Specify the decoder architecture independently of the training loop."""

    vocab_size: int = 65
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = True
    positional_embeddings: bool = True
    attention: str = "manual"
    mlp_ratio: int = 4
    init_std: float = 0.02

    def __post_init__(self) -> None:
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd", "mlp_ratio"):
            _positive_int(name, getattr(self, name))
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(self.init_std) or self.init_std <= 0:
            raise ValueError("init_std must be positive and finite")
        if self.attention not in {"manual", "sdpa"}:
            raise ValueError("attention must be manual or sdpa")
        if not isinstance(self.bias, bool) or not isinstance(self.positional_embeddings, bool):
            raise ValueError("bias and positional_embeddings must be boolean")


@dataclass
class TrainConfig:
    """Define optimization, data, evaluation, and reproducibility settings."""

    batch_size: int = 32
    max_iters: int = 200
    learning_rate: float = 0.003
    min_lr: float = 0.0003
    warmup_iters: int = 20
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 50
    eval_batches: int = 10
    seed: int = 1337
    device: str = "auto"
    precision: str = "float32"
    num_threads: int = 4
    data_path: str = "data/input.txt"
    train_fraction: float = 0.9
    out_dir: str = "runs/tiny"

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_iters", "eval_interval", "eval_batches", "num_threads"):
            _positive_int(name, getattr(self, name))
        for name in ("seed", "warmup_iters"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.warmup_iters >= self.max_iters:
            raise ValueError("warmup_iters must be less than max_iters")
        if not 0 < self.min_lr <= self.learning_rate or not math.isfinite(self.learning_rate):
            raise ValueError("learning rates must satisfy 0 < min_lr <= learning_rate < inf")
        for name in ("weight_decay", "grad_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam betas must be in [0, 1)")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1)")
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device must be auto, cpu, mps, or cuda")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")
        if not self.data_path or not self.out_dir:
            raise ValueError("data_path and out_dir cannot be empty")


@dataclass
class Config:
    """Bundle model and training settings for one fully specified run."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain mapping suitable for JSON, YAML, or checkpoints."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Config":
        """Load settings, rejecting unknown fields rather than silently ignoring typos."""
        if set(values) - {"model", "train"}:
            raise ValueError("unknown configuration section")
        return cls(ModelConfig(**values.get("model", {})), TrainConfig(**values.get("train", {})))


def load_config(path: str | Path) -> Config:
    """Read an experiment YAML file and validate every section."""
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("configuration must be a mapping")
    return Config.from_dict(values)


def save_config(config: Config, path: str | Path) -> None:
    """Write resolved settings, including defaults, to YAML."""
    Path(path).write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")


def _positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
