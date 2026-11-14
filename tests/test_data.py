import random
from pathlib import Path

import numpy as np
import pytest
import torch

from minigpt.config import Config, ModelConfig, TrainConfig, load_config, save_config
from minigpt.data import get_batch, load_corpus
from minigpt.tokenizer import CharTokenizer
from minigpt.utils import seed_everything, select_device


def test_tokenizer(tmp_path: Path) -> None:
    text = "Hello, Welt! äöü\n🙂"
    tokenizer = CharTokenizer(text)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    path = tmp_path / "vocab.json"
    tokenizer.save(path)
    assert CharTokenizer.load(path).encode(text) == tokenizer.encode(text)
    with pytest.raises(ValueError, match="unknown"):
        tokenizer.encode("Z")
    with pytest.raises(ValueError, match="invalid"):
        tokenizer.decode([-1])


def test_split_and_batch(tmp_path: Path) -> None:
    path = tmp_path / "text.txt"
    path.write_text("abcdefghij" * 100)
    corpus = load_corpus(path)
    assert len(corpus.train) == 900
    assert len(corpus.val) == 100
    assert corpus.tokenizer.decode(corpus.train.tolist()) == ("abcdefghij" * 90)
    rng = torch.Generator().manual_seed(7)
    x, y = get_batch(corpus.train, 3, 16, torch.device("cpu"), rng)
    assert x.shape == y.shape == (3, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])
    rng.manual_seed(7)
    assert torch.equal(x, get_batch(corpus.train, 3, 16, torch.device("cpu"), rng)[0])
    with pytest.raises(ValueError):
        get_batch(corpus.val, 1, 100, torch.device("cpu"))


def test_config_round_trip(tmp_path: Path) -> None:
    config = Config()
    path = tmp_path / "config.yaml"
    save_config(config, path)
    assert load_config(path) == config
    with pytest.raises(TypeError):
        Config.from_dict({"model": {"n_layers": 4}})
    with pytest.raises(ValueError):
        ModelConfig(n_embd=63)
    with pytest.raises(ValueError):
        ModelConfig(dropout=1)
    with pytest.raises(ValueError):
        TrainConfig(warmup_iters=200)
    for name in ("tiny", "small"):
        load_config(f"configs/{name}.yaml")


def test_seed_and_device() -> None:
    seed_everything(17)
    values = random.random(), np.random.rand(), torch.rand(2)
    seed_everything(17)
    assert values[0] == random.random()
    assert values[1] == np.random.rand()
    assert torch.equal(values[2], torch.rand(2))
    assert select_device("cpu").type == "cpu"
    assert select_device().type in {"cpu", "mps", "cuda"}
