import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from minigpt.baselines import bigram_loss, loss_metrics, uniform_loss
from minigpt.config import Config, ModelConfig, TrainConfig
from minigpt.model import GPT
from minigpt.train import learning_rate, load_checkpoint, load_model, make_optimizer, run_training


def training_config(tmp_path: Path, out: str) -> Config:
    path = tmp_path / "input.txt"
    path.write_text("abcde" * 100)
    return Config(
        ModelConfig(vocab_size=5, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.2),
        TrainConfig(
            data_path=str(path),
            out_dir=str(tmp_path / out),
            device="cpu",
            num_threads=1,
            batch_size=2,
            max_iters=8,
            warmup_iters=2,
            eval_interval=2,
            eval_batches=2,
        ),
    )


def test_checkpoint_and_exact_resume(tmp_path: Path) -> None:
    full = training_config(tmp_path, "full")
    run_training(full)
    partial = training_config(tmp_path, "partial")
    run_training(partial, stop_after=4)
    run_training(partial, resume=tmp_path / "partial/last.pt")
    left = load_checkpoint(tmp_path / "full/last.pt")
    right = load_checkpoint(tmp_path / "partial/last.pt")
    assert left["step"] == right["step"] == 8
    assert left["config"]["train"]["max_iters"] == 8
    assert left["git_commit"] and left["rng"]
    for name in left["model"]:
        torch.testing.assert_close(left["model"][name], right["model"][name], atol=0, rtol=0)
    for a, b in zip(left["history"], right["history"], strict=True):
        assert a["val_loss"] == b["val_loss"]
        assert a["train_loss"] == b["train_loss"]
    model, checkpoint = load_model(tmp_path / "full/last.pt", torch.device("cpu"))
    reference = GPT(full.model).eval()
    reference.load_state_dict(checkpoint["model"])
    x = torch.tensor([[0, 1, 2]])
    torch.testing.assert_close(model(x)[0], reference(x)[0], atol=0, rtol=0)
    assert model.lm_head.weight is model.token_embedding.weight
    with pytest.raises(FileExistsError):
        run_training(full)
    altered = Config(full.model, replace(full.train, max_iters=10))
    with pytest.raises(ValueError, match="original config"):
        run_training(altered, resume=tmp_path / "full/last.pt")


def test_optimizer_groups_and_schedule() -> None:
    config = TrainConfig(max_iters=10, warmup_iters=2)
    model = GPT(ModelConfig(vocab_size=5, block_size=8, n_embd=16, n_head=2, n_layer=1))
    optimizer = make_optimizer(model, config)
    assert all(p.ndim == 2 for p in optimizer.param_groups[0]["params"])
    assert all(p.ndim == 1 for p in optimizer.param_groups[1]["params"])
    assert optimizer.param_groups[1]["weight_decay"] == 0
    assert learning_rate(0, config) == config.learning_rate / 2
    assert learning_rate(1, config) == config.learning_rate
    assert learning_rate(9, config) == config.min_lr


def test_baselines() -> None:
    assert uniform_loss(4) == math.log(4)
    train = torch.tensor([0, 1] * 100)
    val = torch.tensor([0, 1] * 10)
    assert bigram_loss(train, val, 2) < 0.02
    assert loss_metrics(math.log(4))["perplexity"] == 4
    assert loss_metrics(math.log(4))["bits_per_char"] == 2


def test_resume_new_directory_preserves_best(tmp_path: Path) -> None:
    original = training_config(tmp_path, "original")
    run_training(original, stop_after=4)
    continued = Config(original.model, replace(original.train, out_dir=str(tmp_path / "continued")))
    run_training(continued, resume=tmp_path / "original/last.pt")
    assert (tmp_path / "continued/best.pt").exists()
    assert load_checkpoint(tmp_path / "continued/last.pt")["step"] == 8
    fresh = training_config(tmp_path, "reference")
    run_training(fresh)
    resumed = load_checkpoint(tmp_path / "continued/last.pt")
    reference = load_checkpoint(tmp_path / "reference/last.pt")
    for name in resumed["model"]:
        torch.testing.assert_close(resumed["model"][name], reference["model"][name], atol=0, rtol=0)
