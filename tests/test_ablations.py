from pathlib import Path

import pytest

from minigpt.ablate import build_runs, load_plan, summarize
from minigpt.config import load_config


def test_grid_budget_and_single_factor() -> None:
    plan = load_plan("configs/ablation.yaml")
    base = load_config(plan["base_config"])
    runs = build_runs(base, plan)
    assert len(runs) == 12
    assert len({config.train.out_dir for _, config in runs}) == 12
    for variant, config in runs:
        assert config.train.max_iters * config.train.batch_size * config.model.block_size == 1048576
        changed = [
            key
            for key, value in config.to_dict()["model"].items()
            if value != base.to_dict()["model"][key]
        ]
        assert len(changed) == (0 if variant == "baseline" else 1)
    with pytest.raises(ValueError, match="divisible"):
        build_runs(base, {**plan, "token_budget": 7})


def test_summary_uses_sample_std() -> None:
    rows = []
    for name, _ in build_runs(
        load_config("configs/small.yaml"), load_plan("configs/ablation.yaml")
    ):
        rows.append(
            {
                "variant": name,
                "loss": 1.0 if len(rows) % 2 else 3.0,
                "perplexity": 2.0,
                "bits_per_char": 3.0,
            }
        )
    summary = summarize(rows)
    assert len(summary) == 8
    assert summary[0]["loss_mean"] == 2
    assert summary[0]["loss_std"] == pytest.approx(2**0.5)


def test_plan_requires_two_seeds(tmp_path: Path) -> None:
    text = Path("configs/ablation.yaml").read_text().replace("[1337, 2024]", "[1337]")
    path = tmp_path / "grid.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="two distinct"):
        load_plan(path)
