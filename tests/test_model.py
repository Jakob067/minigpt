from dataclasses import replace

import pytest
import torch

from minigpt.config import ModelConfig, load_config
from minigpt.model import GPT, parameter_formula
from minigpt.utils import count_parameters


def micro_config(**overrides: object) -> ModelConfig:
    return replace(
        ModelConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=16), **overrides
    )


@pytest.mark.parametrize("batch,time", [(1, 1), (2, 4), (3, 8)])
def test_shape(batch: int, time: int) -> None:
    model = GPT(micro_config())
    x = torch.randint(11, (batch, time))
    logits, loss = model(x, x)
    assert logits.shape == (batch, time, 11)
    assert loss is not None and loss.isfinite()
    with pytest.raises(ValueError, match="block_size"):
        model(torch.zeros(1, 9, dtype=torch.long))


@pytest.mark.parametrize("attention", ["manual", "sdpa"])
def test_causality(attention: str) -> None:
    model = GPT(micro_config(attention=attention)).eval()
    first = torch.randint(11, (2, 8))
    second = first.clone()
    second[:, 5:] = (second[:, 5:] + 1) % 11
    with torch.no_grad():
        torch.testing.assert_close(model(first)[0][:, :5], model(second)[0][:, :5], rtol=0, atol=0)


def test_attention_parity() -> None:
    manual = GPT(micro_config()).eval()
    sdpa = GPT(micro_config(attention="sdpa")).eval()
    sdpa.load_state_dict(manual.state_dict())
    x = torch.randint(11, (2, 8))
    torch.testing.assert_close(manual(x)[0], sdpa(x)[0], atol=1e-6, rtol=1e-5)
    manual(x, x)[1].backward()
    sdpa(x, x)[1].backward()
    for left, right in zip(manual.parameters(), sdpa.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("positions", [True, False])
def test_parameter_formula(bias: bool, positions: bool) -> None:
    config = micro_config(bias=bias, positional_embeddings=positions)
    model = GPT(config)
    assert count_parameters(model) == parameter_formula(config)
    assert model.token_embedding.weight is model.lm_head.weight


def test_documented_counts() -> None:
    # Bias-enabled R=4: V*E + T*E + L*(12*E² + 13*E) + 2*E.
    for name, expected in [("tiny", 108352), ("small", 10770816)]:
        config = load_config(f"configs/{name}.yaml").model
        assert parameter_formula(config) == expected
        assert count_parameters(GPT(config)) == expected


def test_overfit_single_batch() -> None:
    model = GPT(micro_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0)
    x = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    y = (x + 1) % 11
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        assert loss is not None and loss.isfinite()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.02
