import pytest
import torch

from minigpt.config import ModelConfig
from minigpt.evaluate import full_validation_loss
from minigpt.generate import filter_logits, generate
from minigpt.model import GPT


def test_deterministic_generation_and_cropping() -> None:
    model = GPT(ModelConfig(vocab_size=7, block_size=4, n_embd=16, n_head=2, n_layer=1))
    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]])
    first = generate(model, prompt, max_new_tokens=9, seed=10)
    assert first.shape == (1, 15)
    assert torch.equal(first, generate(model, prompt, max_new_tokens=9, seed=10))
    assert torch.equal(
        generate(model, prompt, 9, temperature=0), generate(model, prompt, 9, top_k=1, seed=37)
    )
    assert model.training
    with pytest.raises(ValueError):
        generate(model, prompt, temperature=-1)


def test_nucleus_and_top_k() -> None:
    logits = torch.log(torch.tensor([[0.6, 0.25, 0.1, 0.05]]))
    top = filter_logits(logits, top_k=2)
    assert torch.isfinite(top).tolist() == [[True, True, False, False]]
    nucleus = filter_logits(logits, top_p=0.7)
    assert torch.isfinite(nucleus).tolist() == [[True, True, False, False]]
    assert torch.isfinite(filter_logits(logits, top_p=0.01)).sum() == 1
    with pytest.raises(ValueError):
        filter_logits(logits, top_p=0)


@pytest.mark.parametrize("length", [2, 9, 10, 17, 20])
def test_full_eval_token_weighting(length: int) -> None:
    model = GPT(ModelConfig(vocab_size=5, block_size=8, n_embd=16, n_head=2, n_layer=1)).eval()
    tokens = torch.arange(length) % 5
    expected = 0.0
    for start in range(0, length - 1, 8):
        end = min(length - 1, start + 8)
        with torch.no_grad():
            _, loss = model(tokens[None, start:end], tokens[None, start + 1 : end + 1])
        expected += loss.item() * (end - start)
    assert full_validation_loss(model, tokens, batch_size=2) == pytest.approx(
        expected / (length - 1)
    )
