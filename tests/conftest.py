import pytest
import torch


@pytest.fixture(autouse=True)
def reproducible_cpu() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(42)
