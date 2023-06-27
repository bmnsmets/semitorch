import pytest
import torch
from torch.autograd.gradcheck import gradcheck
from typing import Sequence

DEFAULT_RNG_SEED = 0

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@pytest.mark.skip
def test_with_autograd(
        function: torch.Tensor | Sequence[torch.Tensor],
        device: str,
        rng_seed: int = DEFAULT_RNG_SEED
) -> bool:
    torch.manual_seed(rng_seed)

    test_input = (
        torch.randn(1, 10, requires_grad=True, dtype=torch.float64, device=device),
        torch.randn(5, 10, requires_grad=True, dtype=torch.float64, device=device),
    )

    return gradcheck(function, test_input, atol=1e-3, rtol=1e-1)
