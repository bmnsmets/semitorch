import pytest
import torch
from torch.autograd.gradcheck import gradcheck
from typing import Callable, Sequence

DEFAULT_RNG_SEED = 0

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@pytest.mark.skip
def test_with_autograd(
        function: torch.Tensor | Sequence[torch.Tensor],
        device: str,
        rng_seed: int = DEFAULT_RNG_SEED
) -> None:
    torch.manual_seed(rng_seed)

    test_input = (
        torch.randn(1, 10, requires_grad=True, dtype=torch.float64, device=device),
        torch.randn(5, 10, requires_grad=True, dtype=torch.float64, device=device),
    )

    assert gradcheck(function, test_input, atol=1e-3, rtol=1e-1)


@pytest.mark.skip
def test_fw_bw(
        lambda_function: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        semitorch_function: torch.Tensor | Sequence[torch.Tensor],
) -> None:
    torch.manual_seed(DEFAULT_RNG_SEED)
    x1 = torch.randn(10, 10, requires_grad=True, device="cuda")
    a1 = torch.randn(5, 10, requires_grad=True, device="cuda")
    b1 = torch.randn(5, requires_grad=True, device="cuda")
    grad_y = torch.randn(10, 5, device="cuda")

    x2 = torch.clone(x1).detach().requires_grad_(True)
    a2 = torch.clone(a1).detach().requires_grad_(True)
    b2 = torch.clone(b1).detach().requires_grad_(True)

    y1 = lambda_function(x1, a1, b1)
    y2 = semitorch_function(x2, a2, b2)

    assert y1.allclose(y2)

    y1.backward(grad_y)
    y2.backward(grad_y)

    assert x1.grad.allclose(x2.grad)
    assert a1.grad.allclose(a2.grad)
    assert b1.grad.allclose(b2.grad)
