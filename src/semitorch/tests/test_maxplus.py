import torch
import semitorch
import pytest
from semitorch import maxplus, maxplus
from torch.autograd.gradcheck import gradcheck


RNG_SEED = 0

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def test_maxplus():
    torch.manual_seed(RNG_SEED)
    x = torch.Tensor([[0.0, 1.0]])
    a = torch.Tensor([[0.0, 0.0], [0.0, -1.0], [-1.0, -1.0]])
    y = maxplus(x, a)
    assert y.allclose(torch.Tensor([1.0, 0.0, 0.0]))

    x1 = torch.randn(10, 10, requires_grad=True, device="cuda")
    a1 = torch.randn(5, 10, requires_grad=True, device="cuda")
    b1 = torch.randn(5, requires_grad=True, device="cuda")
    grad_y = torch.randn(10, 5, device="cuda")

    x2 = torch.clone(x1).detach().requires_grad_(True)
    a2 = torch.clone(a1).detach().requires_grad_(True)
    b2 = torch.clone(b1).detach().requires_grad_(True)

    y1 = torch.max(x1.unsqueeze(-2) + a1, dim=-1)[0] + b1
    y2 = maxplus(x2, a2, b2)

    assert y1.allclose(y2)

    y1.backward(grad_y)
    y2.backward(grad_y)

    assert x1.grad.allclose(x2.grad)
    assert a1.grad.allclose(a2.grad)
    assert b1.grad.allclose(b2.grad)


def test_maxplus_cpu_autograd():
    torch.manual_seed(RNG_SEED)

    input = (
        torch.randn(1, 10, requires_grad=True, dtype=torch.float64, device="cpu"),
        torch.randn(5, 10, requires_grad=True, dtype=torch.float64, device="cpu"),
    )

    assert gradcheck(maxplus, input, atol=1e-3, rtol=1e-1)


def test_maxplus_cuda_autograd():
    torch.manual_seed(RNG_SEED)

    input = (
        torch.randn(1, 10, requires_grad=True, dtype=torch.float64, device="cuda"),
        torch.randn(5, 10, requires_grad=True, dtype=torch.float64, device="cuda"),
    )

    assert gradcheck(maxplus, input, atol=1e-3, rtol=1e-1)