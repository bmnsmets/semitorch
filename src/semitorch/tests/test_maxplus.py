import torch
from semitorch import maxplus, tests


def test_maxplus():
    torch.manual_seed(tests.DEFAULT_RNG_SEED)
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
    assert tests.test_with_autograd(maxplus, "cpu")


def test_maxplus_cuda_autograd():
    assert tests.test_with_autograd(maxplus, "cuda")
