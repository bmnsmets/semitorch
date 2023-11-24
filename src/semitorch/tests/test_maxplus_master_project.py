import pytest
import torch
from semitorch import maxplusmp

DEFAULT_RNG_SEED = 0
torch.manual_seed(DEFAULT_RNG_SEED)


def test_maxplusmp_should_add() -> None:
    x = torch.Tensor([[5.0]])
    a = torch.Tensor([[1.0], [2.0]])
    y = maxplusmp(x, a)
    assert y.allclose(torch.Tensor([6.0, 7.0]))


def test_maxplusmp_should_take_min() -> None:
    x = torch.Tensor([[0.0, 1.0, 2.0, 3.0, 4.0, -1.0]])
    a = torch.Tensor([[0.0, 0.0, 8.0, 1.2, 0.5, -3.6]])
    y = maxplusmp(x, a)
    assert y.allclose(torch.Tensor([10.0]))


def test_maxplusmp_should_add_and_take_max() -> None:
    x = torch.Tensor([[0.0, 1.0, 2.0, 3.0, 4.0, -1.0]])
    a = torch.Tensor([
        [0.0, 0.0, 8.0, 1.2, 0.5, -3.6],
        [0.0, -1.0, -9.2, 0.3, 4.2, 4.1],
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    ])
    y = maxplusmp(x, a)
    assert y.allclose(torch.Tensor([10.0, 8.2, 3.0]))


def test_maxplusmp_should_add_and_take_max_2D() -> None:
    x = torch.Tensor([[4.0, 1.0]])
    a = torch.Tensor([[1.0, -5.0], [-12.0, 14.0]])
    y = maxplusmp(x, a)
    assert y.allclose(torch.Tensor([5.0, 15.0]))


def test_maxplusmp_should_add_take_max_6D() -> None:
    x = torch.Tensor([
        [4.0, 1.0, 0.1, 0.2, -1.9, -0.0],
        [5.0, -3.0, -0.0, 0.2, -10.7, 2.2],
        [2.1, 0.0, -1.0, -9.2, 0.3, 4.2],
        [1.5, 0.9, 0.0, 8.0, 1.2, 0.5],
        [8.3, -4.1, -3.6, 4.1, 0.0, 0.1],
        [5.0, 3.4, -4.6, -7.2, -2.0, -2.2]],
    )
    a = torch.Tensor([
        [0.0, 0.1, 8.0, 4.5, 1.2, 9.2],
        [0.0, -1.0, -9.2, 0.3, 4.2, 4.1],
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    ])
    y = maxplusmp(x, a)
    assert y.allclose(torch.Tensor([
        [9.2, 4.1, 3.0],
        [11.4, 6.3, 4.0],
        [13.4, 8.3, 3.2],
        [12.5, 8.3, 7.0],
        [9.3, 8.3, 7.3],
        [7.0, 5.0, 4.0],
    ]))


def test_maxplusmp_init() -> None:
    weight = torch.nn.Parameter(
        torch.empty((2, 2))
    )
    k = 2

    with torch.no_grad():
        torch.nn.init.eye_(weight).add_(-1).mul_(k)

    assert weight.allclose(torch.Tensor([
        [0, -2],
        [-2, 0],
    ]))


def test_maxplusmp_fw_bw() -> None:
    x1 = torch.randn(10, 10, requires_grad=True, device="cuda")
    a1 = torch.randn(5, 10, requires_grad=True, device="cuda")
    b1 = torch.randn(5, requires_grad=True, device="cuda")
    grad_y = torch.randn(10, 5, device="cuda")

    x2 = torch.clone(x1).detach().requires_grad_(True)
    a2 = torch.clone(a1).detach().requires_grad_(True)
    b2 = torch.clone(b1).detach().requires_grad_(True)

    y1 = torch.max(x1.unsqueeze(-2) + a1, dim=-1)[0] + b1
    y2 = maxplusmp(x2, a2, b2)

    assert y1.allclose(y2)

    y1.backward(grad_y)
    y2.backward(grad_y)

    assert x1.grad.allclose(x2.grad)
    assert a1.grad.allclose(a2.grad)
    assert b1.grad.allclose(b2.grad)


def test_maxplusmp_should_error_on_different_devices() -> None:
    for [device1, device2] in [["cpu", "cuda"], ["cuda", "cpu"]]:
        test_input = (
            torch.randn(1, 10, requires_grad=True, dtype=torch.float64, device=device1),
            torch.randn(5, 10, requires_grad=True, dtype=torch.float64, device=device2),
        )

        with pytest.raises(AssertionError):
            torch.autograd.gradcheck(maxplusmp, test_input, atol=1e-3, rtol=1e-1)


def test_maxplusmp_should_error_on_wrong_dimensions() -> None:
    test_input = (
        torch.randn(1, 10, 3, requires_grad=True, dtype=torch.float64, device="cuda"),
        torch.randn(5, 10, requires_grad=True, dtype=torch.float64, device="cuda"),
    )

    with pytest.raises(RuntimeError):
        torch.autograd.gradcheck(maxplusmp, test_input, atol=1e-3, rtol=1e-1)

    test_input = (
        torch.randn(1, 10, requires_grad=True, dtype=torch.float64, device="cuda"),
        torch.randn(5, 10, 3, requires_grad=True, dtype=torch.float64, device="cuda"),
    )

    with pytest.raises(RuntimeError):
        torch.autograd.gradcheck(maxplusmp, test_input, atol=1e-3, rtol=1e-1)