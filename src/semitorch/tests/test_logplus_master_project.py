import pytest
import torch
from itertools import product
from semitorch import logplusmp

DEFAULT_RNG_SEED = 0
torch.manual_seed(DEFAULT_RNG_SEED)


def test_log_multiplication_should_add() -> None:
    x = torch.Tensor([[5.0]])
    a = torch.Tensor([[1.0], [2.0]])
    mu = torch.Tensor([1.0])
    y = logplusmp(x, a, mu)
    assert y.allclose(torch.Tensor([6.0, 7.0]), atol=1e-5)


def test_log_addition_should_log_of_sum_of_exps() -> None:
    x = torch.Tensor([[0.0, 1.0, 2.0, 3.0, 4.0, -1.0]])
    a = torch.Tensor([[0.0, 0.0, 8.0, 1.2, 0.5, -3.6]])
    mu = torch.Tensor([1.0])
    y = logplusmp(x, a, mu)
    assert y.allclose(torch.Tensor([10.0073]), atol=1e-5)


def test_log_addition_and_multiplication() -> None:
    x = torch.Tensor([[0.0, 1.0, 2.0, 3.0, 4.0, -1.0]])
    a = torch.Tensor([
        [0.0, 0.0, 8.0, 1.2, 0.5, -3.6],
        [0.0, -1.0, -9.2, 0.3, 4.2, 4.1],
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    ])
    mu = torch.Tensor([1.0])
    y = logplusmp(x, a, mu)
    assert y.allclose(torch.Tensor([10.0073, 8.2140, 3.4562]), atol=1e-5)


def test_log_addition_and_multiplication_2D() -> None:
    x = torch.Tensor([[4.0, 1.0]])
    a = torch.Tensor([[1.0, -5.0], [-12.0, 14.0]])
    mu = torch.Tensor([1.0])
    y = logplusmp(x, a, mu)
    assert y.allclose(torch.Tensor([5.0001, 15.0]), atol=1e-5)


def test_log_addition_and_multiplication_6D() -> None:
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
    mu = torch.Tensor([1.0])
    y = logplusmp(x, a, mu)
    assert y.allclose(torch.Tensor([
        [9.5, 4.8486, 3.1075],
        [11.4356, 6.5434, 4.0734],
        [13.4017, 8.3242, 3.3512],
        [12.5695, 8.3780, 7.0043],
        [9.9271, 8.3520, 7.3154],
        [7.1767, 5.1656, 4.1853],
    ]), atol=1e-5)


def test_log_fw_bw() -> None:
    from torch.nn.functional import linear

    x1 = torch.randn(10, 10, requires_grad=True, device="cuda")
    a1 = torch.randn(5, 10, requires_grad=True, device="cuda")
    mu1 = torch.randn(1, requires_grad=True, device="cuda")
    b1 = torch.randn(5, requires_grad=True, device="cuda")
    grad_y = torch.randn(10, 5, device="cuda")

    x2 = torch.clone(x1).detach().requires_grad_(True)
    a2 = torch.clone(a1).detach().requires_grad_(True)
    mu2 = torch.clone(mu1).detach().requires_grad_(True)
    b2 = torch.clone(b1).detach().requires_grad_(True)

    y1 = torch.log(linear(torch.exp(mu1 * x1), torch.exp(mu1 * a1))) / mu1 + b1
    y2 = logplusmp(x2, a2, mu2, b2)

    assert y1.allclose(y2, atol=1e-5)

    y1.backward(grad_y)
    y2.backward(grad_y)

    assert x1.grad.allclose(x2.grad, atol=1e-5)
    assert a1.grad.allclose(a2.grad, atol=1e-5)
    assert mu1.grad.allclose(mu2.grad, atol=1e-5)
    assert b1.grad.allclose(b2.grad, atol=1e-5)


def test_logplusmp_should_error_on_different_devices() -> None:
    for [device1, device2, device3] in product(["cpu", "cuda"], repeat=3):
        test_input = (
            torch.randn(1, 10, requires_grad=True, dtype=torch.float64, device=device1),
            torch.randn(5, 10, requires_grad=True, dtype=torch.float64, device=device2),
            torch.randn(1, requires_grad=True, dtype=torch.float64, device=device3)
        )

        if not (device1 == device2 and device1 == device3):
            with pytest.raises(AssertionError):
                logplusmp(*test_input)


def test_logplusmp_should_error_on_wrong_dimensions() -> None:
    test_input = (
        torch.randn(1, 10, 3, requires_grad=True, dtype=torch.float64, device="cuda"),
        torch.randn(5, 10, requires_grad=True, dtype=torch.float64, device="cuda"),
        torch.randn(1, requires_grad=True, dtype=torch.float64, device="cuda"),
    )

    with pytest.raises(RuntimeError):
        torch.autograd.gradcheck(logplusmp, test_input, atol=1e-3, rtol=1e-1)

    test_input = (
        torch.randn(1, 10, requires_grad=True, dtype=torch.float64, device="cuda"),
        torch.randn(5, 10, 3, requires_grad=True, dtype=torch.float64, device="cuda"),
        torch.randn(1, requires_grad=True, dtype=torch.float64, device="cuda"),
    )

    with pytest.raises(RuntimeError):
        torch.autograd.gradcheck(logplusmp, test_input, atol=1e-3, rtol=1e-1)

    test_input = (
        torch.randn(1, 10, requires_grad=True, dtype=torch.float64, device="cuda"),
        torch.randn(5, 10, requires_grad=True, dtype=torch.float64, device="cuda"),
        torch.randn(1, 3, requires_grad=True, dtype=torch.float64, device="cuda"),
    )

    with pytest.raises(RuntimeError):
        torch.autograd.gradcheck(logplusmp, test_input, atol=1e-3, rtol=1e-1)