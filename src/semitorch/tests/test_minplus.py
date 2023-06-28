import pytest
import torch
from semitorch import minplus

DEFAULT_RNG_SEED = 0
torch.manual_seed(DEFAULT_RNG_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def test_minplus_should_add() -> None:
    x = torch.Tensor([[5.0]])
    a = torch.Tensor([[1.0], [2.0]])
    y = minplus(x, a)
    assert y.allclose(torch.Tensor([6.0, 7.0]))


def test_minplus_should_take_min() -> None:
    x = torch.Tensor([[0.0, 1.0, 2.0, 3.0, 4.0, -1.0]])
    a = torch.Tensor([[0.0, 0.0, 8.0, 1.2, 0.5, -3.6]])
    y = minplus(x, a)
    assert y.allclose(torch.Tensor([-4.6]))


def test_minplus_should_add_and_take_min() -> None:
    x = torch.Tensor([[0.0, 1.0, 2.0, 3.0, 4.0, -1.0]])
    a = torch.Tensor([
        [0.0, 0.0, 8.0, 1.2, 0.5, -3.6],
        [0.0, -1.0, -9.2, 0.3, 4.2, 4.1],
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    ])
    y = minplus(x, a)
    assert y.allclose(torch.Tensor([-4.6, -7.2, -2.0]))


def test_minplus_should_add_and_take_min_2D() -> None:
    x = torch.Tensor([[4.0, 1.0]])
    a = torch.Tensor([[1.0, -5.0], [-12.0, 14.0]])
    y = minplus(x, a)
    assert y.allclose(torch.Tensor([-4.0, -8.0]))


def test_minplus_should_add_take_min_6D() -> None:
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
    y = minplus(x, a)
    assert y.allclose(torch.Tensor([
        [-0.7, -9.1, -2.9],
        [-9.5, -9.2, -11.7],
        [-4.7, -10.2, -10.2],
        [1.0, -9.2, -1.0],
        [-4.0, -12.8, -5.1],
        [-2.7, -13.8, -8.2],
    ]))


def test_minplus_init() -> None:
    weight = torch.nn.Parameter(
        torch.empty((2, 2))
    )
    k = 2

    with torch.no_grad():
        torch.nn.init.eye_(weight).add_(-1).mul_(-k)

    assert weight.allclose(torch.Tensor([
        [0, 2],
        [2, 0],
    ]))


def test_minplus_fw_bw() -> None:
    x1 = torch.randn(10, 10, requires_grad=True, device="cuda")
    a1 = torch.randn(5, 10, requires_grad=True, device="cuda")
    b1 = torch.randn(5, requires_grad=True, device="cuda")
    grad_y = torch.randn(10, 5, device="cuda")


    y1 = torch.min(x1.unsqueeze(-2) + a1, dim=-1)[0] + b1
    y2 = minplus(x2, a2, b2)


    y1.backward(grad_y)
    y2.backward(grad_y)



def test_minplus_should_error_on_different_devices() -> None:
    for [device1, device2] in [["cpu", "cuda"], ["cuda", "cpu"]]:
        test_input = (
            torch.randn(1, 10, requires_grad=True, dtype=torch.float64, device=device1),
            torch.randn(5, 10, requires_grad=True, dtype=torch.float64, device=device2),
        )

        with pytest.raises(AssertionError):
            torch.autograd.gradcheck(minplus, test_input, atol=1e-3, rtol=1e-1)


def test_minplus_should_error_on_wrong_dimensions() -> None:
    test_input = (
        torch.randn(1, 10, 3, requires_grad=True, dtype=torch.float64, device="cuda"),
        torch.randn(5, 10, requires_grad=True, dtype=torch.float64, device="cuda"),
    )

    with pytest.raises(RuntimeError):
        torch.autograd.gradcheck(minplus, test_input, atol=1e-3, rtol=1e-1)

    test_input = (
        torch.randn(1, 10, requires_grad=True, dtype=torch.float64, device="cuda"),
        torch.randn(5, 10, 3, requires_grad=True, dtype=torch.float64, device="cuda"),
    )

    with pytest.raises(RuntimeError):
        torch.autograd.gradcheck(minplus, test_input, atol=1e-3, rtol=1e-1)
