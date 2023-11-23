import pytest
import torch
from semitorch import viterbi

DEFAULT_RNG_SEED = 0
torch.manual_seed(DEFAULT_RNG_SEED)


def test_viterbi_should_multiply() -> None:
    x = torch.Tensor([[1.0]])
    a = torch.Tensor([[1.0], [0.5]])
    y = viterbi(x, a)
    assert y.allclose(torch.Tensor([1.0, 0.5]))


def test_viterbi_should_multiply_with_bias() -> None:
    x = torch.Tensor([[1.0]])
    a = torch.Tensor([[1.0], [0.5]])
    b = torch.Tensor([0.0, 1.0])
    y = viterbi(x, a, b)
    assert y.allclose(torch.Tensor([1.0, 1.0]))


def test_viterbi_should_multiply_and_take_max() -> None:
    x = torch.Tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 1.0]])
    a = torch.Tensor([[0.0, 0.0, 1.0, 0.8, 0.3, 0.1]])
    y = viterbi(x, a)
    assert y.allclose(torch.Tensor([0.24]))


def test_viterbi_should_multiply_and_take_max_with_bias() -> None:
    x = torch.Tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 1.0]])
    a = torch.Tensor([[0.0, 0.0, 1.0, 0.8, 0.3, 0.1]])
    b = torch.ones(6)
    y = viterbi(x, a, b)
    assert y.allclose(torch.ones(6))


def test_viterbi_should_multiply_and_take_max2() -> None:
    x = torch.Tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 1.0]])
    a = torch.Tensor([
        [0.0, 0.0, 0.8, 0.12, 0.5, 0.36],
        [0.0, 1.0, 0.92, 0.3, 0.42, 0.41],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ])
    y = viterbi(x, a)
    assert y.allclose(torch.Tensor([0.36, 0.41, 1.0]))


def test_viterbi_should_multiply_and_take_max2_with_bias() -> None:
    x = torch.Tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 1.0]])
    a = torch.Tensor([
        [0.0, 0.0, 0.8, 0.12, 0.5, 0.36],
        [0.0, 1.0, 0.92, 0.3, 0.42, 0.41],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ])
    b = torch.ones((3, 1))
    y = viterbi(x, a, b)
    assert y.allclose(torch.ones((3, 1)))


def test_viterbi_should_multiply_and_take_max_2D() -> None:
    x = torch.Tensor([[0.1, 1.0]])
    a = torch.Tensor([[1.0, 0.5], [0.12, 0.14]])
    y = viterbi(x, a)
    assert y.allclose(torch.Tensor([0.5, 0.14]))


def test_viterbi_should_multiply_and_take_max_2D_with_bias() -> None:
    x = torch.Tensor([[0.1, 1.0]])
    a = torch.Tensor([[1.0, 0.5], [0.12, 0.14]])
    b = torch.ones((2, 1))
    y = viterbi(x, a, b)
    assert y.allclose(torch.ones((2, 1)))


def test_viterbi_should_multiply_take_max_6D() -> None:
    x = torch.Tensor([
        [0.4, 1.0, 0.1, 0.2, 0.19, 0.0],
        [0.5, 0.3, 0.0, 0.2, 0.17, 0.22],
        [0.21, 0.0, 1.0, 0.92, 0.3, 0.42],
        [0.15, 0.9, 0.0, 0.8, 0.12, 0.5],
        [0.83, 0.41, 0.36, 0.41, 0.0, 0.1],
        [0.5, 0.34, 0.46, 0.72, 0.2, 0.22]],
    )
    a = torch.Tensor([
        [0.0, 0.1, 0.8, 0.45, 0.12, 0.92],
        [0.0, 1.0, 0.92, 0.3, 0.42, 0.41],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ])
    y = viterbi(x, a)
    assert y.allclose(torch.Tensor([
        [0.1, 1.0, 1.0],
        [0.2024, 0.3, 0.5],
        [0.8, 0.92, 1.0],
        [0.46, 0.9, 0.9],
        [0.288, 0.41, 0.83],
        [0.368, 0.4232, 0.72],
    ]))


def test_viterbi_should_multiply_take_max_6D_with_bias() -> None:
    x = torch.Tensor([
        [0.4, 1.0, 0.1, 0.2, 0.19, 0.0],
        [0.5, 0.3, 0.0, 0.2, 0.17, 0.22],
        [0.21, 0.0, 1.0, 0.92, 0.3, 0.42],
        [0.15, 0.9, 0.0, 0.8, 0.12, 0.5],
        [0.83, 0.41, 0.36, 0.41, 0.0, 0.1],
        [0.5, 0.34, 0.46, 0.72, 0.2, 0.22]],
    )
    a = torch.Tensor([
        [0.0, 0.1, 0.8, 0.45, 0.12, 0.92],
        [0.0, 1.0, 0.92, 0.3, 0.42, 0.41],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ])
    b = torch.ones((6, 3))
    y = viterbi(x, a, b)
    assert y.allclose(torch.ones((6, 3)))


def test_viterbi_fw_bw() -> None:
    x1 = torch.rand(10, 10, requires_grad=True, device="cuda")
    a1 = torch.rand(5, 10, requires_grad=True, device="cuda")
    b1 = torch.rand(5, requires_grad=True, device="cuda")
    grad_y = torch.rand(10, 5, device="cuda")

    x2 = torch.clone(x1).detach().requires_grad_(True)
    a2 = torch.clone(a1).detach().requires_grad_(True)
    b2 = torch.clone(b1).detach().requires_grad_(True)

    y1 = torch.max(b1, torch.max(x1.unsqueeze(-2) * a1, dim=-1)[0])
    y2 = viterbi(x2, a2, b2)

    assert y1.allclose(y2)

    y1.backward(grad_y)
    y2.backward(grad_y)

    assert x1.grad.allclose(x2.grad)
    assert a1.grad.allclose(a2.grad)
    assert b1.grad.allclose(b2.grad)


def test_viterbi_should_error_on_wrong_input_outside_01_interval() -> None:
    t1 = torch.tensor([5.0])
    t2 = torch.tensor([0.5])
    t3 = torch.tensor([0.0])

    with pytest.raises(AssertionError):
        viterbi(t1, t2)
        viterbi(t1, t2, t3)
    with pytest.raises(AssertionError):
        viterbi(t2, t1)
        viterbi(t2, t1, t3)
    with pytest.raises(AssertionError):
        viterbi(t2, t3, t1)


def test_viterbi_should_error_on_different_devices() -> None:
    for [device1, device2] in [["cpu", "cuda"], ["cuda", "cpu"]]:
        test_input = (
            torch.rand(1, 10, requires_grad=True, dtype=torch.float64, device=device1),
            torch.rand(5, 10, requires_grad=True, dtype=torch.float64, device=device2),
        )

        with pytest.raises(AssertionError):
            torch.autograd.gradcheck(viterbi, test_input, atol=1e-3, rtol=1e-1)


def test_viterbi_should_error_on_wrong_dimensions() -> None:
    test_input = (
        torch.rand(1, 10, 3, requires_grad=True, dtype=torch.float64, device="cuda"),
        torch.rand(5, 10, requires_grad=True, dtype=torch.float64, device="cuda"),
    )

    with pytest.raises(RuntimeError):
        torch.autograd.gradcheck(viterbi, test_input, atol=1e-3, rtol=1e-1)

    test_input = (
        torch.rand(1, 10, requires_grad=True, dtype=torch.float64, device="cuda"),
        torch.rand(5, 10, 3, requires_grad=True, dtype=torch.float64, device="cuda"),
    )

    with pytest.raises(RuntimeError):
        torch.autograd.gradcheck(viterbi, test_input, atol=1e-3, rtol=1e-1)
