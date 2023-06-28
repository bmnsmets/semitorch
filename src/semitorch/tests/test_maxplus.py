import torch
from semitorch import maxplus, tests


def test_maxplus_should_add() -> None:
    torch.manual_seed(tests.DEFAULT_RNG_SEED)
    x = torch.Tensor([[5.0]])
    a = torch.Tensor([[1.0], [2.0]])
    y = maxplus(x, a)
    assert y.allclose(torch.Tensor([6.0, 7.0]))


def test_maxplus_should_add_and_take_max() -> None:
    torch.manual_seed(tests.DEFAULT_RNG_SEED)
    x = torch.Tensor([[0.0, 1.0, 2.0, 3.0, 4.0, -1.0]])
    a = torch.Tensor([
        [0.0, 0.0, 8.0, 1.2, 0.5, -3.6],
        [0.0, -1.0, -9.2, 0.3, 4.2, 4.1],
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    ])
    y = maxplus(x, a)
    assert y.allclose(torch.Tensor([10.0, 8.2, 3.0]))


def test_maxplus_should_add_and_take_max_2D() -> None:
    torch.manual_seed(tests.DEFAULT_RNG_SEED)
    x = torch.Tensor([[4.0, 1.0]])
    a = torch.Tensor([[1.0, -5.0], [-12.0, 14.0]])
    y = maxplus(x, a)
    assert y.allclose(torch.Tensor([5.0, 15.0]))


def test_maxplus_should_add_take_max_6D() -> None:
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
    y = maxplus(x, a)
    print(y)
    assert y.allclose(torch.Tensor([
        [9.2, 4.1, 3.0],
        [11.4, 6.3, 4.0],
        [13.4, 8.3, 3.2],
        [12.5, 8.3, 7.0],
        [9.3, 8.3, 7.3],
        [7.0, 5.0, 4.0],
    ]))


def test_maxplus_fw_bw() -> None:
    tests.test_fw_bw(
        lambda x, a, b: torch.max(x.unsqueeze(-2) + a, dim=-1)[0] + b,
        maxplus
    )


def test_maxplus_cpu_autograd():
    tests.test_with_autograd(maxplus, "cpu")


def test_maxplus_cuda_autograd():
    tests.test_with_autograd(maxplus, "cuda")
