import semitorch
import pytest
import torch
from torch.autograd.gradcheck import gradcheck
from semitorch import logconv2d, LogConv2d
from torch.autograd.gradcheck import gradcheck

RNG_SEED = 42

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

RNG_SEED = 0


def test_logconv2d():
    device = "cuda"
    torch.manual_seed(RNG_SEED)

    input = (
        torch.randn(
            1, 4, 10, 10, requires_grad=True, dtype=torch.float64, device=device
        ),
        torch.randn(2, 4, 3, 3, requires_grad=True, dtype=torch.float64, device=device),
    )

    assert gradcheck(semitorch.logconv._logconv2d, input, atol=1e-3, rtol=1e-1)

    x = torch.randn(1, 4, 10, 10, device=device)
    w = torch.randn(2, 4, 3, 3, device=device)
    y = logconv2d(x, w)
    assert y.shape == (1, 2, 8, 8)

    m = LogConv2d(4, 2, 3, mu=1.0).to(device)
    y = m(x)
    assert y.shape == (1, 2, 8, 8)

    m = LogConv2d(4, 4, 3, mu=1.0, padding="same", device=device)
    y = m(x)
    assert y.shape == x.shape