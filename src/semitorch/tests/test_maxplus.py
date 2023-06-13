import torch
import semitorch
import pytest


RNG_SEED = 0

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def test_maxplus_cuda_autograd():
    from semitorch import maxplus, MaxPlus
    from torch.autograd.gradcheck import gradcheck

    torch.manual_seed(RNG_SEED)

    input = (
        torch.randn(1, 10, requires_grad=True, dtype=torch.float64, device='cuda'),
        torch.randn(5, 10, requires_grad=True, dtype=torch.float64, device='cuda')
    )


    assert gradcheck(maxplus, input, atol=1e-3, rtol=1e-1)

