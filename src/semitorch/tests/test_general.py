import semitorch
import pytest
import torch
from torch.autograd.gradcheck import gradcheck
from semitorch import LayerScaler, LayerNorm2d

RNG_SEED = 42

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def test_layerscale():
    torch.manual_seed(RNG_SEED)
    x = torch.ones(1, 2, 3, 4)
    m = LayerScaler(init_value = 1.0, channels=4)
    assert m(x).allclose(x)
    gammas = torch.Tensor([1.0, 2.0, 3.0, 4.0])
    m.gamma = torch.nn.Parameter(gammas)
    assert m(x).allclose(x * gammas)

