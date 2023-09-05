import torch
from torch import Tensor
from torch.nn.functional import conv2d
from typing import Optional
from torch.utils.checkpoint import checkpoint


def _logconv2d(
    input: Tensor, weight: Tensor, bias=Optional[Tensor], mu: float = 1.0, **kwargs
):
    input = torch.exp(input.mul(mu))
    weight = torch.exp(weight.mul(mu))
    if isinstance(bias, Tensor):
        bias = torch.exp(mu * bias)
        y = conv2d(input, weight, bias, **kwargs)
    else:
        y = conv2d(input, weight, **kwargs)
    return torch.log(y).div(mu)


def logconv2d(
    input: Tensor, weight: Tensor, bias=Optional[Tensor], mu: float = 1.0, **kwargs
):
    if (
        input.requires_grad
        or weight.requires_grad
        or (isinstance(bias, Tensor) and bias.requires_grad)
    ):
        return checkpoint(_logconv2d, input, weight, bias, mu, **kwargs)
    else:
        return _logconv2d(input, weight, bias, mu, **kwargs)
