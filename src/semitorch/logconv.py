import torch
from torch import Tensor
from torch.nn.functional import conv2d
from typing import Optional


def logconv2d(
    input: Tensor, weight: Tensor, bias=Optional[Tensor], mu: float = 1.0, **kwargs
):
    input = torch.exp(mu * input)
    weight = torch.exp(mu * weight)
    if bias != None:
        bias = torch.exp(mu * bias)
    y = conv2d(input, weight, bias, **kwargs)
    return 1 / mu * torch.log(y)
