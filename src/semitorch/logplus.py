import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import linear
from torch import Tensor
from typing import Tuple, Optional
import math
from itertools import chain


def _logplus(
    x: Tensor,
    w: Tensor,
    mu: float = 1.0,
    bias: Optional[Tensor] = None,
) -> Tensor:
    x = x.mul(mu).exp()
    w = w.mul(mu).exp()
    if bias:
        bias = bias.mul(mu).exp()
    y = linear(x, w, bias)
    return y.log().div(mu)


logplus = _logplus


class LogPlus(nn.Module):
    """
    Fully connected quasi-linear operator in the logarithmic semiring.
    """

    __constants__ = ["in_features", "out_features", "mu"]
    in_features: int
    out_features: int
    weight: torch.Tensor
    mu: float

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mu: float = 1.0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.mu = mu
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return logplus(input, self.weight, self.mu, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, mu={}, bias={}".format(
            self.in_features, self.out_features, self.mu, self.bias is not None
        )


def logplus_parameters(model):
    return chain.from_iterable(
        m.parameters() for m in model.modules() if isinstance(m, LogPlus)
    )


def nonlogplus_parameters(model):
    return chain.from_iterable(
        m.parameters()
        for m in model.modules()
        if not isinstance(m, LogPlus) and list(m.children()) == []
    )