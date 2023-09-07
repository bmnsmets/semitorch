import torch
from torch import Tensor
from torch.nn.functional import conv2d
from typing import Optional, Union, Tuple
from torch.utils.checkpoint import checkpoint
from .utils import ntuple
import torch.nn.functional as F
from itertools import chain


def _logconv2d(
    input: Tensor, weight: Tensor, bias=Optional[Tensor], mu: float = 1.0, **kwargs
):
    input = torch.exp(input.mul(mu))
    weight = torch.exp(weight.mul(mu))
    if isinstance(bias, Tensor):
        bias = torch.exp(bias.mul(mu))
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
        return checkpoint(
            _logconv2d, input, weight, bias, use_reentrant=False, mu=mu, **kwargs
        )
    else:
        return _logconv2d(input, weight, bias, mu, **kwargs)


_size_2_t = Union[int, Tuple[int, int]]


class LogConv2d(torch.nn.modules.conv._ConvNd):
    """
    Applies a convolution in the Logarithmic semiring.
    """

    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
        "mu",
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        mu: float = 1.0,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ):
        kernel_size = ntuple(kernel_size, 2)
        stride = ntuple(stride, 2)
        padding = padding if isinstance(padding, str) else ntuple(padding, 2)
        dilation = ntuple(dilation, 2)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            (0, 0),
            groups,
            bias,
            padding_mode,
            device=device,
            dtype=dtype,
        )
        self.mu = mu

    def forward(self, input: Tensor) -> Tensor:
        if self.padding_mode != "zeros":
            return logconv2d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                self.weight,
                self.bias,
                mu=self.mu,
                stride=self.stride,
                padding=(0, 0),
                dilation=self.dilation,
                groups=self.groups,
            )
        return logconv2d(
            input,
            self.weight,
            self.bias,
            mu=self.mu,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


def logconv_parameters(model):
    return chain.from_iterable(
        m.parameters() for m in model.modules() if isinstance(m, LogConv2d)
    )


def nonlogconv_parameters(model):
    return chain.from_iterable(
        m.parameters()
        for m in model.modules()
        if not isinstance(m, LogConv2d) and list(m.children()) == []
    )