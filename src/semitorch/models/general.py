import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional, Callable
from ..utils import ntuple


def named_apply(
    fn: Callable,
    module: nn.Module,
    name="",
    depth_first: bool = True,
    include_root: bool = True,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels of 2D spatial NCHW feature maps"""

    def __init__(self, channels, eps=1e-6, elementwise_affine=True):
        super().__init__(channels, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class LayerScaler(nn.Module):
    """Scale per channel"""

    def __init__(self, init_value: float, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(
            init_value * torch.ones((channels,)), requires_grad=True
        )

    def forward(self, x):
        return x * self.gamma


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    if drop_prob == 0.0 or not training:
        return x
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    keep = x.new_empty(shape).bernoulli_(1 - drop_prob)
    if drop_prob < 1.0 and scale_by_keep:
        keep.div_(1 - drop_prob)
    return keep * x


class DropPath(nn.Module):
    """Drop paths per sample"""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob = {round(self.drop_prob,2):0.2f}"
