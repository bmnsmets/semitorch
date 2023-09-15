import torch
from typing import Tuple
import taichi as ti
import taichi.math as tm
import math
from itertools import chain


@ti.kernel
def maxplus_inference_kernel_v1(
    y: ti.types.ndarray(ndim=2),  # [B,Dy]
    x: ti.types.ndarray(ndim=2),  # [B,Dx]
    a: ti.types.ndarray(ndim=2),  # [Dy,Dx]
):
    """
    Kernel for inference only, the locations of the maxima (the `hits`) are not recorded.
    """
    for b, i in y:
        v = -tm.inf
        for j in range(a.shape[-1]):
            v = tm.max(v, x[b, j] + a[i, j])
        y[b, i] = v


@ti.kernel
def maxplus_fw_kernel_v1(
    y: ti.types.ndarray(ndim=2),  # [B,Dy]
    hits: ti.types.ndarray(dtype=ti.i32, ndim=2),  # [B,Dy]
    x: ti.types.ndarray(ndim=2),  # [B,Dx]
    a: ti.types.ndarray(ndim=2),  # [Dy,Dx]
):
    """
    Forward pass kernel
    """
    for b, i in y:
        v = -tm.inf
        hit: ti.i32 = -1
        for j in range(a.shape[-1]):
            w = x[b, j] + a[i, j]
            if w > v:
                v = w
                hit = j
        y[b, i] = v
        hits[b, i] = hit


@ti.kernel
def maxplus_bw_x_kernel_v1(
    gradx: ti.types.ndarray(ndim=2),  # [B,Dx]
    hits: ti.types.ndarray(dtype=ti.i32, ndim=2),  # [B,Dy]
    grady: ti.types.ndarray(ndim=2),  # [B,Dy]
):
    """
    Backward pass kernel for the `x` input
    """
    for b, i in gradx:
        val = 0 * gradx[b, i]
        for j in range(hits.shape[1]):
            if hits[b, j] == i:
                val = val + grady[b, j]
        gradx[b, i] = val


@ti.kernel
def maxplus_bw_a_kernel_v1(
    grada: ti.types.ndarray(ndim=2),  # [Dy,Dx]
    hits: ti.types.ndarray(dtype=ti.i32, ndim=2),  # [B,Dy]
    grady: ti.types.ndarray(ndim=2),  # [B,Dy]
):
    """
    Backward pass kernel for the `a` input
    """
    for j, i in grada:
        val = 0.0
        for b in range(hits.shape[0]):
            if hits[b, j] == i:
                val = val + grady[b, j]
        grada[j, i] = val


class MaxPlusFunction_v1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, grad_enabled=True):
        assert (
            x.device == a.device
        ), "inputs x and a should be on the same device but are on f{x.device} resp. f{a.device}"
        x = x.contiguous()
        a = a.contiguous()

        y = torch.empty((*x.shape[0:-1], a.shape[0]), device=x.device, dtype=x.dtype)

        ctx.a_shape = a.shape
        ctx.x_shape = x.shape

        if (x.requires_grad or a.requires_grad) and grad_enabled:
            hits = torch.empty_like(x, dtype=torch.int32)
            maxplus_fw_kernel_v1(y, hits, x, a)
            ctx.save_for_backward(hits)
        else:
            maxplus_inference_kernel_v1(y, x, a)

        ti.sync()
        return y

    @staticmethod
    def backward(ctx, grady):
        (hits,) = ctx.saved_tensors

        grada = torch.empty(ctx.a_shape, dtype=grady.dtype, device=grady.device)
        gradx = torch.empty(ctx.x_shape, dtype=grady.dtype, device=grady.device)

        maxplus_bw_a_kernel_v1(grada, hits, grady)
        maxplus_bw_x_kernel_v1(gradx, hits, grady)

        ti.sync()
        return gradx, grada, None


def maxplus_v1(x, a, bias=None):
    prefix_shape = x.shape[0:-1]
    x = torch.reshape(x, (-1, x.shape[-1]))
    y = MaxPlusFunction_v1.apply(x, a, torch.is_grad_enabled())
    if bias != None:
        y.add_(bias)
    return y.reshape((*prefix_shape, -1))


maxplus = maxplus_v1


class MaxPlus(torch.nn.Module):
    """
    Applies a tropical max-plus transformation to the supplied data.
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        maxplus_init_fair_(self.weight, k=-1)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, -1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return maxplus(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


def maxplus_init_fair_(w: torch.Tensor, k: float = -1) -> torch.Tensor:
    with torch.no_grad():
        #torch.nn.init.eye_(w).add_(-1).mul_(-k)
        torch.nn.init.kaiming_uniform_(w).sub_(k).mul_(torch.eye(*w.shape).add_(-1))
    return w


def maxplus_parameters(model):
    return chain.from_iterable(
        m.parameters() for m in model.modules() if isinstance(m, MaxPlus)
    )


def nonmaxplus_parameters(model):
    return chain.from_iterable(
        m.parameters()
        for m in model.modules()
        if not isinstance(m, MaxPlus) and list(m.children()) == []
    )
