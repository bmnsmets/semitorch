import torch
from itertools import chain
from torch.utils.checkpoint import checkpoint


def minplusmp_fw_kernel_v1(
        x: torch.Tensor,  # [B,Dx]
        a: torch.Tensor,  # [Dy,Dx]
):
    """
    Forward pass kernel
    """
    return torch.min(x.unsqueeze(-2) + a, dim=-1)[0]


def minplusmp_v1(x, a, bias=None):
    prefix_shape = x.shape[0:-1]
    x = torch.reshape(x, (-1, x.shape[-1]))

    assert (x.device == a.device), \
        f"inputs x and a should be on the same device but are on {x.device} resp. {a.device}"
    x = x.contiguous().requires_grad_(True)
    a = a.contiguous().requires_grad_(True)

    y = checkpoint(minplusmp_fw_kernel_v1, x, a)
    if bias is not None:
        y.add_(bias)
    return y.reshape((*prefix_shape, -1))


minplusmp = minplusmp_v1


class MinPlusMP(torch.nn.Module):
    """
    Applies a tropical min-plus transformation to the supplied data.
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
            k: float = 2.0,
            device=None,
            dtype=None,
    ) -> None:
        if in_features < 0:
            raise ValueError(f"Invalid in_features: {in_features}, should be > 0")
        if out_features < 0:
            raise ValueError(f"Invalid out_features: {out_features}, should be > 0")
        if k <= 0.0:
            raise ValueError(f"Invalid k: {k}, should be > 0")

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
        self.reset_parameters(k=k)

    def reset_parameters(self, k: float,) -> None:
        minplusmp_init_fair_(self.weight, k=k)

        if self.bias is not None:
            torch.nn.init.constant_(self.bias, -k)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return minplusmp(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


def minplusmp_init_fair_(w: torch.Tensor, k: float) -> torch.Tensor:
    with torch.no_grad():
        torch.nn.init.eye_(w).add_(-1).mul_(-k)
    return w


def minplusmp_parameters(model):
    return chain.from_iterable(
        m.parameters() for m in model.modules() if isinstance(m, MinPlusMP)
    )


def nonminplusmp_parameters(model):
    return chain.from_iterable(
        m.parameters()
        for m in model.modules()
        if not isinstance(m, MinPlusMP) and list(m.children()) == []
    )