import torch
from itertools import chain
from torch.nn.functional import linear
from torch.utils.checkpoint import checkpoint


def logplusmp_fw_kernel_v1(
        x: torch.tensor,  # [B,Dx]
        a: torch.tensor,  # [Dy,Dx]
        mu: torch.tensor,  # [1, 1]
):
    """
    Forward pass kernel
    """
    x = torch.exp(mu * x)
    a = torch.exp(mu * a)
    v = linear(x, a)
    return torch.log(v) / mu


def logplusmp_v1(x, a, mu, bias=None):
    prefix_shape = x.shape[0:-1]
    x = torch.reshape(x, (-1, x.shape[-1]))

    assert (x.device == a.device) and (x.device == mu.device), \
        f"inputs x, a, and mu should be on the same device but are on {x.device} resp. {a.device} resp. {mu.device}"
    x = x.contiguous().requires_grad_(True)
    a = a.contiguous().requires_grad_(True)
    mu = mu.contiguous().requires_grad_(True)

    y = checkpoint(logplusmp_fw_kernel_v1, x, a, mu)
    if bias is not None:
        y.add_(bias)
    return y.reshape((*prefix_shape, -1))


logplusmp = logplusmp_v1


class LogPlusMP(torch.nn.Module):
    """
    Applies the logarithmic semiring transformation to the supplied data.
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor
    mu: torch.Tensor

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            mu: float = 1.0,
            device=None,
            dtype=None,
    ) -> None:
        if in_features < 0:
            raise ValueError(f"Invalid in_features: {in_features}, should be > 0")
        if out_features < 0:
            raise ValueError(f"Invalid out_features: {out_features}, should be > 0")
        if mu == 0:
            raise ValueError(f"Invalid mu: {mu}, should be unequal to 0")

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.mu = torch.nn.Parameter(torch.empty(1, **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters(mu=mu)

    def reset_parameters(self, mu: float) -> None:
        from math import sqrt

        torch.nn.init.constant_(self.mu, mu)
        torch.nn.init.kaiming_uniform_(self.weight, a=sqrt(5))

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return logplusmp(input, self.weight, self.mu, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


def logplusmp_parameters(model):
    return chain.from_iterable(
        m.parameters() for m in model.modules() if isinstance(m, LogPlusMP)
    )


def nonlogplusmp_parameters(model):
    return chain.from_iterable(
        m.parameters()
        for m in model.modules()
        if not isinstance(m, LogPlusMP) and list(m.children()) == []
    )
