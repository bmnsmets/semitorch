import torch
from itertools import chain
from torch.nn.functional import linear
from torch.utils.checkpoint import checkpoint


def semilog_shifted_scaled_fw_kernel_v1(
        x: torch.tensor,  # [B,Dx]
        a: torch.tensor,  # [Dy,Dx]
        mu: torch.tensor,  # [1, 1]
):
    """
    Forward pass kernel
    """
    device = x.device
    scaling_const = torch.exp(mu) - 1

    x = torch.exp(mu * x) - torch.ones(x.shape, device=device)
    x_div = torch.ones(x.shape, device=device) * scaling_const
    x = x / x_div
    a = torch.exp(mu * a) - torch.ones(a.shape, device=device)
    a_div = torch.ones(a.shape, device=device) * scaling_const
    a = a / a_div
    v = linear(x, a)
    v_mul = torch.ones(v.shape, device=device) * scaling_const
    v = v * v_mul + torch.ones(v.shape, device=device)
    return torch.log(v) / mu


def semilog_shifted_scaled_v1(x, a, mu, bias=None):
    prefix_shape = x.shape[0:-1]
    x = torch.reshape(x, (-1, x.shape[-1]))

    assert (x.device == a.device) and (x.device == mu.device), \
        f"inputs x, a, and mu should be on the same device but are on {x.device} resp. {a.device} resp. {mu.device}"
    x = x.contiguous().requires_grad_(True)
    a = a.contiguous().requires_grad_(True)
    mu = mu.contiguous().requires_grad_(True)

    y = checkpoint(semilog_shifted_scaled_fw_kernel_v1, x, a, mu)
    if bias is not None:
        y.add_(bias)
    return y.reshape((*prefix_shape, -1))


semilog_shifted_scaled = semilog_shifted_scaled_v1


class SemiLogShiftedScaled(torch.nn.Module):
    """
    Applies the logarithmic semiring transformation to the supplied data, shifted and scaled.
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
            k: float = 2.0,
            mu: float = 1.0,
            device=None,
            dtype=None,
    ) -> None:
        if in_features < 0:
            raise ValueError(f"Invalid in_features: {in_features}, should be > 0")
        if out_features < 0:
            raise ValueError(f"Invalid out_features: {out_features}, should be > 0")
        if k <= 0.0:
            raise ValueError(f"Invalid k: {k}, should be > 0")
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
        self.reset_parameters(k=k, mu=mu)

    def reset_parameters(self, k: float, mu: float) -> None:
        semilog_init_fair_(self.weight, k=k, mu=mu)

        torch.nn.init.constant_(self.mu, mu)

        if self.bias is not None:
            torch.nn.init.constant_(self.bias, k * sign(mu))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return semilog_shifted_scaled(input, self.weight, self.mu, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


def semilog_init_fair_(w: torch.Tensor, k: float, mu: float) -> torch.Tensor:
    with torch.no_grad():
        torch.nn.init.eye_(w).add_(-1).mul_(k * sign(mu))
    return w


def sign(x: float) -> int:
    return bool(x > 0) - bool(x < 0)


def semilog_shifted_scaled_parameters(model):
    return chain.from_iterable(
        m.parameters() for m in model.modules() if isinstance(m, SemiLogShiftedScaled)
    )


def nonsemilog_shifted_scaled_parameters(model):
    return chain.from_iterable(
        m.parameters()
        for m in model.modules()
        if not isinstance(m, SemiLogShiftedScaled) and list(m.children()) == []
    )