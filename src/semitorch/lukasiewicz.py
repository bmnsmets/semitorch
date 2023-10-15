import typing
import torch
from itertools import chain
from torch.utils.checkpoint import checkpoint


def lukasiewicz_fw_kernel_v1(
        x: torch.Tensor,  # [B,Dx]
        a: torch.Tensor,  # [Dy,Dx]
        b: typing.Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Forward pass kernel
    """
    device = x.device
    intermediate = (x.unsqueeze(-2) + a - 1).to(device)
    y = torch.max(torch.max(torch.zeros(intermediate.shape, device=device), intermediate), dim=-1)[0]
    if b is not None:
        y = torch.max(y, b)

    return y


def lukasiewicz_v1(x, a, bias=None):
    assert lukasiewicz_check_tensor_values_in_01_interval(x), \
        f"all inputs x should be contained in the interval [0,1]"
    assert lukasiewicz_check_tensor_values_in_01_interval(a), \
        f"all inputs a should be contained in the interval [0,1]"
    if bias is not None:
        assert lukasiewicz_check_tensor_values_in_01_interval(bias), \
            f"bias should be contained in the interval [0,1]"

    prefix_shape = x.shape[0:-1]
    x = torch.reshape(x, (-1, x.shape[-1]))

    assert (x.device == a.device), \
        f"inputs x and a should be on the same device but are on {x.device} resp. {a.device}"
    x = x.contiguous().requires_grad_(True)
    a = a.contiguous().requires_grad_(True)

    if bias is not None:
        assert (x.device == bias.device), \
            f"inputs x, a, and b should be on the same device but are on {x.device}, {a.device}, resp. {bias.device}"
        bias = bias.contiguous().requires_grad_(True)

    y = checkpoint(lukasiewicz_fw_kernel_v1, x, a, bias)
    return y.reshape((*prefix_shape, -1))


lukasiewicz = lukasiewicz_v1


class Lukasiewicz(torch.nn.Module):
    """
    Applies a Lukasiewicz semiring transformation on [0,1] to the supplied data.
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
        if in_features < 0:
            raise ValueError(f"Invalid in_features: {in_features}, should be > 0")
        if out_features < 0:
            raise ValueError(f"Invalid out_features: {out_features}, should be > 0")

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
        torch.nn.init.eye_(self.weight)

        if self.bias is not None:
            from math import sqrt
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return lukasiewicz(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


def lukasiewicz_parameters(model):
    return chain.from_iterable(
        m.parameters() for m in model.modules() if isinstance(m, Lukasiewicz)
    )


def nonlukasiewicz_parameters(model):
    return chain.from_iterable(
        m.parameters()
        for m in model.modules()
        if not isinstance(m, Lukasiewicz) and list(m.children()) == []
    )


def lukasiewicz_check_tensor_values_in_01_interval(tensor: torch.tensor) -> bool:
    return not torch.lt(tensor, 0.0).any().item() and not torch.gt(tensor, 1.0).any().item()
