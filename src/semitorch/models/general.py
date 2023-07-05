import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels of 2D spatial NCHW feature maps"""

    def __init__(self, channels, eps=1e-6, affine=True):
        super().__init__(channels, eps=eps, elementwise_affine=affine)

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
    

class DownSample(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1, dilation=1):
        self.pool = nn.Identity()
        
        if in_chs != out_chs:
            self.conv = nn.Conv2d(in_chs, out_chs, 1, stride=1)
        
    def forward(self, x):
        x = self.pool(x) if hasattr(self, 'pool') else x
        x = self.conv(x) if hasattr(self, 'conv') else x
        return x