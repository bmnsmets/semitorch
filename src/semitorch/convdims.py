import taichi as ti
from typing import Optional, Union, Tuple, TypeVar


@ti.dataclass
class ConvDims:
    input_size: ti.types.vector(2, ti.i32)
    kernel_size: ti.types.vector(2, ti.i32)
    in_channels: ti.i32
    out_channels: ti.i32
    padding: ti.types.vector(4, ti.i32)
    stride: ti.types.vector(2, ti.i32)
    dilation: ti.types.vector(2, ti.i32)
    groups: ti.i32

    def output_size(self):
        insize = [self.input_size[i] + self.padding[2*i+1] + self.padding[2*i]
                  for i in range(2)]
        ksize = [(self.kernel_size[i] - 1) * self.dilation[i] + 1 for i in range(2)]
        return [(insize[i] - ksize[i]) // self.stride[i] + 1 for i in range(2)]


def create_convdims2d(
    input_size: Tuple[int, int],
    kernel_size: Tuple[int, int],
    in_channels: int,
    out_channels: int,
    padding: Union[str, int, Tuple[int, int, int, int]] = 0,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1
):
    stride = ntuple(stride, 2)
    dilation = ntuple(dilation, 2)

    if isinstance(padding, str):
        assert padding in {'same'}, 'padding = \'{padding}\' is not a valid argument.'
        if padding == 'same':
            padding = [0, 0, 0, 0]
            for d, k, i in zip(dilation, kernel_size,
                               range(len(kernel_size))):
                total_padding = d * (k - 1)
                left_pad = total_padding // 2
                padding[2*i] = left_pad
                padding[2*i+1] = total_padding - left_pad

    padding = ntuple(padding, 4)

    return ConvDims(
        input_size=input_size,
        kernel_size=kernel_size,
        in_channels=in_channels,
        out_channels=out_channels,
        padding=padding,
        stride=stride,
        dilation=dilation,
        groups=groups
    )