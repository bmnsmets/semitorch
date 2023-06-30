r""" Semiring based variants of ConvNeXt

Based on the following sources:

* `A ConvNet for the 2020s` - https://arxiv.org/pdf/2201.03545.pdf
    @article{liu2022convnet,
    author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
    title   = {A ConvNet for the 2020s},
    journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year    = {2022},
    }

    Original code and weights from: https://github.com/facebookresearch/ConvNeXt, original copyright notice:
    # ConvNeXt
    # Copyright (c) Meta Platforms, Inc. and affiliates.
    # All rights reserved.
    # This source code is licensed under the MIT license

* `timm (pytorch-image-models)` - https://github.com/huggingface/pytorch-image-models
    @misc{rw2019timm,
    author = {Ross Wightman},
    title = {PyTorch Image Models},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    doi = {10.5281/zenodo.4414861},
    howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
    }
    Licenced under the Apache 2.0 license, original copyright notice:
    Modifications and additions for timm hacked together by / Copyright 2022, Ross Wightman

Our modifications substantially change the workings of the depthwise convolution and MLP operators in the networks.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional
from ..utils import ntuple
from .general import LayerNorm2d


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: Optional[int] = None,
        kernel_size: Union[int, Tuple[int, int]] = 7,
        stride: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs or in_chs
        self.kernel_size = ntuple(kernel_size, 2)
        self.stride = ntuple(stride, 2)
        self.dilation = ntuple(dilation, 2)
        self.mlp_ratio = mlp_ratio

    def forward(self, x):
        y = self.conv_dw(x)
        y = self.norm(y)
        y = self.mlp(y)
        return self.skip(x) + y


class ConvNeXtBlock_Classic(ConvNeXtBlock):
    def __init__(
        self,
        in_chs: int,
        out_chs: Optional[int] = None,
        kernel_size: Union[int, Tuple[int, int]] = 7,
        stride: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        mlp_ratio: int = 4,
    ):
        super().__init__(
            in_chs=in_chs,
            out_chs=out_chs,
            kernel_size=ntuple(kernel_size, 2),
            stride=ntuple(stride, 2),
            dilation=ntuple(dilation, 2),
        )
        out_chs = out_chs or in_chs
        kernel_size = ntuple(kernel_size, 2)
        stride = ntuple(stride, 2)
        dilation = ntuple(dilation, 2)

        self.conv_dw = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=in_chs,
        )


class ConvNeXtBlock_MaxPlusMLP(ConvNeXtBlock):
    pass


class ConvNeXtStage(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        depth: int = 2,
        downsample: Union[None, int] = None,
        block_type=ConvNeXtBlock_Classic,
    ):
        super().__init__()
        if downsample == None:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                LayerNorm2d(in_chs, eps=1e-6, elementwise_affine=True),
                nn.Conv2d(in_chs, out_chs, kernel_size=downsample, stride=downsample),
            )

        stage_blocks = []

        for i in range(depth):
            pass

        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x
