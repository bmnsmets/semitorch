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
from typing import Union, Tuple, Optional, List
from ..utils import ntuple
from .general import LayerNorm2d, LayerScaler, DropPath
from timm.models.registry import register_model


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: Optional[int] = None,
        kernel_size: Union[int, Tuple[int, int]] = 7,
        stride: Union[int, Tuple[int, int]] = 1,
        padding="same",
        mlp_ratio: int = 4,
        drop_prob: float = 0.0,
        layerscale_init_value=1e-6,
        conv_bias: bool = True,
    ):
        super().__init__()
        out_chs = out_chs or in_chs
        self.conv_dw = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=in_chs,
            bias=conv_bias,
        )
        self.norm = LayerNorm2d(out_chs)
        self.mlp = nn.Sequential(
            nn.Linear(out_chs, out_chs * mlp_ratio),
            nn.GELU(),
            nn.Linear(mlp_ratio * out_chs, out_chs),
        )
        self.scale = LayerScaler(layerscale_init_value, out_chs)

        if drop_prob > 0.0:
            self.drop_path = DropPath(drop_prob)
        else:
            self.drop_path = nn.Identity()

        skip_modules = []

        if in_chs != out_chs:
            skip_modules.append(nn.Conv2d(in_chs, out_chs, 1))

        if padding != "same" or stride != 1:
            skip_modules.append(
                nn.AvgPool2d(
                    kernel_size, stride=stride, padding=padding, ceil_mode=True
                )
            )

        if len(skip_modules) > 1:
            self.skip = nn.Sequential(*skip_modules)
        elif len(skip_modules) == 1:
            self.skip = skip_modules[0]
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        # depthwise conv + normalization
        y = self.conv_dw(x)
        y = self.norm(y)

        # pixel-wise MLP
        y = y.permute(0, 2, 3, 1)  # N C H W --> N H W C
        y = self.mlp(y)
        y = self.scale(y)
        y = y.permute(0, 3, 1, 2)  # N H W C --> N C H W

        return self.skip(x) + self.drop_path(y)


class ConvNeXtBlock_MaxPlusMLP(nn.Module):
    pass


class ConvNeXtStage(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        depth: int = 2,
        downsample: Union[None, int] = None,
        drop_probs: Union[float, List[float]] = 0.0,
        make_block=lambda in_chs, out_chs, drop_prob: ConvNeXtBlock(
            in_chs, out_chs, drop_prob=drop_prob
        ),
    ):
        super().__init__()
        if downsample == None:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                LayerNorm2d(in_chs, eps=1e-6, elementwise_affine=True),
                nn.Conv2d(in_chs, out_chs, kernel_size=downsample, stride=downsample),
            )

        drop_probs = ntuple(drop_probs, depth)

        stage_blocks = [
            make_block(out_chs, out_chs, drop_probs[i]) for i in range(depth)
        ]
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class ConvNeXt(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        depths: Tuple[int, ...] = (3, 3, 9, 3),
        channels: Tuple[int, ...] = (96, 192, 384, 768),
        head_drop_rate: float = 0.0,
        path_drop_rate: float = 0.0,
        stem_patch_size: int = 4,
        stage_downsample_rate: int = 2,
        make_block=lambda in_chs, out_chs, drop_prob: ConvNeXtBlock(
            in_chs, out_chs, drop_prob=drop_prob
        ),
    ):
        super.__init__()
        self.in_channels = in_channels
        self.num_classes = 1000

        assert len(depths) == len(
            channels
        ), f"depths and channels should have the same length"
        nstages = len(depths)

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size,
            ),
            LayerNorm2d(channels[0], elementwise_affine=True),
        )

        stages = []
        for i in range(nstages):
            prev_chs = channels[0] if i == 0 else channels[i - 1]
            next_chs = channels[i]
            stages.append(
                ConvNeXtStage(
                    in_chs=prev_chs,
                    out_chs=next_chs,
                    depth=depths[i],
                    downsample=stage_downsample_rate,
                    make_block=make_block,
                )
            )

        self.stages = nn.ModuleList(stages)

        self.head = nn.Sequential(
            LayerNorm2d(channels[-1], elementwise_affine=True),
        )
