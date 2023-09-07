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
from .general import LayerNorm2d, LayerScaler, DropPath, named_apply
from timm.models.registry import register_model
from functools import partial
from ..maxplus import MaxPlus
from ..logconv import logconv2d, LogConv2d


@torch.no_grad()
def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d) or isinstance(module, LogConv2d):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        nn.init.zeros_(module.bias)
        if name and "head." in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


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
            MaxPlus(out_chs, out_chs * mlp_ratio),
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


class LogConvNeXtBlock(nn.Module):
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
        mu: float = 1.0,
    ):
        super().__init__()
        out_chs = out_chs or in_chs
        self.logconv_dw = LogConv2d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=in_chs,
            bias=conv_bias,
            mu=mu,
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
        y = self.logconv_dw(x)
        y = self.norm(y)

        # pixel-wise MLP
        y = y.permute(0, 2, 3, 1)  # N C H W --> N H W C
        y = self.mlp(y)
        y = self.scale(y)
        y = y.permute(0, 3, 1, 2)  # N H W C --> N C H W

        return self.skip(x) + self.drop_path(y)


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
        if downsample == None or downsample == 1.0:
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
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: Tuple[int, ...] = (3, 3, 9, 3),
        channels: Tuple[int, ...] = (96, 192, 384, 768),
        head_drop_rate: float = 0.0,
        path_drop_rate: float = 0.0,
        patch_size: int = 4,
        stage_downsample_rate: int = 2,
        make_block=lambda in_chs, out_chs, drop_prob: ConvNeXtBlock(
            in_chs, out_chs, drop_prob=drop_prob
        ),
        head_init_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_chans
        self.num_classes = 1000
        self.head_init_scale = head_init_scale

        assert len(depths) == len(
            channels
        ), f"depths and channels should have the same length"
        nstages = len(depths)

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_chans,
                channels[0],
                kernel_size=patch_size,
                stride=patch_size,
            ),
            LayerNorm2d(channels[0], elementwise_affine=True),
        )

        stages = []
        for i in range(nstages):
            prev_chs = channels[0] if i == 0 else channels[i - 1]
            next_chs = channels[i]
            downsample = stage_downsample_rate if i > 0 else 1.0
            stages.append(
                ConvNeXtStage(
                    in_chs=prev_chs,
                    out_chs=next_chs,
                    depth=depths[i],
                    downsample=downsample,
                    make_block=make_block,
                    drop_probs=path_drop_rate,
                )
            )

        self.stages = nn.ModuleList(stages)

        self.head = nn.Sequential(
            LayerNorm2d(channels[-1], elementwise_affine=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # N C H W -> N C 1 1
            nn.Flatten(),
            nn.Dropout(head_drop_rate),
            nn.Linear(channels[-1], num_classes),
        )

        self.reset_parameters()

    def forward(self, x):
        x = self.stem(x)
        for i in range(len(self.stages)):
            x = self.stages[i](x)
        x = self.head(x)
        return x

    def reset_parameters(self):
        named_apply(partial(_init_weights, head_init_scale=self.head_init_scale), self)


@register_model
def convnext_st_classic_atto(pretrained=False, **kwargs) -> ConvNeXt:
    model_args = dict(depths=(2, 2, 6, 2), channels=(40, 80, 160, 320))
    model = ConvNeXt(**dict(model_args, **kwargs))
    return model


@register_model
def convnext_st_maxplusmlp_atto(pretrained=False, **kwargs) -> ConvNeXt:
    model_args = dict(
        depths=(2, 2, 6, 2),
        channels=(40, 80, 160, 320),
        make_block=lambda in_chs, out_chs, drop_prob: ConvNeXtBlock_MaxPlusMLP(
            in_chs, out_chs, drop_prob=drop_prob
        ),
    )
    model = ConvNeXt(**dict(model_args, **kwargs))
    return model


@register_model
def logconvnext_st_atto(pretrained=False, **kwargs) -> ConvNeXt:
    model_args = dict(
        depths=(2, 2, 6, 2),
        channels=(40, 80, 160, 320),
        make_block=lambda in_chs, out_chs, drop_prob: LogConvNeXtBlock(
            in_chs, out_chs, drop_prob=drop_prob, mu=1.0
        ),
    )
    model = ConvNeXt(**dict(model_args, **kwargs))
    return model
