from torch.cuda import is_available as is_cuda_available
from torch.utils.cpp_extension import load
from pathlib import Path
from glob import glob
import taichi as ti

from .utils import *
from .maxplus import maxplus, MaxPlus, maxplus_parameters, nonmaxplus_parameters
from .minplus import minplus, MinPlus, minplus_parameters, nonminplus_parameters
from .models.general import LayerNorm2d, LayerScaler, DropPath
from .models.convnext import (
    ConvNeXtBlock,
    ConvNeXtBlock_MaxPlusMLP,
    ConvNeXtStage,
    ConvNeXt,
)
from .optim import MultiOptimizer, MultiLRScheduler
from .logconv import logconv2d, LogConv2d, logconv_parameters, nonlogconv_parameters
from .logplus import logplus, LogPlus, logplus_parameters, nonlogplus_parameters

__version__ = '0.1.1'

_libsemitorch = None
_verbose_jit_load = True


def _load_jit_extension():
    global _libsemitorch
    csources = glob(f"{Path(__file__).parent.absolute()}/**/*.cpp", recursive=True)
    if is_cuda_available():
        cudasources = glob(
            f"{Path(__file__).parent.absolute()}/**/*.cu", recursive=True
        )
        _libsemitorch = load(
            name="libsemitorch",
            sources=[*csources, *cudasources],
            extra_cflags=["-DWITH_CUDA=1"],
            extra_cuda_cflags=["-DWITH_CUDA=1"],
            verbose=_verbose_jit_load,
            with_cuda=True,
        )
    else:
        _libsemitorch = load(
            name="libsemitorch", sources=csources, verbose=_verbose_jit_load
        )


if is_cuda_available():
    ti.init(arch=ti.gpu)
else:
    ti.init(arch=ti.cpu)
