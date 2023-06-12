from torch.utils.cpp_extension import load
from pathlib import Path
from glob import glob
import torch

from .utils import *
from .maxplus import *

__version__ = '0.1.0'

_libsemitorch = None
_verbose_jit_load = True


def _load_jit_extension():
    global _libsemitorch
    csources = glob(f"{Path(__file__).parent.absolute()}/**/*.cpp", recursive=True)
    if torch.cuda.is_available():
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


_load_jit_extension()

