from torch.utils.cpp_extension import load
from pathlib import Path
from glob import glob

_libsemitorch = None

def _load_jit_extension():
    csources = glob(f'{Path(__file__).parent.absolute()}/**/*.cpp', recursive=True)
    cudasources = glob(f'{Path(__file__).parent.absolute()}/**/*.cu', recursive=True)
    global _libsemitorch 
    _libsemitorch = load(name='libsemitorch', sources=[*csources, *cudasources], verbose=True, with_cuda=True)

_load_jit_extension()
