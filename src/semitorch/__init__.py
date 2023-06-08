from torch.utils.cpp_extension import load
from pathlib import Path
from glob import glob

csources = glob(f'{Path(__file__).parent.absolute()}/**/*.cpp', recursive=True)
cudasources = glob(f'{Path(__file__).parent.absolute()}/**/*.cu', recursive=True)

libsemitorch = load(name='libsemitorch', sources=[*csources, *cudasources], verbose=True, with_cuda=True)
