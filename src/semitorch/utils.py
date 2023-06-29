#
#
#
import time
import torch
import collections
from itertools import repeat
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import ipywidgets as widgets
import numpy as np


class Timer:
    def __init__(self):
        pass

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, *exc_info):
        t = time.time() - self.t0
        if t > 1.0:
            print(f'Elapsed: {t:>3.2f} s')
        elif t > 1e-3:
            print(f'Elapsed: {(1e3 * t):>3.2f} ms')
        elif t > 1e-6:
            print(f'Elapsed: {(1e6 * t):>3.2f} μs')
        else:
            print(f'Elapsed: {(1e9 * t):>3.2f} ns')


class CUDATimer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.stop = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()

    def __exit__(self, *exc_info):
        self.stop.record()
        torch.cuda.synchronize()
        t = self.start.elapsed_time(self.stop) / 1000

        if t > 1.0:
            print(f'Elapsed: {t:>3.2f} s')
        elif t > 1e-3:
            print(f'Elapsed: {(1e3 * t):>3.2f} ms')
        elif t > 1e-6:
            print(f'Elapsed: {(1e6 * t):>3.2f} μs')
        else:
            print(f'Elapsed: {(1e9 * t):>3.2f} ns')


def ntuple(x, n: int):
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    else:
        return tuple(repeat(x, n))


def mnistplot(x):
    assert x.ndim >= 2

    if x.ndim > 2:
        x = torch.reshape(x, (-1, x.shape[-2], x.shape[-1]))

    vmin = 0
    vmax = 1

    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    else:
        x = np.array(x)

    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]

    if len(x.shape) == 2:
        plt.axis('off')
        plt.imshow(x, cmap='gray', vmin=vmin, vmax=vmax)
    elif len(x.shape) == 3:
        ncols = 10
        nrows = -(x.shape[0] // -4)

        fig = plt.figure(figsize=(1 * ncols, 1 * nrows))
        grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols))

        for ax in grid:
            ax.axis('off')

        for i, ax in enumerate(grid):
            if i >= x.shape[0]:
                break
            ax.imshow(x[i], cmap='gray', vmin=vmin, vmax=vmax)

        plt.show()
