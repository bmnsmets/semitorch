import semitorch
import pytest
import torch
from torch.autograd.gradcheck import gradcheck
from semitorch import logconv2d

RNG_SEED = 42

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

