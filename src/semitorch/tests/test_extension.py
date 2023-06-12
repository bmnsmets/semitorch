import semitorch
import pytest
import torch

def test_cuda_version():
    assert semitorch._libsemitorch.cuda_version() > 11070