import pytest
import torch
from semitorch import rescaleNonNegativeToUnitInterval
from numpy import tanh

DEFAULT_RNG_SEED = 0
torch.manual_seed(DEFAULT_RNG_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def test_rescaleNonNegativeToUnitInterval_should_error_on_negative_inputs() -> None:
    x = [-1, 0, 2]

    with pytest.raises(AssertionError):
        x_transform, maximum = rescaleNonNegativeToUnitInterval(x)


def test_rescaleNonNegativeToUnitInterval_should_error_on_negative_maximum() -> None:
    x = [0, 2, 3]
    maximum = -1

    with pytest.raises(AssertionError):
        x_transform, _ = rescaleNonNegativeToUnitInterval(x, maximum=maximum)


def test_rescaleNonNegativeToUnitInterval() -> None:
    x = [0, 1, 2, 3, 4]

    x_transform, maximum = rescaleNonNegativeToUnitInterval(x)
    test = [tanh(item / 4) for item in x]

    assert maximum == 4
    assert x_transform == test
    assert all([item >= 0 for item in x_transform])
    assert all([item <= 1 for item in x_transform])
