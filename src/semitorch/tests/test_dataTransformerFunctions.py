import pytest
import torch
from semitorch import (
    rescaleNonNegativeToUnitInterval,
    rescaleNonPositiveToUnitInterval,
    rescaleClosedIntervalToUnitInterval,
    rescaleRealsToUnitInterval,
    oneHotEncode,
)
from numpy import exp, tanh

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


def test_rescaleNonNegativeToUnitInterval_provided_maximum() -> None:
    x = [0, 1, 2, 3, 4]

    x_transform, maximum = rescaleNonNegativeToUnitInterval(x, maximum=3)
    test = [tanh(item / 3) for item in x]

    assert maximum == 3
    assert x_transform == test
    assert all([item >= 0 for item in x_transform])
    assert all([item <= 1 for item in x_transform])


def test_rescaleNonPositiveToUnitInterval_should_error_on_negative_inputs() -> None:
    x = [-1, 0, 2]

    with pytest.raises(AssertionError):
        x_transform, minimum = rescaleNonPositiveToUnitInterval(x)


def test_rescaleNonPositiveToUnitInterval_should_error_on_negative_maximum() -> None:
    x = [0, -2, -3]
    minimum = 1

    with pytest.raises(AssertionError):
        x_transform, _ = rescaleNonPositiveToUnitInterval(x, minimum=minimum)


def test_rescaleNonPositiveToUnitInterval() -> None:
    x = [0, -1, -2, -3, -4]

    x_transform, minimum = rescaleNonPositiveToUnitInterval(x)
    test = [tanh(item / -4) for item in x]

    assert minimum == -4
    assert x_transform == test
    assert all([item >= 0 for item in x_transform])
    assert all([item <= 1 for item in x_transform])


def test_rescaleNonPositiveToUnitInterval_provided_minimum() -> None:
    x = [0, -1, -2, -3, -4]

    x_transform, minimum = rescaleNonPositiveToUnitInterval(x, minimum=-3)
    test = [tanh(item / -3) for item in x]

    assert minimum == -3
    assert x_transform == test
    assert all([item >= 0 for item in x_transform])
    assert all([item <= 1 for item in x_transform])


def test_rescaleClosedIntervalToUnitInterval_should_error_on_equal_maximum_minimum() -> None:
    x = [0, 2, -3]
    maximum = 1
    minimum = 1

    with pytest.raises(AssertionError):
        x_transform, _ = rescaleClosedIntervalToUnitInterval(x, maximum=maximum, minimum=minimum)


def test_rescaleClosedIntervalToUnitInterval() -> None:
    x = [0, 1, -2, -3, 4]

    x_transform, maximum, minimum = rescaleClosedIntervalToUnitInterval(x)
    test = [(item - (-3)) / (4 - (-3)) for item in x]

    assert maximum == 4
    assert minimum == -3
    assert x_transform == test
    assert all([item >= 0 for item in x_transform])
    assert all([item <= 1 for item in x_transform])


def test_rescaleClosedIntervalToUnitInterval_should_error_on_provided_maximum_too_large() -> None:
    x = [0, 1, -2, -3, 4]

    with pytest.raises(AssertionError):
        rescaleClosedIntervalToUnitInterval(x, maximum=3)


def test_rescaleClosedIntervalToUnitInterval_should_error_on_provided_minimum_too_small() -> None:
    x = [0, 1, -2, -3, 4]

    with pytest.raises(AssertionError):
        rescaleClosedIntervalToUnitInterval(x, minimum=-2)


def test_rescaleClosedIntervalToUnitInterval_should_error_on_provided_maximum_too_large_minimum_too_small() -> None:
    x = [0, 1, -2, -3, 4]

    with pytest.raises(AssertionError):
        rescaleClosedIntervalToUnitInterval(x, maximum=3, minimum=-2)


def test_rescaleClosedIntervalToUnitInterval_provided_maximum() -> None:
    x = [0, 1, -2, -3, 4]

    x_transform, maximum, minimum = rescaleClosedIntervalToUnitInterval(x, maximum=5)
    test = [(item - (-3)) / (5 - (-3)) for item in x]

    assert maximum == 5
    assert minimum == -3
    assert x_transform == test
    assert all([item >= 0 for item in x_transform])
    assert all([item <= 1 for item in x_transform])


def test_rescaleClosedIntervalToUnitInterval_provided_minimum() -> None:
    x = [0, 1, -2, -3, 4]

    x_transform, maximum, minimum = rescaleClosedIntervalToUnitInterval(x, minimum=-5)
    test = [(item - (-5)) / (4 - (-5)) for item in x]

    assert maximum == 4
    assert minimum == -5
    assert x_transform == test
    assert all([item >= 0 for item in x_transform])
    assert all([item <= 1 for item in x_transform])


def test_rescaleClosedIntervalToUnitInterval_provided_maximum_minimum() -> None:
    x = [0, 1, -2, -3, 4]

    x_transform, maximum, minimum = rescaleClosedIntervalToUnitInterval(x, maximum=5, minimum=-5)
    test = [(item - (-5)) / (5 - (-5)) for item in x]

    assert maximum == 5
    assert minimum == -5
    assert x_transform == test
    assert all([item >= 0 for item in x_transform])
    assert all([item <= 1 for item in x_transform])


def test_rescaleRealsToUnitInterval() -> None:
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + exp(-x))

    x = [0, 1, -2, -3, 4]

    x_transform, maximum, minimum = rescaleRealsToUnitInterval(x)
    test = [sigmoid(2 * (item - (-3)) / (4 - (-3)) - 1) for item in x]

    assert maximum == 4
    assert minimum == -3
    assert x_transform == test
    assert all([item >= 0 for item in x_transform])
    assert all([item <= 1 for item in x_transform])


def test_rescaleRealsToUnitInterval_provided_maximum() -> None:
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + exp(-x))

    x = [0, 1, -2, -3, 4]

    x_transform, maximum, minimum = rescaleRealsToUnitInterval(x, maximum=5)
    test = [sigmoid(2 * (item - (-3)) / (5 - (-3)) - 1) for item in x]

    assert maximum == 5
    assert minimum == -3
    assert x_transform == test
    assert all([item >= 0 for item in x_transform])
    assert all([item <= 1 for item in x_transform])


def test_rescaleRealsToUnitInterval_provided_minimum() -> None:
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + exp(-x))

    x = [0, 1, -2, -3, 4]

    x_transform, maximum, minimum = rescaleRealsToUnitInterval(x, minimum=-5)
    test = [sigmoid(2 * (item - (-5)) / (4 - (-5)) - 1) for item in x]

    assert maximum == 4
    assert minimum == -5
    assert x_transform == test
    assert all([item >= 0 for item in x_transform])
    assert all([item <= 1 for item in x_transform])


def test_rescaleRealsToUnitInterval_provided_maximum_minimum() -> None:
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + exp(-x))

    x = [0, 1, -2, -3, 4]

    x_transform, maximum, minimum = rescaleRealsToUnitInterval(x, maximum=5, minimum=-5)
    test = [sigmoid(2 * (item - (-5)) / (5 - (-5)) - 1) for item in x]

    assert maximum == 5
    assert minimum == -5
    assert x_transform == test
    assert all([item >= 0 for item in x_transform])
    assert all([item <= 1 for item in x_transform])


def test_rescaleRealsToUnitInterval_provided_maximum_2() -> None:
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + exp(-x))

    x = [0, 1, -2, -3, 4, 10, -10]

    x_transform, maximum, minimum = rescaleRealsToUnitInterval(x, maximum=5)
    test = [sigmoid(2 * (item - (-10)) / (5 - (-10)) - 1) for item in x]

    assert maximum == 5
    assert minimum == -10
    assert x_transform == test
    assert all([item >= 0 for item in x_transform])
    assert all([item <= 1 for item in x_transform])


def test_rescaleRealsToUnitInterval_provided_minimum_2() -> None:
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + exp(-x))

    x = [0, 1, -2, -3, 4, 10, -10]

    x_transform, maximum, minimum = rescaleRealsToUnitInterval(x, minimum=-5)
    test = [sigmoid(2 * (item - (-5)) / (10 - (-5)) - 1) for item in x]

    assert maximum == 10
    assert minimum == -5
    assert x_transform == test
    assert all([item >= 0 for item in x_transform])
    assert all([item <= 1 for item in x_transform])


def test_rescaleRealsToUnitInterval_provided_maximum_minimum_2() -> None:
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + exp(-x))

    x = [0, 1, -2, -3, 4, 10, -10]

    x_transform, maximum, minimum = rescaleRealsToUnitInterval(x, maximum=5, minimum=-5)
    test = [sigmoid(2 * (item - (-5)) / (5 - (-5)) - 1) for item in x]

    assert maximum == 5
    assert minimum == -5
    assert x_transform == test
    assert all([item >= 0 for item in x_transform])
    assert all([item <= 1 for item in x_transform])


def test_oneHotEncode() -> None:
    x = [0, 4, 2, 3, 1]

    x_transform = oneHotEncode(x)
    test = [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0]]

    assert x_transform == test
    assert all([[item >= 0 for item in arr] for arr in x_transform])
    assert all([[item <= 1 for item in arr] for arr in x_transform])


def test_oneHotEncode_2() -> None:
    x = [0, 2, 2, 8, 2]

    x_transform = oneHotEncode(x)
    test = [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]

    assert x_transform == test
    assert all([[item >= 0 for item in arr] for arr in x_transform])
    assert all([[item <= 1 for item in arr] for arr in x_transform])


def test_oneHotEncode_should_error_on_list_with_single_category() -> None:
    x = [1, 1, 1, 1, 1, 1]

    with pytest.raises(AssertionError):
        oneHotEncode(x)
