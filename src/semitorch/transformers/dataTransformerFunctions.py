from numpy import tanh


def rescaleNonNegativeToUnitInterval(list_to_transform: list, maximum: float = None) -> tuple:
    assert len(list(filter(lambda item: item < 0.0, list_to_transform))) == 0, \
        f"Cannot rescale non negative input if input is not non negative"

    if maximum is None:
        maximum = max(list_to_transform)
    assert maximum >= 0, f"Cannot rescale non negative input to unit interval with maximum {maximum}"

    return [tanh(item / maximum) for item in list_to_transform], maximum
