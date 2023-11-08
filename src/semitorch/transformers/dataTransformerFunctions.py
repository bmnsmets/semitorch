from numpy import exp, tanh


def rescaleNonNegativeToUnitInterval(list_to_transform: list, maximum: float = None) -> tuple:
    """
    Rescale non-negative input from an unknown range to [0, 1].

    :param list_to_transform: the input list to rescale
    :param maximum: the maximum to scale with, if None the maximum over the input list will be used
    :returns: transformed input x via transform(x) = tanh( x / maximum ), maximum
    """

    assert len(list(filter(lambda item: item < 0.0, list_to_transform))) == 0, \
        f"Cannot rescale non negative input if input is not non negative"

    if maximum is None:
        maximum = max(list_to_transform)
    assert maximum >= 0, f"Cannot rescale non negative input to unit interval with maximum {maximum}"

    return [tanh(item / maximum) for item in list_to_transform], maximum


def rescaleNonPositiveToUnitInterval(list_to_transform: list, minimum: float = None) -> tuple:
    """
    Rescale non-positive input from an unknown range to [0, 1].

    :param list_to_transform: the input list to rescale
    :param minimum: the minimum to scale with, if None the minimum over the input list will be used
    :returns: transformed input x via transform(x) = tanh( x / minimum ), minimum
    """

    assert len(list(filter(lambda item: item > 0.0, list_to_transform))) == 0, \
        f"Cannot rescale non positive input if input is not non positive"

    if minimum is None:
        minimum = min(list_to_transform)
    assert minimum <= 0, f"Cannot rescale non positive input to unit interval with minimum {minimum}"

    return [tanh(item / minimum) for item in list_to_transform], minimum


def rescaleClosedIntervalToUnitInterval(list_to_transform: list, maximum: float = None, minimum: float = None) -> tuple:
    """
    Rescale input from a known closed interval [a, b] to [0, 1].

    :param list_to_transform: the input list to rescale
    :param maximum: the maximum to scale with corresponding to the upper bound of the known closed interval (b),
    if None the maximum over the input list will be used
    :param minimum: the minimum to scale with corresponding to the lower bound of the known closed interval (a),
    if None the minimum over the input list will be used
    :returns: transformed input x via transform(x) = (x - minimum) / (maximum - minimum), maximum, minimum
    """

    list_max, list_min = max(list_to_transform), min(list_to_transform)
    if maximum is None:
        maximum = list_max
    if minimum is None:
        minimum = list_min
    assert maximum != minimum, f"Cannot rescale input to unit interval if maximum equals minimum ({maximum})"
    assert list_max <= maximum, f"Cannot rescale input larger than provided maximum {maximum}"
    assert list_min >= minimum, f"Cannot rescale input smaller than provided minimum {minimum}"

    return [(item - minimum) / (maximum - minimum) for item in list_to_transform], maximum, minimum


def rescaleRealsToUnitInterval(list_to_transform: list, maximum: float = None, minimum: float = None) -> tuple:
    """
    Rescale input from an unknown range over the reals to [0, 1].

    :param list_to_transform: the input list to rescale
    :param maximum: the maximum to scale with corresponding to the upper bound of the known closed interval (b),
    if None the maximum over the input list will be used
    :param minimum: the minimum to scale with corresponding to the lower bound of the known closed interval (a),
    if None the minimum over the input list will be used
    :returns: transformed input x via transform(x) = sigmoid( 2 * (x - minimum) / (maximum - minimum) - 1 ),
    maximum, minimum
    """

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + exp(-x))

    if maximum is None:
        maximum = max(list_to_transform)
    if minimum is None:
        minimum = min(list_to_transform)

    return [sigmoid(2 * (item - minimum) / (maximum - minimum) - 1) for item in list_to_transform], maximum, minimum


def oneHotEncode(list_to_transform: list, categories: list = None) -> tuple:
    """
    Apply one hot encoding to the input. Transformation assumes all categories are present at least once in the input.

    :param list_to_transform: the input list to rescale
    :param categories: the list of seen categories, if None this will be computed over the input list
    :returns: one hot encoded input of x, categories
    """

    if categories is None:
        categories = sorted(list(set(list_to_transform)))
    assert len(categories) > 1, f"Cannot one hot encode input with only 1 single category, remove column instead"

    onehot_encoded = []
    for category in list_to_transform:
        item = [0 for _ in range(len(categories))]
        item[categories.index(category)] = 1
        onehot_encoded.append(item)

    return onehot_encoded, categories
