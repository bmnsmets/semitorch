from .datasetFromSubset import DatasetFromSubset
from .dataTransformerFunctions import (
    rescaleNonNegativeToUnitInterval,
    rescaleNonPositiveToUnitInterval,
    rescaleClosedIntervalToUnitInterval,
    rescaleRealsToUnitInterval,
    oneHotEncode,
)
from torch import Tensor


class IrisDatasetTransformer:
    """
    Transforms the train and test data of the iris dataset to the unit interval [0, 1].

    :param x_train: the iris train dataset to be rescaled
    :param x_test: the iris test dataset to be rescaled (uses the maximum of the train data for rescaling)
    :returns: the transformed train and test data x via transform(x) = tanh( x / x_max(x_train) ).
    """

    def __init__(self, x_train: DatasetFromSubset, x_test: DatasetFromSubset, device: str):
        self.x_train = x_train
        self.x_test = x_test
        self.device = device

    def transform(self) -> tuple:
        x_train, x_test = [], []
        for i in range(len(self.x_train)):
            x, y = self.x_train[i]
            x_train.append(x.tolist())
        for i in range(len(self.x_test)):
            x, y = self.x_test[i]
            x_test.append(x.tolist())
        num_input_features = len(x_train[0])

        for i in range(num_input_features):
            x_train_data = [x_train[j][i] for j in range(len(x_train))]
            x_test_data = [x_test[j][i] for j in range(len(x_test))]

            if i == 0:
                # First input feature is non-negative -> rescaleNonNegativeToUnitInterval
                x_train_data[:], maximum = rescaleNonNegativeToUnitInterval(x_train_data)
                x_test_data[:], _ = rescaleNonNegativeToUnitInterval(x_test_data, maximum=maximum)
            elif i == 1:
                # Second input feature is non-negative -> rescaleNonNegativeToUnitInterval
                x_train_data[:], maximum = rescaleNonNegativeToUnitInterval(x_train_data)
                x_test_data[:], _ = rescaleNonNegativeToUnitInterval(x_test_data, maximum=maximum)
            elif i == 2:
                # Third input feature is non-negative -> rescaleNonNegativeToUnitInterval
                x_train_data[:], maximum = rescaleNonNegativeToUnitInterval(x_train_data)
                x_test_data[:], _ = rescaleNonNegativeToUnitInterval(x_test_data, maximum=maximum)
            elif i == 3:
                # Fourth input feature is non-negative -> rescaleNonNegativeToUnitInterval
                x_train_data[:], maximum = rescaleNonNegativeToUnitInterval(x_train_data)
                x_test_data[:], _ = rescaleNonNegativeToUnitInterval(x_test_data, maximum=maximum)
            else:
                raise ValueError("No more than four input features expected")

            for j in range(len(x_train)):
                x_train[j][i] = x_train_data[j]
            for j in range(len(x_test)):
                x_test[j][i] = x_test_data[j]

        output_train, output_test = [], []
        for i in range(len(self.x_train)):
            _, y = self.x_train[i]
            x = Tensor(x_train[i]).float().to(self.device)
            output_train.append((x, y))
        for i in range(len(self.x_test)):
            _, y = self.x_test[i]
            x = Tensor(x_train[i]).float().to(self.device)
            output_test.append((x, y))

        return output_train, output_test
