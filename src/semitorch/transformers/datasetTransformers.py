from .datasetFromSubset import DatasetFromSubset
from .dataTransformerFunctions import (
    rescaleNonNegativeToUnitInterval,
    rescaleNonPositiveToUnitInterval,
    rescaleClosedIntervalToUnitInterval,
    rescaleRealsToUnitInterval,
    oneHotEncode,
)
from numpy import array, reshape
from torch import Tensor


class BaseDatasetTransformer:
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

        onehotencode_offset = 0
        for i in range(num_input_features):
            i_index = i + onehotencode_offset
            x_train_data = [x_train[j][i_index] for j in range(len(x_train))]
            x_test_data = [x_test[j][i_index] for j in range(len(x_test))]

            x_train_data, x_test_data, onehotencode_offset = (
                self.doTransform(i, x_train_data, x_test_data, onehotencode_offset))

            for j in range(len(x_train)):
                if isinstance(x_train_data[j], list):
                    x_train[j][i_index:i_index + 1] = x_train_data[j]
                else:
                    x_train[j][i_index] = x_train_data[j]
            for j in range(len(x_test)):
                if isinstance(x_train_data[j], list):
                    x_test[j][i_index:i_index + 1] = x_test_data[j]
                else:
                    x_test[j][i_index] = x_test_data[j]

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

    def doTransform(self, i: int, x_train_data: list, x_test_data: list, onehotencode_offset: int) -> tuple:
        pass


class IrisDatasetTransformer(BaseDatasetTransformer):
    def doTransform(self, i: int, x_train_data: list, x_test_data: list, onehotencode_offset: int):
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

        return x_train_data, x_test_data, onehotencode_offset


class HeartDiseaseDatasetTransformer(BaseDatasetTransformer):
    def doTransform(self, i: int, x_train_data: list, x_test_data: list, onehotencode_offset: int):
        if i == 0:
            # First input feature is non-negative -> rescaleNonNegativeToUnitInterval
            x_train_data[:], maximum = rescaleNonNegativeToUnitInterval(x_train_data)
            x_test_data[:], _ = rescaleNonNegativeToUnitInterval(x_test_data, maximum=maximum)
        elif i == 1:
            # Second input feature is categorical -> oneHotEncode
            x_train_data[:], categories = oneHotEncode(x_train_data)
            x_test_data[:], _ = oneHotEncode(x_test_data, categories=categories)
            onehotencode_offset += len(categories) - 1
        elif i == 2:
            # Third input feature is categorical -> oneHotEncode
            x_train_data[:], categories = oneHotEncode(x_train_data)
            x_test_data[:], _ = oneHotEncode(x_test_data, categories=categories)
            onehotencode_offset += len(categories) - 1
        elif i == 3:
            # Fourth input feature is non-negative -> rescaleNonNegativeToUnitInterval
            x_train_data[:], maximum = rescaleNonNegativeToUnitInterval(x_train_data)
            x_test_data[:], _ = rescaleNonNegativeToUnitInterval(x_test_data, maximum=maximum)
        elif i == 4:
            # Fifth input feature is non-negative -> rescaleNonNegativeToUnitInterval
            x_train_data[:], maximum = rescaleNonNegativeToUnitInterval(x_train_data)
            x_test_data[:], _ = rescaleNonNegativeToUnitInterval(x_test_data, maximum=maximum)
        elif i == 5:
            # Sixth input feature is categorical -> oneHotEncode
            x_train_data[:], categories = oneHotEncode(x_train_data)
            x_test_data[:], _ = oneHotEncode(x_test_data, categories=categories)
            onehotencode_offset += len(categories) - 1
        elif i == 6:
            # Seventh input feature is categorical -> oneHotEncode
            x_train_data[:], categories = oneHotEncode(x_train_data)
            x_test_data[:], _ = oneHotEncode(x_test_data, categories=categories)
            onehotencode_offset += len(categories) - 1
        elif i == 7:
            # Eight input feature is non-negative -> rescaleNonNegativeToUnitInterval
            x_train_data[:], maximum = rescaleNonNegativeToUnitInterval(x_train_data)
            x_test_data[:], _ = rescaleNonNegativeToUnitInterval(x_test_data, maximum=maximum)
        elif i == 8:
            # Ninth input feature is categorical -> oneHotEncode
            x_train_data[:], categories = oneHotEncode(x_train_data)
            x_test_data[:], _ = oneHotEncode(x_test_data, categories=categories)
            onehotencode_offset += len(categories) - 1
        elif i == 9:
            # Tenth input feature is non-negative -> rescaleNonNegativeToUnitInterval
            x_train_data[:], maximum = rescaleNonNegativeToUnitInterval(x_train_data)
            x_test_data[:], _ = rescaleNonNegativeToUnitInterval(x_test_data, maximum=maximum)
        elif i == 10:
            # Eleventh input feature is categorical -> oneHotEncode
            x_train_data[:], categories = oneHotEncode(x_train_data)
            x_test_data[:], _ = oneHotEncode(x_test_data, categories=categories)
            onehotencode_offset += len(categories) - 1
        elif i == 11:
            # Twelfth input feature is categorical -> oneHotEncode
            x_train_data[:], categories = oneHotEncode(x_train_data)
            x_test_data[:], _ = oneHotEncode(x_test_data, categories=categories)
            onehotencode_offset += len(categories) - 1
        elif i == 12:
            # Thirteenth input feature is categorical -> oneHotEncode
            x_train_data[:], categories = oneHotEncode(x_train_data)
            x_test_data[:], _ = oneHotEncode(x_test_data, categories=categories)
            onehotencode_offset += len(categories) - 1
        else:
            raise ValueError("No more than thirteen input features expected")

        return x_train_data, x_test_data, onehotencode_offset


class CirclesDatasetTransformer(BaseDatasetTransformer):
    def doTransform(self, i: int, x_train_data: list, x_test_data: list, onehotencode_offset: int):
        if i == 0:
            # First input feature is from unknown interval over the reals -> rescaleRealsToUnitInterval
            x_train_data[:], maximum, minimum = rescaleRealsToUnitInterval(x_train_data)
            x_test_data[:], _, _ = rescaleRealsToUnitInterval(x_test_data, maximum=maximum, minimum=minimum)
        elif i == 1:
            # Second input feature is from unknown interval over the reals -> rescaleRealsToUnitInterval
            x_train_data[:], maximum, minimum = rescaleRealsToUnitInterval(x_train_data)
            x_test_data[:], _, _ = rescaleRealsToUnitInterval(x_test_data, maximum=maximum, minimum=minimum)
        else:
            raise ValueError("No more than two input features expected")

        return x_train_data, x_test_data, onehotencode_offset


class RingsDatasetTransformer(BaseDatasetTransformer):
    def doTransform(self, i: int, x_train_data: list, x_test_data: list, onehotencode_offset: int):
        if i == 0:
            # First input feature is from unknown interval over the reals -> rescaleRealsToUnitInterval
            x_train_data[:], maximum, minimum = rescaleRealsToUnitInterval(x_train_data)
            x_test_data[:], _, _ = rescaleRealsToUnitInterval(x_test_data, maximum=maximum, minimum=minimum)
        elif i == 1:
            # Second input feature is from unknown interval over the reals -> rescaleRealsToUnitInterval
            x_train_data[:], maximum, minimum = rescaleRealsToUnitInterval(x_train_data)
            x_test_data[:], _, _ = rescaleRealsToUnitInterval(x_test_data, maximum=maximum, minimum=minimum)
        elif i == 2:
            # Third input feature is from unknown interval over the reals -> rescaleRealsToUnitInterval
            x_train_data[:], maximum, minimum = rescaleRealsToUnitInterval(x_train_data)
            x_test_data[:], _, _ = rescaleRealsToUnitInterval(x_test_data, maximum=maximum, minimum=minimum)
        else:
            raise ValueError("No more than three input features expected")

        return x_train_data, x_test_data, onehotencode_offset


class SpheresDatasetTransformer(BaseDatasetTransformer):
    def doTransform(self, i: int, x_train_data: list, x_test_data: list, onehotencode_offset: int):
        if i == 0:
            # First input feature is from unknown interval over the reals -> rescaleRealsToUnitInterval
            x_train_data[:], maximum, minimum = rescaleRealsToUnitInterval(x_train_data)
            x_test_data[:], _, _ = rescaleRealsToUnitInterval(x_test_data, maximum=maximum, minimum=minimum)
        elif i == 1:
            # Second input feature is from unknown interval over the reals -> rescaleRealsToUnitInterval
            x_train_data[:], maximum, minimum = rescaleRealsToUnitInterval(x_train_data)
            x_test_data[:], _, _ = rescaleRealsToUnitInterval(x_test_data, maximum=maximum, minimum=minimum)
        elif i == 2:
            # Third input feature is from unknown interval over the reals -> rescaleRealsToUnitInterval
            x_train_data[:], maximum, minimum = rescaleRealsToUnitInterval(x_train_data)
            x_test_data[:], _, _ = rescaleRealsToUnitInterval(x_test_data, maximum=maximum, minimum=minimum)
        else:
            raise ValueError("No more than three input features expected")

        return x_train_data, x_test_data, onehotencode_offset


class FashionMNISTDatasetTransformer:
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

        for i, (x_train_item, x_test_item) in enumerate(zip(x_train, x_test)):
            x_train_item, x_test_item = self.doTransform(x_train_item, x_test_item)
            x_train[i], x_test[i] = x_train_item, x_test_item

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

    def doTransform(self, x_train_data: list, x_test_data: list):
        # Flatten list from 28x28 to 784x1
        x_train_data = array(x_train_data).flatten().tolist()
        x_test_data = array(x_test_data).flatten().tolist()

        # All input features are from closed interval [0, 255] -> rescaleClosedIntervalToUnitInterval
        x_train_data[:], _, _ = rescaleClosedIntervalToUnitInterval(x_train_data, maximum=255, minimum=0)
        x_test_data[:], _, _ = rescaleClosedIntervalToUnitInterval(x_test_data, maximum=255, minimum=0)

        # Reshape list from 784x1 to 28x28
        x_train_data = reshape(array(x_train_data), (1, 28, 28)).tolist()
        x_test_data = reshape(array(x_test_data), (1, 28, 28)).tolist()

        return x_train_data, x_test_data
