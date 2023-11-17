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
    def __init__(self, data_to_transform: list, device: str):
        self.data_to_transform = data_to_transform
        self.device = device

    def transform(self) -> Tensor:
        data_to_transform = []
        for i in range(len(self.data_to_transform)):
            x = self.data_to_transform[i]
            data_to_transform.append(x.tolist())
        num_input_features = len(data_to_transform[0])

        onehotencode_offset = 0
        for i in range(num_input_features):
            i_index = i + onehotencode_offset
            data = [data_to_transform[j][i_index] for j in range(len(data_to_transform))]

            data, onehotencode_offset = self.doTransform(i, data, onehotencode_offset)

            for j in range(len(data_to_transform)):
                if isinstance(data[j], list):
                    data_to_transform[j][i_index:i_index + 1] = data[j]
                else:
                    data_to_transform[j][i_index] = data[j]

        output = []
        for i in range(len(self.data_to_transform)):
            output.append(data_to_transform[i])

        return Tensor(output).float().to(self.device)

    def doTransform(self, i: int, data: list, onehotencode_offset: int) -> tuple:
        pass


class IrisDatasetTransformer(BaseDatasetTransformer):
    def doTransform(self, i: int, data: list, onehotencode_offset: int):
        if i == 0:
            # First input feature is non-negative -> rescaleNonNegativeToUnitInterval
            data[:], _ = rescaleNonNegativeToUnitInterval(data)
        elif i == 1:
            # Second input feature is non-negative -> rescaleNonNegativeToUnitInterval
            data[:], _ = rescaleNonNegativeToUnitInterval(data)
        elif i == 2:
            # Third input feature is non-negative -> rescaleNonNegativeToUnitInterval
            data[:], _ = rescaleNonNegativeToUnitInterval(data)
        elif i == 3:
            # Fourth input feature is non-negative -> rescaleNonNegativeToUnitInterval
            data[:], _ = rescaleNonNegativeToUnitInterval(data)
        else:
            raise ValueError("No more than four input features expected")

        return data, onehotencode_offset


class HeartDiseaseDatasetTransformer(BaseDatasetTransformer):
    def doTransform(self, i: int, data: list, onehotencode_offset: int):
        if i == 0:
            # First input feature is non-negative -> rescaleNonNegativeToUnitInterval
            data[:], _ = rescaleNonNegativeToUnitInterval(data)
        elif i == 1:
            # Second input feature is categorical -> oneHotEncode
            data[:], categories = oneHotEncode(data)
            onehotencode_offset += len(categories) - 1
        elif i == 2:
            # Third input feature is categorical -> oneHotEncode
            data[:], categories = oneHotEncode(data)
            onehotencode_offset += len(categories) - 1
        elif i == 3:
            # Fourth input feature is non-negative -> rescaleNonNegativeToUnitInterval
            data[:], _ = rescaleNonNegativeToUnitInterval(data)
        elif i == 4:
            # Fifth input feature is non-negative -> rescaleNonNegativeToUnitInterval
            data[:], _ = rescaleNonNegativeToUnitInterval(data)
        elif i == 5:
            # Sixth input feature is categorical -> oneHotEncode
            data[:], categories = oneHotEncode(data)
            onehotencode_offset += len(categories) - 1
        elif i == 6:
            # Seventh input feature is categorical -> oneHotEncode
            data[:], categories = oneHotEncode(data)
            onehotencode_offset += len(categories) - 1
        elif i == 7:
            # Eight input feature is non-negative -> rescaleNonNegativeToUnitInterval
            data[:], _ = rescaleNonNegativeToUnitInterval(data)
        elif i == 8:
            # Ninth input feature is categorical -> oneHotEncode
            data[:], categories = oneHotEncode(data)
            onehotencode_offset += len(categories) - 1
        elif i == 9:
            # Tenth input feature is non-negative -> rescaleNonNegativeToUnitInterval
            data[:], _ = rescaleNonNegativeToUnitInterval(data)
        elif i == 10:
            # Eleventh input feature is categorical -> oneHotEncode
            data[:], categories = oneHotEncode(data)
            onehotencode_offset += len(categories) - 1
        elif i == 11:
            # Twelfth input feature is categorical -> oneHotEncode
            data[:], categories = oneHotEncode(data)
            onehotencode_offset += len(categories) - 1
        elif i == 12:
            # Thirteenth input feature is categorical -> oneHotEncode
            data[:], categories = oneHotEncode(data)
            onehotencode_offset += len(categories) - 1
        else:
            raise ValueError("No more than thirteen input features expected")

        return data, onehotencode_offset


class CirclesDatasetTransformer(BaseDatasetTransformer):
    def doTransform(self, i: int, data: list, onehotencode_offset: int):
        if i == 0:
            # First input feature is from unknown interval over the reals -> rescaleRealsToUnitInterval
            data[:], _, _ = rescaleRealsToUnitInterval(data)
        elif i == 1:
            # Second input feature is from unknown interval over the reals -> rescaleRealsToUnitInterval
            data[:], _, _ = rescaleRealsToUnitInterval(data)
        else:
            raise ValueError("No more than two input features expected")

        return data, onehotencode_offset


class RingsDatasetTransformer(BaseDatasetTransformer):
    def doTransform(self, i: int, data: list, onehotencode_offset: int):
        if i == 0:
            # First input feature is from unknown interval over the reals -> rescaleRealsToUnitInterval
            data[:], _, _ = rescaleRealsToUnitInterval(data)
        elif i == 1:
            # Second input feature is from unknown interval over the reals -> rescaleRealsToUnitInterval
            data[:], _, _ = rescaleRealsToUnitInterval(data)
        elif i == 2:
            # Third input feature is from unknown interval over the reals -> rescaleRealsToUnitInterval
            data[:], _, _ = rescaleRealsToUnitInterval(data)
        else:
            raise ValueError("No more than three input features expected")

        return data, onehotencode_offset


class SpheresDatasetTransformer(BaseDatasetTransformer):
    def doTransform(self, i: int, data: list, onehotencode_offset: int):
        if i == 0:
            # First input feature is from unknown interval over the reals -> rescaleRealsToUnitInterval
            data[:], _, _ = rescaleRealsToUnitInterval(data)
        elif i == 1:
            # Second input feature is from unknown interval over the reals -> rescaleRealsToUnitInterval
            data[:], _, _ = rescaleRealsToUnitInterval(data)
        elif i == 2:
            # Third input feature is from unknown interval over the reals -> rescaleRealsToUnitInterval
            data[:], _, _ = rescaleRealsToUnitInterval(data)
        else:
            raise ValueError("No more than three input features expected")

        return data, onehotencode_offset
