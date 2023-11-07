from torch.utils.data import Dataset


class DatasetFromSubset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, index):
        x, y = self.subset[index]

        return x, y

    def __len__(self):
        return len(self.subset)
