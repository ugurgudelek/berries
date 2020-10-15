import torch
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from torch.utils.data.dataset import TensorDataset
from berries.utils.transform import Normalizer

from berries.datasets.dataset import StatelessTimeseriesDataset
import numpy as np
from berries.utils.data import open_data

from torch import nn


class ECG5000Inner(Dataset):
    def __init__(self, timeseries, targets):
        self.timeseries = torch.from_numpy(timeseries).float()
        self.targets = targets

    def __getitem__(self, ix):
        raise NotImplementedError()

    def __len__(self):
        return len(self.timeseries)

    def get_targets(self):
        return self.targets


class ECG5000InnerforVAE(ECG5000Inner):
    def __init__(self, timeseries, targets):
        super().__init__(timeseries, targets)

    def __getitem__(self, ix):
        return {
            'data': torch.as_tensor(self.timeseries[ix], dtype=torch.float32),
            'target': torch.as_tensor(self.timeseries[ix], dtype=torch.float32)
        }


class ECG5000InnerforClassification(ECG5000Inner):
    def __init__(self, timeseries, targets):
        super().__init__(timeseries, targets)

    def __getitem__(self, ix):
        return {
            'data': torch.as_tensor(self.timeseries[ix], dtype=torch.float32),
            'target': torch.as_tensor(self.targets[ix], dtype=torch.float32)
        }


class ECG5000:
    def __init__(self, forVAE=True):
        self.forVAE = forVAE

        X_train, X_val, y_train, y_val = open_data('./input', ratio_train=0.9)

        num_classes = len(np.unique(y_train))
        base = np.min(y_train)  # Check if data is 0-based
        if base != 0:
            y_train -= base
        y_val -= base

        if self.forVAE:
            self.trainset = ECG5000InnerforVAE(timeseries=X_train,
                                               targets=y_train)
            self.testset = ECG5000InnerforVAE(timeseries=X_val, targets=y_val)
        else:
            self.trainset = ECG5000InnerforClassification(timeseries=X_train,
                                                          targets=y_train)
            self.testset = ECG5000InnerforClassification(timeseries=X_val,
                                                         targets=y_val)


if __name__ == "__main__":
    dataset = ECG5000()
