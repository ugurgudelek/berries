import torch
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from torch.utils.data.dataset import TensorDataset
from berries.utils.transform import Normalizer

from berries.datasets.dataset import StatelessTimeseriesDataset
import numpy as np
from berries.utils.data import open_data

from torch import nn


class ECG5000Inner(nn.Module):
    def __init__(self, timeseries, targets):
        self.timeseries = torch.from_numpy(timeseries).float()
        self.targets = targets

    def __getitem__(self, ix):
        return {'data': self.timeseries[ix], 'target': self.timeseries[ix]}

    def __len__(self):
        return len(self.timeseries)

    def get_targets(self):
        return self.targets


class ECG5000:
    def __init__(self):
        X_train, X_val, y_train, y_val = open_data('./input', ratio_train=0.9)

        num_classes = len(np.unique(y_train))
        base = np.min(y_train)  # Check if data is 0-based
        if base != 0:
            y_train -= base
        y_val -= base

        self.trainset = ECG5000Inner(timeseries=X_train, targets=y_train)
        self.testset = ECG5000Inner(timeseries=X_val, targets=y_val)


if __name__ == "__main__":
    dataset = ECG5000()
