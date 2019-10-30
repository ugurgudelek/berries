__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision import transforms


class ToolWearDataset():
    PATH = "../input/machining/tool_wear/tool2/data_sample.csv"

    def __init__(self):
        self.path = self.PATH
        self.transform = transforms.Compose([transforms.Lambda(lambda x: torch.from_numpy(x)),
                                             transforms.Lambda(lambda x: x.float())])

        self.raw_data = pd.read_csv(self.path, index_col=0)

        data = self.raw_data['acc'].values.astype(float)
        targets = self.raw_data['wear_len'].values.astype(float)

        train_size = int(len(data) * 0.8)
        self.trainset = self.InnerDataset(data[:train_size], targets[:train_size], self.transform)
        self.testset = self.InnerDataset(data[train_size:], targets[train_size:], self.transform)




    class InnerDataset(Dataset):
        def __init__(self, data, targets, transform):
            self.transform = transform
            self.data = self.transform(data)
            self.targets = self.transform(targets)
            self.seq_len = 500

        def __len__(self):
            return len(self.targets) - self.seq_len

        def __getitem__(self, ix):
            return self.data[ix:ix + self.seq_len, np.newaxis], self.targets[ix:ix + self.seq_len]
