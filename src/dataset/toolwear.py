__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision import transforms

from pathlib import Path
import pickle

import torch
import numpy as np

from torchvision import transforms

from dataset.generic import Standardizer, TimeSeriesDatasetWrapper

class ToolwearDataset:
    CSV_PATH = Path('../input/machining/tool_wear/tool4/raw/data.csv')
    PICKLE_FPATH = Path('../input/machining/tool_wear/tool4/labeled')

    def __init__(self, data, label, train, scaler, params, hyperparams):
        self.params = params
        self.hyperparams = hyperparams
        self.data = data
        self.label = label

        self.train = train
        self.scaler = scaler
        self.augment = params['augment'] if train else False

        self.preprocessing()  # augment data and labels + applies scaler

        self.seq_len = self.hyperparams['seq_len']
        self.batch_size = self.hyperparams['train_batch_size']
        self.input_dim = 1
        self.time_skip = self.data.size(0) // self.batch_size
        self.data = self.data.narrow(0, 0, self.time_skip * self.batch_size)
        self.batched_data = self.data.contiguous().view(self.batch_size, -1, self.input_dim).transpose(0, 1)

    @staticmethod
    def augmentation(data, label, std, noise_ratio=0.05, noise_interval=0.0005, max_length=100000):
        noiseSeq = torch.randn(data.size())
        augmentedData = data.clone()
        augmentedLabel = label.clone()
        for i in np.arange(0, noise_ratio, noise_interval):
            scaled_noiseSeq = noise_ratio * std.expand_as(data) * noiseSeq
            augmentedData = torch.cat([augmentedData, data + scaled_noiseSeq], dim=0)
            augmentedLabel = torch.cat([augmentedLabel, label])
            if len(augmentedData) > max_length:
                augmentedData = augmentedData[:max_length]
                augmentedLabel = augmentedLabel[:max_length]
                break

        return augmentedData, augmentedLabel

    def preprocessing(self):
        if self.train:
            # Train the scaler
            self.scaler.fit(self.data)
            # Augment the data
            if self.augment:
                self.data, self.label = self.augmentation(self.data, self.label, std=self.scaler.std)

        # Apply standardization or normalization
        self.data = self.scaler.transform(self.data)

    @classmethod
    def from_pickle(cls, train=True, **kwargs):
        path = cls.PICKLE_FPATH / ('train' if train else 'test') / 'toolwear.pkl'

        def load_data(path):
            with open(str(path), 'rb') as f:
                df = pickle.load(f)
                # label = torch.FloatTensor(df.loc[:, 'wear_len'].values)
                # data = torch.FloatTensor(df.loc[:, 'acc'].values.reshape(-1, 1))
                df.index = pd.to_datetime(df.index)
                label = torch.FloatTensor(df.loc[:, 'wear_len'].resample('1S').mean().values)
                data = torch.FloatTensor(df.loc[:, 'acc'].resample('1S').std().values.reshape((-1, 1)))
                # data = torch.FloatTensor(df.loc[:, 'acc'].resample('1S').std().diff(1).fillna(0).values.reshape((-1, 1)))
            return data, label

        data, label = load_data(path)
        return cls(data=data, label=label, train=train, **kwargs)

    @classmethod
    def from_file(cls, train=True, **kwargs):
        # 
        #         data = self.raw_data['acc'].values.astype(float)
        #         targets = self.raw_data['wear_len'].values.astype(float)
        raw_path = cls.CSV_PATH
        raw_data = pd.read_csv(raw_path, index_col=0)
        train_path = raw_path.parent.parent.joinpath('labeled', 'train', 'toolwear.pkl').with_suffix('.pkl')

        train_size = int(raw_data.shape[0]*0.8)
        train_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(train_path), 'wb') as pkl:
            pickle.dump(raw_data[:train_size], pkl)

        test_path = raw_path.parent.parent.joinpath('labeled', 'test', 'toolwear.pkl').with_suffix('.pkl')
        test_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(test_path), 'wb') as pkl:
            pickle.dump(raw_data[train_size:], pkl)

        return cls.from_pickle(train, **kwargs)





if __name__ == '__main__':
    import os
    os.chdir('..')
    scaler = Standardizer()


    # ToolwearDataset.from_file()
    train_dataset = ToolwearDataset.from_pickle(train=True, scaler=scaler)
    test_dataset = ToolwearDataset.from_pickle(train=False, scaler=scaler)

    dataset = TimeSeriesDatasetWrapper(trainset=train_dataset,
                                        testset=test_dataset)

