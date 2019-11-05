from pathlib import Path
import pickle

import torch
import numpy as np

from torchvision import transforms







class NYCTaxiDataset:
    CSV_PATH = Path('../input/nyc_taxi/raw/nyc_taxi.csv')
    PICKLE_FPATH = Path('../input/nyc_taxi/labeled')

    def __init__(self, data, label, train, scaler, augment=False):
        self.data = data
        self.label = label

        self.train = train
        self.scaler = scaler
        self.augment = augment

        self.preprocessing()  # augment data and labels + applies scaler

        # self.data = torch.FloatTensor(np.array(list(range(13104))).reshape(-1, 1))
        # self.data = self.batchify(self.data, 64)

        # data = list(range(13104))
        # data = np.array([data, data, data]).T
        # self.data = torch.from_numpy(data).float()

        self.seq_len = 50
        self.batch_size = 64
        self.time_skip = self.data.size(0) // self.batch_size
        self.data = self.data.narrow(0, 0, self.time_skip * self.batch_size)
        self.batched_data = self.data.contiguous().view(self.batch_size, -1, 3).transpose(0,1)


        # w_tensor = None
        # for i in range(0, len(self.batched_data), self.seq_len):
        #     w = self.batched_data[i:i+self.seq_len]
        #     if w_tensor is None:
        #         w_tensor = w
        #     else:
        #         w_tensor = torch.cat((w_tensor, w), 1)
        #
        # self.batched_data = w_tensor
        #
        #
        # for i in range(0, self.time_skip//self.seq_len+2, 1):
        #     for j in range(0, 13056, self.time_skip):
        #         yield i+j

        #
        #
        # inputSeq0, targetSeq0 = self[:64]
        # inputSeq1, targetSeq1 = self.__getitem__(1)



        print()

    # def __getitem__(self, ix):
    #     return (self.data[ix:ix+self.seq_len],
    #             self.data[ix+1:ix+1+self.seq_len])
    #
    # def __len__(self):
    #     return len(self.data) - self.seq_len

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
        path = cls.PICKLE_FPATH / ('train' if train else 'test') / 'nyc_taxi.pkl'

        def load_data(path):
            with open(str(path), 'rb') as f:
                loaded = torch.FloatTensor(pickle.load(f))
                label = loaded[:, -1]
                data = loaded[:, :-1]
            return data, label

        data, label = load_data(path)
        return cls(data=data, label=label, train=train, **kwargs)

    @classmethod
    def from_file(cls):
        nyc_taxi_raw_path = cls.CSV_PATH
        labeled_data = []
        with open(str(nyc_taxi_raw_path), 'r') as f:
            for i, line in enumerate(f):
                tokens = [float(token) for token in line.strip().split(',')[1:]]
                tokens.append(1) if 150 < i < 250 or \
                                    5970 < i < 6050 or \
                                    8500 < i < 8650 or \
                                    8750 < i < 8890 or \
                                    10000 < i < 10200 or \
                                    14700 < i < 14800 \
                    else tokens.append(0)
                labeled_data.append(tokens)
        nyc_taxi_train_path = nyc_taxi_raw_path.parent.parent.joinpath('labeled', 'train',
                                                                       nyc_taxi_raw_path.name).with_suffix('.pkl')
        nyc_taxi_train_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(nyc_taxi_train_path), 'wb') as pkl:
            pickle.dump(labeled_data[:13104], pkl)

        nyc_taxi_test_path = nyc_taxi_raw_path.parent.parent.joinpath('labeled', 'test',
                                                                      nyc_taxi_raw_path.name).with_suffix('.pkl')
        nyc_taxi_test_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(nyc_taxi_test_path), 'wb') as pkl:
            pickle.dump(labeled_data[13104:], pkl)

        return cls(fpath=(nyc_taxi_raw_path / '../../labeled').resolve())


class NYCTaxiDatasetWrapper():
    def __init__(self, trainset, testset):
        self.trainset = trainset
        self.testset = testset

        self.feature_dim = self.trainset.data.size(1)


if __name__ == '__main__':
    scaler = Standardizer()

    train_dataset = NYCTaxiDataset.from_pickle(train=True, scaler=scaler)
    test_dataset = NYCTaxiDataset.from_pickle(train=False, scaler=scaler)

    dataset = NYCTaxiDatasetWrapper(trainset=train_dataset,
                                    testset=test_dataset)
