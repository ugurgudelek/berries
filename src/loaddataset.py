import pandas as pd
import torch
import numpy as np


class LoadFullDataset():
    def __init__(self, csv_path, train_valid_ratio=0.9, train_day=None, seq_length=96) -> None:
        self.dataset_values = pd.read_csv(csv_path).loc[:, 'actual'].values

        # 1 Jan	Mon	New Year's Day	National
        # 30 Mar	Fri	Good Friday	National
        # 2 Apr	Mon	Easter Monday	National
        # 1 May	Tue	Labour Day	National
        # 10 May	Thu	Ascension Day	National
        # 21 May	Mon	Whit Monday	National
        # 3 Oct	Wed	Day of German Unity	National
        # 25 Dec	Tue	Christmas Day	National
        # 26 Dec	Wed	2nd Day of Christmas	National
        #
        # 6 Jan	Sat	Epiphany		BW, BY & ST
        # 1 Apr	Sun	Easter Sunday	BB
        # 20 May	Sun	Whit Sunday	BB
        # 31 May	Thu	Corpus Christi	BW, BY, HE, NW, RP, SL,SN & TH
        # 15 Aug	Wed	Assumption Day	BY & SL
        # 31 Oct	Wed	Reformation Day	BB, MV, SN, ST & TH
        # 1 Nov	Thu	All Saints' Day	BW, BY, NW, RP & SL
        # 21 Nov	Wed	Repentance Day	SN

        dataset_len = self.dataset_values.shape[0]

        # === CREATE PERIODIC SIGNALS
        daycount = self.dataset_values.shape[0] // seq_length
        self.dataset_values = self.dataset_values[:daycount * seq_length]  # remove uncomplete days

        def create_period_signal(freq, Fs):
            t = np.arange(Fs)
            return np.sin(2 * np.pi * freq * t / Fs)

        p_day = create_period_signal(daycount * seq_length / 96, daycount * seq_length)
        p_week = create_period_signal(daycount * seq_length / (96 * 7), daycount * seq_length)
        p_month = create_period_signal(daycount * seq_length / (96 * 30), daycount * seq_length)
        p_year = create_period_signal(daycount * seq_length / (96 * 365), daycount * seq_length)

        self.dataset_values = np.stack((self.dataset_values, p_day, p_week, p_month, p_year), axis=1)

        # TODO: fix reshape to estimate quarters. seq_length should be added in forwward pass
        # self.dataset_values = np.reshape(self.dataset_values, (-1, seq_length, 5))

        # SPLIT TRAIN & VALID
        if train_day is None:
            train_day = int(daycount * train_valid_ratio)
        valid_day = daycount - train_day

        train_len = train_day * seq_length
        valid_len = valid_day * seq_length

        # train_values = self.dataset_values[:train_len, :, :]
        # valid_values = self.dataset_values[train_len:, :, :]

        train_values = self.dataset_values[:train_len, :]
        valid_values = self.dataset_values[train_len:, :]

        self.train_dataset = LoadDataset(train_values, seq_length=seq_length)
        self.valid_dataset = LoadDataset(valid_values, seq_length=seq_length)


class LoadDataset(torch.utils.data.Dataset):
    """

        Args:
            seq_length:
        Attributes:
            dataset:
            X:
            y:
    """

    def __init__(self, dataset, seq_length, shuffle=True):
        # normalize data. otherwise criterion cannot calculate loss
        self.dataset, self.min_norm_term, self.max_norm_term = self.normalize(dataset)
        # split data wrt period
        # e.g. period = 96 -> (day_size, quarter_in_day)


        self.seq_length = seq_length

        # TODO: add shuffling later.
        # if shuffle:
        #     np.random.shuffle(self.dataset)

        # rearrange X and targets
        # X = (d1,d2,d3...dn-1)
        # y = (d2,d3,d4...dn)

        # self.y = self.dataset[1:, :, 0]
        # self.X = self.dataset[:-1, :, :]

        self.y = self.dataset[1:, 0]
        self.X = self.dataset[:-1, :]

    def normalize(self, arr):
        """

        Args:
            arr:

        Returns:

        """
        return (arr - arr.min()) / (arr.max() - arr.min()), arr.min(), arr.max()

    def inverse_normalize(self, arr, min_term, max_term):
        """

        Args:
            arr:
            min_term:
            max_term:

        Returns:

        """
        return arr*(max_term-min_term) + min_term

    def __len__(self):
        """

        Returns:
            int: data count

        """
        return self.X.shape[0] - self.seq_length

    def __getitem__(self, ix):
        """

        Args:
            ix:

        Returns:
            (np.ndarray, np.ndarray):

        """
        # (row, seq_len, input_size)
        # return self.X[ix, :, :], self.y[ix, :]
        return self.X[ix:ix + self.seq_length, :], self.y[ix + self.seq_length - 1]