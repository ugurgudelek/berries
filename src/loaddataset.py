import pandas as pd
import torch
import numpy as np

import warnings
class LoadFullDataset():
    def __init__(self, csv_path, train_valid_ratio=0.9, train_day=None, valid_day=None, seq_length=96) -> None:

        # date,from,to,actual,forecast
        self.raw_dataset = pd.read_csv(csv_path)
        self.raw_dataset['date'] = self.raw_dataset['date'].astype('datetime64[ns]')
        self.raw_dataset['from'] = self.raw_dataset['from'].astype('datetime64[ns]')

        # parse date
        # self.raw_dataset['date'] = self.raw_dataset.apply(
        #     lambda row: pd.to_datetime(row['date'], format='%Y-%m-%d %H:%M:%S'), axis=1)

        # (years, months, weeks, days) = zip(*self.raw_dataset['date'].apply(lambda x: (x.year, x.month, x.week, x.day)))

        years = self.raw_dataset['date'].dt.year.values
        months = self.raw_dataset['date'].dt.month.values
        weeks = self.raw_dataset['date'].dt.weekofyear.values
        days = self.raw_dataset['date'].dt.day.values
        hours = self.raw_dataset['from'].dt.hour.values

        self.dataset = self.raw_dataset.loc[:, 'actual'].values

        self.dataset = np.stack((self.dataset, hours, days, weeks, months, years), axis=1)

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

        dataset_len = self.dataset.shape[0]


        daycount = self.dataset.shape[0] // seq_length
        self.dataset = self.dataset[:daycount * seq_length]  # remove uncomplete days
        self.raw_dataset = self.raw_dataset[:daycount * seq_length]


        # normalize
        self.dataset, self.min_norm_term, self.max_norm_term = self.normalize(self.dataset)

        # def create_period_signal(freq, Fs):
        #     t = np.arange(Fs)
        #     return np.sin(2 * np.pi * freq * t / Fs)
        #
        # p_day = create_period_signal(daycount * seq_length / 96, daycount * seq_length)
        # p_week = create_period_signal(daycount * seq_length / (96 * 7), daycount * seq_length)
        # p_month = create_period_signal(daycount * seq_length / (96 * 30), daycount * seq_length)
        # p_year = create_period_signal(daycount * seq_length / (96 * 365), daycount * seq_length)
        #
        # self.dataset = np.stack((self.dataset, p_day, p_week, p_month, p_year), axis=1)

        # TODO: fix reshape to estimate quarters. seq_length should be added in forward pass
        # self.dataset_values = np.reshape(self.dataset_values, (-1, seq_length, 5))

        # SPLIT TRAIN & VALID
        if train_day is None:
            train_day = int(daycount * train_valid_ratio)

        if valid_day is None:
            valid_day = daycount - train_day

        self.train_len = train_len = train_day * seq_length
        self.valid_len = valid_len = valid_day * seq_length

        # train_values = self.dataset_values[:train_len, :, :]
        # valid_values = self.dataset_values[train_len:, :, :]

        missing_day_amount = ((train_len + valid_len) - self.dataset.shape[0]) / seq_length
        if missing_day_amount != 0:
            warnings.warn('{} day data is missing for validation'.format(missing_day_amount), UserWarning)


        train_values = self.dataset[:train_len, :]
        valid_values = self.dataset[train_len:train_len+valid_len, :]

        raw_train_dataset = self.raw_dataset.iloc[:train_len, :]
        raw_valid_dataset = self.raw_dataset.iloc[train_len:train_len+valid_len, :]

        self.train_dataset = LoadDataset(train_values, seq_length=seq_length, raw_dataset=raw_train_dataset)
        self.valid_dataset = LoadDataset(valid_values, seq_length=seq_length, raw_dataset=raw_valid_dataset)

    def get_raw_valid_dataset(self):
        return self.raw_dataset[self.train_len:self.train_len+self.valid_len]

    def normalize(self, arr):
        """

        Args:
            arr:

        Returns:

        """
        return (arr - arr.min(axis=0)) / (arr.max(axis=0) - arr.min(axis=0)), arr.min(axis=0), arr.max(axis=0)

    def inverse_normalize(self, arr, min_term=None, max_term=None, only_first=False):
        """

        Args:
            arr:
            min_term:
            max_term:

        Returns:

        """


        if min_term is None:
            min_term = self.min_norm_term
        if max_term is None:
            max_term = self.max_norm_term

        if only_first:
            return arr*(max_term[0]-min_term[0]) + min_term[0]

        return arr*(max_term-min_term) + min_term


class LoadDataset(torch.utils.data.Dataset):
    """

        Args:
            seq_length:
        Attributes:
            dataset:
            X:
            y:
    """

    def __init__(self, dataset, seq_length, raw_dataset, shuffle=True):
        # normalize data. otherwise criterion cannot calculate loss
        self.dataset = dataset
        self.raw_dataset = raw_dataset
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

    def get_sample(self):
        ix = np.random.randint(low=0, high=self.__len__())
        return ix, self.__getitem__(ix=ix)

    def get_attributes(self, ix):
        return self.raw_dataset.iloc[ix:ix + self.seq_length, :]

    def __len__(self):
        """

        Returns:
            int: data count

        """
        return self.X.shape[0] - self.seq_length*2

    def __getitem__(self, ix):
        """

        Args:
            ix:

        Returns:
            (np.ndarray, np.ndarray):

        """
        # (row, seq_len, input_size)
        # return self.X[ix, :, :], self.y[ix, :]
        return self.X[ix:ix + self.seq_length, :], self.y[ix + self.seq_length-1: ix + self.seq_length*2-1]