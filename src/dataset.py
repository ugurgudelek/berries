"""
Ugur Gudelek
dataset
ugurgudelek
06-Mar-18
finance-cnn
"""
import torch
from torchvision import transforms
import pandas as pd
import numpy as np
from talib import RSI, SMA, MACD, WILLR, ULTOSC, MFI, STOCH

import utils

import os
import warnings
from sklearn import preprocessing
from collections import defaultdict
from config import Config

import matplotlib.pyplot as plt


class InnerIndicatorDataset(torch.utils.data.Dataset):
    """

    Args:
        dataset(pd.DataFrame):
    """

    def __init__(self, dataset):
        self.dataset = dataset

        self.X = self.dataset.drop(['date', 'open', 'high', 'low', 'close', 'name',
                                    'label_sell', 'label_buy', 'label_hold', 'raw_adjusted_close'], axis=1)

        self.y = self.dataset[['label_sell', 'label_buy', 'label_hold']]
        self.name = self.dataset[['name']]

        self.image_width = self.X.shape[1]

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.dataset.shape[0] - self.image_width - 1

    def __getitem__(self, ix):
        X = self.X.iloc[ix: ix + self.image_width, :]
        y = self.y.iloc[ix + self.image_width - 1, :]

        name = self.dataset['name'].iloc[ix + self.image_width - 1]
        date = self.dataset['date'].iloc[ix + self.image_width - 1]
        extra_info = {'name': name, 'date': date}

        # change type to numpy
        X = X.values.astype(float)
        y = y.values.astype(float)

        X = np.expand_dims(X, axis=0)

        return X, y, extra_info

    def _reshape(self, data):
        # (in_channels, width, height)
        return data.reshape((1, data.shape[0], data.shape[1]))

    def get_sample(self):
        ix = np.random.randint(low=0, high=self.__len__())
        return ix, self.__getitem__(ix=ix)


class IndicatorStandardizer:
    def __init__(self):
        self.means = defaultdict(dict)
        self.stds = defaultdict(dict)

    def apply_standardization(self, series, stock_name, kind):
        stock_name = stock_name
        series_name = series.name

        first_idx = series.index[0]
        if isinstance(series[first_idx], float) or isinstance(series[first_idx], np.integer):
            if kind == 'train':
                mu = series.mean()
                sigma = series.std()
                # save
                self.means[stock_name][series_name] = series.mean()
                self.stds[stock_name][series_name] = series.std()

            elif kind == 'validation':
                mu = self.means[stock_name][series_name]
                sigma = self.stds[stock_name][series_name]

            else:
                raise Exception('Invalid Type. Only train and validation allowed')

        return (series - mu) / sigma


class IndicatorDataset():
    """

    """

    def __init__(self, dataset_name, input_path, stock_names, save_dataset, train_valid_ratio):

        self.dataset_name = dataset_name
        self.input_path = input_path
        self.train_valid_ratio = train_valid_ratio
        self.stock_names = stock_names
        self.save_dataset = save_dataset

        self.standardizer = IndicatorStandardizer()

        raw_dataset = pd.read_csv(input_path)
        if stock_names is not None:
            raw_dataset = raw_dataset.loc[np.isin(raw_dataset['name'], stock_names)]

        train_len = int(raw_dataset.shape[0] * train_valid_ratio)
        raw_train_dataset = raw_dataset.iloc[:train_len, :]
        raw_valid_dataset = raw_dataset.iloc[train_len:, :]

        self.preprocessed_train_dataset = self.preprocess_dataset(dataset=raw_train_dataset, kind='train')
        self.preprocessed_valid_dataset = self.preprocess_dataset(dataset=raw_valid_dataset, kind='validation')

        if save_dataset:
            self.preprocessed_train_dataset.to_csv(
                os.path.join('/'.join(input_path.split('/')[:-1]), 'train_preprocessed_indicator_dataset.csv'),
                index=False)
            self.preprocessed_valid_dataset.to_csv(
                os.path.join('/'.join(input_path.split('/')[:-1]), 'valid_preprocessed_indicator_dataset.csv'),
                index=False)

        self.train_dataset = InnerIndicatorDataset(dataset=self.preprocessed_train_dataset)
        self.valid_dataset = InnerIndicatorDataset(dataset=self.preprocessed_valid_dataset)

    def preprocess_dataset(self, dataset, kind='train'):

        dataset['date'] = dataset['date'].astype('datetime64[ns]')
        dataset['high'] = dataset['high'].values.astype(np.float)
        dataset['low'] = dataset['low'].values.astype(np.float)
        dataset['adjusted_close'] = dataset['adjusted_close'].values.astype(np.float)
        dataset['volume'] = dataset['volume'].values.astype(np.float)

        # labelize with up,down,hold
        dataset = self.labelize_with_windows_slide(dataset)

        # calculate technical analysis values from stock data
        # this creates a new dataset depends on technical analysis
        dataset = IndicatorDataset.technical_analysis(dataset)

        # add seasonality
        dataset['year'] = dataset['date'].dt.year.values.astype(int)
        dataset['month'] = dataset['date'].dt.month.values.astype(int)
        dataset['week'] = dataset['date'].dt.week.values.astype(int)
        dataset['weekday'] = dataset['date'].dt.weekday.values.astype(int)
        dataset['day'] = dataset['date'].dt.day.values.astype(int)

        dataset = utils.save_column(dataset, col_name='adjusted_close')

        # make stationary, standardize
        # self.dataset = self.normalize(self.dataset).dropna(axis=0)
        dataset = self.standardize(dataset,
                                    neg_subset=['date', 'name', 'label_buy', 'label_sell', 'label_hold',
                                                'raw_adjusted_close'], kind=kind)

        # sort dataset
        dataset = dataset.sort_values(by=['date', 'name']).reset_index(drop=True)

        # # equalize up,down and hold labels
        dataset = self.updown_scaling(dataset)

        return dataset

    def updown_scaling(self, stocks):

        def inner_func(stock_data):
            buy_count = stock_data['label_buy'].sum()
            sell_count = stock_data['label_sell'].sum()
            hold_count = stock_data['label_hold'].sum()

            new_downs = utils.pick_random_samples(df=stock_data, on='label_buy', condition=1,
                                                  n=int(hold_count - buy_count))
            new_ups = utils.pick_random_samples(df=stock_data, on='label_sell', condition=1,
                                                n=int(hold_count - sell_count))

            return pd.concat((stock_data, new_ups, new_downs))

        return stocks.groupby('name').apply(inner_func).sort_values(by=['date', 'name']).reset_index(drop=True)

    def labelize_with_windows_slide(self, stocks, window=28):

        def inner_func(stock_data):
            """look future windowth price values to label each row
            if window[0] is max in given window then label it with sell,
            if window[0] is min in given window then label it with buy,
            otherwise hold.
            """
            stock_data['max_28'] = stock_data['adjusted_close'].rolling(window).apply(utils.roll_is_max).shift(
                periods=-window + 1)
            stock_data['min_28'] = stock_data['adjusted_close'].rolling(window).apply(utils.roll_is_min).shift(
                periods=-window + 1)

            stock_data['label_buy'] = stock_data['min_28'].values
            stock_data['label_sell'] = stock_data['max_28'].values
            stock_data['label_hold'] = 0.0
            stock_data.loc[(stock_data['label_buy'] != 1.0) & (stock_data['label_sell'] != 1.0), 'label_hold'] = 1.0
            return stock_data.drop(['max_28', 'min_28'], axis=1)

        return stocks.groupby('name').apply(inner_func).dropna()

    def _read_dir(self, stocks_dir, stock_names):
        """

        Args:
            stocks_dir:

        Returns: (dict of pd.DataFrame) stock dictionary

        """
        stocks = dict()
        for fullfilename in os.listdir(stocks_dir):
            filename, extension = fullfilename.split('.')
            if extension == 'csv':  # check extension
                if filename in stock_names:
                    stocks[filename] = pd.read_csv(os.path.join(stocks_dir, fullfilename))

        return stocks

    def standardize(self, stocks, neg_subset, kind):

        # for (stock_name, stock_data) in stocks.groupby('name'):
        #     stock_neg_subset = stock_data[neg_subset]
        #     stock_subset = stock_data.drop(neg_subset, axis=1)
        #
        #     stock_subset = stock_subset.apply(, args=(mu, sigma), axis=0)
        #
        #     stock_data = pd.concat((stock_neg_subset, stock_subset), axis=1)
        #
        #     # line below is ugly. can we fix this ?
        #     stocks.loc[stocks['name'] == stock_name] = stock_data
        #
        #     mus_out[stock_name] = stock_data.mean()
        #     sigmas_out[stock_name] = stock_data.std()

        def inner_func(data, kind):
            stock_name = data['name'].iloc[0]
            data_neg_subset = data[neg_subset]
            data_subset = data.drop(neg_subset, axis=1)
            data_subset = data_subset.apply(lambda x: self.standardizer.apply_standardization(x, kind=kind, stock_name=stock_name), axis=0)
            return pd.concat((data_neg_subset, data_subset),axis=1)

        return stocks.groupby('name').apply(lambda x: inner_func(x, kind)).dropna()



    def make_stationary(self, stocks):

        def inner_func(data):
            # change values to percentage change
            data.loc[:, 'open'] = data.loc[:, 'open'].pct_change()
            data.loc[:, 'high'] = data.loc[:, 'high'].pct_change()
            data.loc[:, 'low'] = data.loc[:, 'low'].pct_change()
            data.loc[:, 'close'] = data.loc[:, 'close'].pct_change()
            data.loc[:, 'adjusted_close'] = data.loc[:, 'adjusted_close'].pct_change()
            # data.loc[:, 'sma_15'] = data.loc[:, 'sma_15'].pct_change()
            data.loc[:, 'sma_20'] = data.loc[:, 'sma_20'].pct_change()
            # data.loc[:, 'sma_25'] = data.loc[:, 'sma_25'].pct_change()
            data.loc[:, 'sma_30'] = data.loc[:, 'sma_30'].pct_change()
            data.loc[:, 'volume'] = data.loc[:, 'volume'].pct_change()

            return data

        return stocks.groupby('name').apply(inner_func)

    def normalize(self, stocks):

        def inner_func(data):
            data['year'] = utils.normalize(data['year'])
            data['month'] = utils.normalize(data['month'])
            data['week'] = utils.normalize(data['week'])
            data['weekday'] = utils.normalize(data['weekday'])
            data['day'] = utils.normalize(data['day'])

            # other technical values are already in normalized form.
            return data

        return stocks.groupby('name').apply(inner_func)

    @staticmethod
    def technical_analysis(stocks: pd.DataFrame):
        """
        Wrapper for indicator calculation
        Args:
            stocks:

        Returns:

        """

        return stocks.groupby('name').apply(IndicatorDataset.indicators)

    @staticmethod
    def indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
        """

        Args:
            dataframe(pd.DataFrame): should have 'high', 'low', 'adjusted_close', 'volume'

        Returns: (pd.DataFrame) calculated indicators

        """

        high = dataframe['high'].values
        low = dataframe['low'].values
        close = dataframe['adjusted_close'].values
        volume = dataframe['volume'].values

        dataframe['rsi_15'] = RSI(close, timeperiod=15) / 50 - 1
        dataframe['rsi_20'] = RSI(close, timeperiod=20) / 50 - 1
        # dataframe['rsi_25'] = RSI(close, timeperiod=25)/50 - 1
        dataframe['rsi_30'] = RSI(close, timeperiod=30) / 50 - 1

        # dataframe['sma_15'] = SMA(close, timeperiod=15)
        dataframe['sma_20'] = SMA(close, timeperiod=20)
        # dataframe['sma_25'] = SMA(close, timeperiod=25)
        dataframe['sma_30'] = SMA(close, timeperiod=30)

        dataframe['macd_12'], macdsignal, dataframe['macdhist_12'] = MACD(close, fastperiod=12, slowperiod=26,
                                                                          signalperiod=9)
        dataframe['macd_14'], macdsignal, dataframe['macdhist_14'] = MACD(close, fastperiod=14, slowperiod=28,
                                                                          signalperiod=10)
        dataframe['macd_16'], macdsignal, dataframe['macdhist_16'] = MACD(close, fastperiod=16, slowperiod=30,
                                                                          signalperiod=11)

        dataframe['willR_14'] = WILLR(high, low, close, timeperiod=14) / 50 + 1
        # dataframe['willR_18'] = WILLR(high, low, close, timeperiod=18)/50 + 1
        dataframe['willR_22'] = WILLR(high, low, close, timeperiod=22) / 50 + 1

        dataframe['ultimate_osc_7'] = ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28) / 50 - 1
        # dataframe['ultimate_osc_8'] = ULTOSC(high, low, close, timeperiod1=8, timeperiod2=16, timeperiod3=32)/50 - 1
        dataframe['ultimate_osc_9'] = ULTOSC(high, low, close, timeperiod1=9, timeperiod2=18, timeperiod3=36) / 50 - 1

        dataframe['mfi_14'] = MFI(high, low, close, volume, timeperiod=14) / 50 - 1
        dataframe['mfi_18'] = MFI(high, low, close, volume, timeperiod=18) / 50 - 1
        dataframe['mfi_22'] = MFI(high, low, close, volume, timeperiod=22) / 50 - 1

        slowk, slowd = STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3,
                             slowd_matype=0)
        dataframe['kddiff_14'] = slowk - slowd
        slowk, slowd = STOCH(high, low, close, fastk_period=18, slowk_period=3, slowk_matype=0, slowd_period=3,
                             slowd_matype=0)
        dataframe['kddiff_18'] = slowk - slowd
        slowk, slowd = STOCH(high, low, close, fastk_period=22, slowk_period=3, slowk_matype=0, slowd_period=3,
                             slowd_matype=0)
        dataframe['kddiff_22'] = slowk - slowd

        return dataframe


class LoadDataset():
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
        valid_values = self.dataset[train_len:train_len + valid_len, :]

        raw_train_dataset = self.raw_dataset.iloc[:train_len, :]
        raw_valid_dataset = self.raw_dataset.iloc[train_len:train_len + valid_len, :]

        self.train_dataset = InnerLoadDataset(train_values, seq_length=seq_length, raw_dataset=raw_train_dataset)
        self.valid_dataset = InnerLoadDataset(valid_values, seq_length=seq_length, raw_dataset=raw_valid_dataset)

    def get_raw_valid_dataset(self):
        return self.raw_dataset[self.train_len:self.train_len + self.valid_len]

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
            return arr * (max_term[0] - min_term[0]) + min_term[0]

        return arr * (max_term - min_term) + min_term


class InnerLoadDataset(torch.utils.data.Dataset):
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
        return self.X.shape[0] - self.seq_length * 2

    def __getitem__(self, ix):
        """

        Args:
            ix:

        Returns:
            (np.ndarray, np.ndarray):

        """
        # (row, seq_len, input_size)
        # return self.X[ix, :, :], self.y[ix, :]
        return self.X[ix:ix + self.seq_length, :], self.y[ix + self.seq_length - 1: ix + self.seq_length * 2 - 1]


def get_dataset_cls_from_name(name):
    if name == 'IndicatorDataset':
        return IndicatorDataset

    if name == 'LoadDataset':
        return LoadDataset


if __name__ == "__main__":
    config = Config()

    dataset = IndicatorDataset(dataset_name='IndicatorDataset',
                               input_path='../dataset/finance/stocks/stocks_sample.csv',
                               train_valid_ratio=0.9)
    print(dataset.train_dataset.__getitem__(14))
    print()
