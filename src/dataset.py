"""
Ugur Gudelek
dataset
ugurgudelek
06-Mar-18
finance-cnn
"""
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
# from talib import RSI, SMA, MACD, WILLR, ULTOSC, MFI, STOCH

import os
from sklearn import preprocessing

from config import Config

import matplotlib.pyplot as plt


class InnerDataset(Dataset):
    """

    Args:
        dataset(pd.DataFrame):
    """
    def __init__(self, dataset):

        self.dataset = dataset

        self.X = self.dataset.drop(['label','name'], axis=1)
        self.y = self.dataset[['label']]
        self.name = self.dataset[['name']]

        self.image_width = self.X.shape[1]

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.dataset.shape[0] - self.image_width - 1

    def __getitem__(self, ix):
        X = self.X.iloc[ix: ix + self.image_width, :]
        y = self.y.iloc[ix + self.image_width - 1]

        # change type to numpy
        X = X.values
        y = y.values.flatten()

        X = X.astype(float)
        y = y.astype(float)


        X = np.expand_dims(X, axis=0)

        return (X,y)

    def _reshape(self, data):
        # (in_channels, width, height)
        return data.reshape((1, data.shape[0], data.shape[1]))

class IndicatorDataset(Dataset):
    """

    """

    def __init__(self, stocks_dir, stock_names=None, label_after=20, row_len=28):
        self.stocks_dir = stocks_dir

        # read only necessary stocks
        self.stocks = self._read_dir(self.stocks_dir, stock_names)

        # assign labels
        for stock_name, stock_data in self.stocks.items():
            self.stocks[stock_name]['label'] = stock_data.loc[:, 'adjusted_close'].shift(-label_after)

        # calculate technical analysis values from stock data
        # this creates a new dataset depends on technical analysis
        self.dataset = IndicatorDataset.technical_analysis(self.stocks)

        for stock_name, data in self.dataset.items():

            # change dtypes
            data['date'] = pd.to_datetime(data['date'])
            data['high'] = data['high'].values.astype(np.float)
            data['low'] = data['low'].values.astype(np.float)
            data['adjusted_close'] = data['adjusted_close'].values.astype(np.float)
            data['volume'] = data['volume'].values.astype(np.float)

            # add seasonality
            data['day'] = data['date'].apply(lambda x: x.day)
            data['weekday'] = data['date'].apply(lambda x: x.weekday())
            data['week'] = data['date'].apply(lambda x: x.week)
            data['month'] = data['date'].apply(lambda x: x.month)
            data['year'] = data['date'].apply(lambda x: x.year)

            self.normalize(data)

            # dropna
            data = data.dropna(axis=0)

            # filter features
            indexes = data.index
            dates = data['date']
            self.dataset[stock_name] = data = data.drop(['date', 'open', 'high', 'low', 'close'], axis=1)

            # scaler = preprocessing.StandardScaler().fit(data)
            # data = scaler.transform(data)
            # self.dataset[stock_name].iloc[:, :] = data

            # assign fall,rise and hold labels
            # label_split_threshold = 0.27
            # label_pct = self.dataset[stock_name].copy().loc[:, 'label']

            # self.dataset[stock_name].loc[(label_pct <= -label_split_threshold), 'label'] = 0 # fall
            # self.dataset[stock_name].loc[(label_pct >= label_split_threshold), 'label'] = 2 # rise
            # self.dataset[stock_name].loc[((-label_split_threshold < label_pct) & (label_pct < label_split_threshold)), 'label'] = 1 # steady

            # set multiindex(index,date)
            self.dataset[stock_name].index = pd.MultiIndex.from_tuples(list(zip(*[indexes, dates])),
                                                                       names=['index', 'date'])

            # check shape
            assert data.shape[1] == row_len + 1  # +1 for label


        # merged all stocks into one big chunk of data
        merged = np.empty(shape=(0,data.shape[1] + 1))
        for stock_name, stock_df in self.dataset.items():
            stock_df['name'] = stock_name
            merged = np.vstack((merged, stock_df.values))

        col_names = self.dataset['spy'].columns
        self.dataset = pd.DataFrame(merged, columns=col_names)


        train_len = int(self.dataset.shape[0] * 0.9)
        self.train_dataset = InnerDataset(self.dataset.iloc[:train_len, :])
        self.valid_dataset = InnerDataset(self.dataset.iloc[train_len:, :])

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

    def normalize(self, data):

        # change values to percentage change
        data.loc[:, 'open'] = data.loc[:, 'open'].pct_change()
        data.loc[:, 'high'] = data.loc[:, 'high'].pct_change()
        data.loc[:, 'low'] = data.loc[:, 'low'].pct_change()
        data.loc[:, 'close'] = data.loc[:, 'close'].pct_change()

        data.loc[:, 'adjusted_close'] = data.loc[:, 'adjusted_close'].pct_change()
        data.loc[:, 'label'] = data.loc[:, 'label'].pct_change()
        # data.loc[:, 'sma_15'] = data.loc[:, 'sma_15'].pct_change()
        data.loc[:, 'sma_20'] = data.loc[:, 'sma_20'].pct_change()
        # data.loc[:, 'sma_25'] = data.loc[:, 'sma_25'].pct_change()
        data.loc[:, 'sma_30'] = data.loc[:, 'sma_30'].pct_change()
        data.loc[:, 'volume'] = data.loc[:, 'volume'].pct_change()

        # other technical values are already in normalized form.

    @staticmethod
    def technical_analysis(stocks):
        """
        Wrapper for indicator calculation
        Args:
            stocks:

        Returns:

        """
        r = dict()
        for (name, stock) in stocks.items():
            r[name] = IndicatorDataset.indicators(stock.copy())
        return r

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

        dataframe['rsi_15'] = RSI(close, timeperiod=15)/50 - 1
        dataframe['rsi_20'] = RSI(close, timeperiod=20)/50 - 1
        # dataframe['rsi_25'] = RSI(close, timeperiod=25)/50 - 1
        dataframe['rsi_30'] = RSI(close, timeperiod=30)/50 - 1

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

        dataframe['willR_14'] = WILLR(high, low, close, timeperiod=14)/50 + 1
        # dataframe['willR_18'] = WILLR(high, low, close, timeperiod=18)/50 + 1
        dataframe['willR_22'] = WILLR(high, low, close, timeperiod=22)/50 + 1

        dataframe['ultimate_osc_7'] = ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)/50 - 1
        # dataframe['ultimate_osc_8'] = ULTOSC(high, low, close, timeperiod1=8, timeperiod2=16, timeperiod3=32)/50 - 1
        dataframe['ultimate_osc_9'] = ULTOSC(high, low, close, timeperiod1=9, timeperiod2=18, timeperiod3=36)/50 - 1

        dataframe['mfi_14'] = MFI(high, low, close, volume, timeperiod=14)/50 - 1
        dataframe['mfi_18'] = MFI(high, low, close, volume, timeperiod=18)/50 - 1
        dataframe['mfi_22'] = MFI(high, low, close, volume, timeperiod=22)/50 - 1

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


if __name__ == "__main__":
    config = Config()

    dataset = IndicatorDataset(config=config, stock_names=['spy'], label_after=30)
    print(dataset.train_dataset.__getitem__(0))

