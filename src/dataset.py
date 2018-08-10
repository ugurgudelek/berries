import torch
from torchvision import transforms
import pandas as pd
import numpy as np
from talib import RSI, SMA, MACD, WILLR, ULTOSC, MFI, STOCH



import os
import warnings
from sklearn import preprocessing
from collections import defaultdict

import matplotlib.pyplot as plt

class InnerIndicatorDataset(torch.utils.data.Dataset):
    """

    Args:
        dataset(pd.DataFrame):
    """

    def __init__(self, dataset, seq_len):
        self.dataset = dataset

        self.X = self.dataset.drop(['date', 'name', 'open', 'high', 'low', 'close',
                                    'label', 'raw_adjusted_close'], axis=1)

        self.y = self.dataset[['label']]

        # turn categorical to one hot encoding
        self.y = pd.get_dummies(self.y)

        self.name = self.dataset[['name']]

        self.feature_dim = self.X.shape[1]
        self.output_dim = self.y.shape[1]
        self.data_dim = self.X.shape[0]
        self.seq_len = seq_len

        # self._X = self.X.values.reshape(-1, self.feature_dim, self.seq_len)
        # self._y = self.y.values.reshape(-1, self.output_dim, self.seq_len)


        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.dataset.shape[0] - self.seq_len

    def __getitem__(self, ix):
        X = self.X.iloc[ix: ix + self.seq_len, :]
        y = self.y.iloc[ix + self.seq_len - 1, :]

        name = self.dataset['name'].iloc[ix: ix + self.seq_len - 1]
        date = self.dataset['date'].iloc[ix: ix + self.seq_len - 1]
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

    def __init__(self, dataset_name, input_path, save_dataset, train_valid_ratio, seq_len):

        self.dataset_name = dataset_name
        self.input_path = input_path
        self.train_valid_ratio = train_valid_ratio
        self.save_dataset = save_dataset
        self.seq_len = seq_len

        self.standardizer = IndicatorStandardizer()

        raw_dataset = pd.read_csv(input_path)
        raw_dataset['name'] = 'spy'
        train_len = int(raw_dataset.shape[0] * train_valid_ratio)
        self.raw_train_dataset = raw_dataset.iloc[:train_len, :]
        self.raw_valid_dataset = raw_dataset.iloc[train_len:, :]

        self.preprocessed_train_dataset = self.preprocess_dataset(dataset=self.raw_train_dataset, kind='train')
        self.preprocessed_valid_dataset = self.preprocess_dataset(dataset=self.raw_valid_dataset, kind='validation')

        print('Train ----\n'
              'Shape: {} \n'
              'First date: {} \n'
              'Last date: {}'.format(self.preprocessed_train_dataset.shape,
                                     self.preprocessed_train_dataset['date'].iloc[0],
                                     self.preprocessed_train_dataset['date'].iloc[-1]))

        print('Valid ----\n'
              'Shape: {} \n'
              'First date: {} \n'
              'Last date: {}'.format(self.preprocessed_valid_dataset.shape,
                                     self.preprocessed_valid_dataset['date'].iloc[0],
                                     self.preprocessed_valid_dataset['date'].iloc[-1]))

        if save_dataset:
            self.preprocessed_train_dataset.to_csv(
                os.path.join('/'.join(input_path.split('/')[:-1]), 'train_preprocessed_indicator_dataset.csv'),
                index=False)
            self.preprocessed_valid_dataset.to_csv(
                os.path.join('/'.join(input_path.split('/')[:-1]), 'valid_preprocessed_indicator_dataset.csv'),
                index=False)

        self.train_dataset = InnerIndicatorDataset(dataset=self.preprocessed_train_dataset, seq_len=self.seq_len)
        self.valid_dataset = InnerIndicatorDataset(dataset=self.preprocessed_valid_dataset, seq_len=self.seq_len)

    def preprocess_dataset(self, dataset, kind='train'):

        dataset['date'] = dataset['date'].astype('datetime64[ns]')
        dataset['high'] = dataset['high'].values.astype(np.float)
        dataset['low'] = dataset['low'].values.astype(np.float)
        dataset['adjusted_close'] = dataset['adjusted_close'].values.astype(np.float)
        dataset['volume'] = dataset['volume'].values.astype(np.float)

        # labelize with up,down,hold
        dataset = self.label_wrt_center_max_min(dataset, window=15)
        dataset = self.dilate(dataset, window=3)

        # calculate technical analysis values from stock data
        # this creates a new dataset depends on technical analysis
        dataset = IndicatorDataset.technical_analysis(dataset)

        # add seasonality
        dataset['year'] = dataset['date'].dt.year.values.astype(int)
        dataset['month'] = dataset['date'].dt.month.values.astype(int)
        dataset['week'] = dataset['date'].dt.week.values.astype(int)
        dataset['weekday'] = dataset['date'].dt.weekday.values.astype(int)
        dataset['day'] = dataset['date'].dt.day.values.astype(int)

        dataset = dataset.dropna(axis=0).reset_index(drop=True)

        dataset['raw_adjusted_close'] = dataset['adjusted_close'].values

        # make stationary, standardize
        dataset = self.differentiate(dataset, subset=['open','high','low','close',
                                                      'adjusted_close'])
        dataset = self.standardize(dataset,
                                    neg_subset=['date', 'name', 'label',
                                                'raw_adjusted_close'], kind=kind)

        # sort dataset
        dataset = dataset.sort_values(by=['date', 'name']).reset_index(drop=True)


        # # equalize up,down and hold labels
        # if kind == 'train':
        #     dataset = self.updown_scaling(dataset)

        return dataset

    def get_data(self, name, date):
        return self.raw_train_dataset.loc[
            (self.raw_train_dataset['name'] == name)&
            (self.raw_train_dataset['date'] == date)]

    def get_data_seq(self, name, first_date, last_date):
        return self.raw_train_dataset.loc[
            (self.raw_train_dataset['name'] == name) &
            (self.raw_train_dataset['date'] >= first_date)&
            (self.raw_train_dataset['date'] <= last_date)]

    def differentiate(self, stocks, subset):

        def inner_func(stock_data):
            stock_data_subset = stock_data[subset]
            stock_data_neg_subset = stock_data.drop(subset, axis=1)
            stock_data_subset = stock_data_subset.pct_change()

            return pd.concat((stock_data_subset, stock_data_neg_subset), axis=1)

        return stocks.groupby('name').apply(inner_func).dropna()

    def updown_scaling(self, stocks):

        def inner_func(stock_data):
            top_count = stock_data.loc[stock_data['label'] == 'top'].sum()
            mid_count = stock_data.loc[stock_data['label'] == 'mid'].sum()
            bot_count = stock_data.loc[stock_data['label'] == 'bot'].sum()

            def pick_random_samples(df, on, condition, n):
                return df.loc[df[on] == condition].sample(n=n, replace=True)

            new_tops = pick_random_samples(df=stock_data, on='label', condition='top',
                                                  n=int(mid_count - top_count))
            new_bots = pick_random_samples(df=stock_data, on='label', condition='bot',
                                                n=int(mid_count - bot_count))

            return pd.concat((stock_data, new_tops, new_bots))

        return stocks.groupby('name').apply(inner_func).sort_values(by=['date', 'name']).reset_index(drop=True)

    # def labelize_with_windows_slide(self, stocks, window=28):
    #
    #     def inner_func(stock_data):
    #         """look future windowth price values to label each row
    #         if window[0] is max in given window then label it with sell,
    #         if window[0] is min in given window then label it with buy,
    #         otherwise hold.
    #         """
    #         stock_data['max_28'] = stock_data['adjusted_close'].rolling(window).apply(utils.roll_is_max).shift(
    #             periods=-window + 1)
    #         stock_data['min_28'] = stock_data['adjusted_close'].rolling(window).apply(utils.roll_is_min).shift(
    #             periods=-window + 1)
    #
    #         stock_data['label_buy'] = stock_data['min_28'].values
    #         stock_data['label_sell'] = stock_data['max_28'].values
    #         stock_data['label_hold'] = 0.0
    #         stock_data.loc[(stock_data['label_buy'] != 1.0) & (stock_data['label_sell'] != 1.0), 'label_hold'] = 1.0
    #         return stock_data.drop(['max_28', 'min_28'], axis=1)
    #
    #     return stocks.groupby('name').apply(inner_func).dropna()

    def label_wrt_center_max_min(self, stocks, window=7):

        def is_center_max(window_data):
            return np.max(window_data) == window_data[len(window_data)//2]

        def is_center_min(window_data):
            return np.min(window_data) == window_data[len(window_data)//2]




        def inner_func(stock_data):
            stock_data['maxs'] = stock_data['adjusted_close'].rolling(window, center=True, min_periods=window).apply(is_center_max)
            stock_data['mins'] = stock_data['adjusted_close'].rolling(window, center=True, min_periods=window).apply(is_center_min)

            stock_data['label'] = 'mid'
            stock_data.loc[stock_data['maxs'] == 1, 'label'] = 'top'
            stock_data.loc[stock_data['mins'] == 1, 'label'] = 'bot'



            stock_data = stock_data.drop(['maxs','mins'], axis=1)

            # plt.plot(stock_data['adjusted_close'])
            # only_maxs = stock_data.loc[stock_data['label'] == 'top', 'adjusted_close']
            # only_mins = stock_data.loc[stock_data['label'] == 'bot', 'adjusted_close']
            # plt.scatter(x=only_maxs.index.values, y=only_maxs, c='r')
            # plt.scatter(x=only_mins.index.values, y=only_mins, c='y')
            # plt.grid(which='minor')
            # plt.show()
            return stock_data

        return stocks.groupby('name').apply(inner_func).dropna()

    def dilate(self, stocks, window=3):

        def inner_func(stock_data):
            stock_data['label2'] = stock_data['label']
            stock_data['label2'] = np.convolve((stock_data['label'] == 'top').values, np.ones(window),'same')
            stock_data['label'].loc[stock_data['label2'] == 1] = 'top'

            stock_data['label2'] = stock_data['label']
            stock_data['label2'] = np.convolve((stock_data['label'] == 'bot').values, np.ones(window), 'same')
            stock_data['label'].loc[stock_data['label2'] == 1] = 'bot'

            return stock_data.drop('label2', axis=1)

        return stocks.groupby('name').apply(inner_func)

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
        dataframe['sma_20'] = SMA(close, timeperiod=20)

        dataframe['macd_12'], macdsignal, dataframe['macdhist_12'] = MACD(close, fastperiod=12, slowperiod=26,
                                                                          signalperiod=9)

        dataframe['willR_14'] = WILLR(high, low, close, timeperiod=14) / 50 + 1

        dataframe['ultimate_osc_7'] = ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28) / 50 - 1

        dataframe['mfi_14'] = MFI(high, low, close, volume, timeperiod=14) / 50 - 1

        slowk, slowd = STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3,
                             slowd_matype=0)
        dataframe['kddiff_14'] = slowk - slowd

        return dataframe

if __name__ == "__main__":
    path = '../input/spy.csv'
