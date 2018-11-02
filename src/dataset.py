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

import plots
import utils
import os
import warnings
from sklearn import preprocessing
from collections import defaultdict
from scipy import spatial

import matplotlib.pyplot as plt

from torchvision import datasets, transforms

from sklearn.preprocessing import OneHotEncoder

from functools import partial
from itertools import cycle

class GenericDataset():



    def random_train_sample(self, n):
        return GenericDataset._random_sample(self.train_dataset, n)

    def random_valid_sample(self, n):
        return GenericDataset._random_sample(self.valid_dataset, n)

    @staticmethod
    def _random_sample(dataset, n):
        perm = np.random.randint(0, dataset.__len__(), size=n)
        data = dataset.data.numpy()[perm]
        labels = dataset.labels.numpy()[perm]

        return torch.Tensor(data).unsqueeze(dim=1), torch.Tensor(labels).long()

"""
1. Sequence Learning Problem
2. Value Memorization
3. Echo Random Integer
4. Echo Random Subsequences
5. Sequence Classification
"""


class SequenceLearningOneToOne(GenericDataset):

    def __init__(self):
        dataset = np.array(list(np.arange(0, 10, 1))*10)

        self.train_dataset = self.Inner(dataset=dataset)
        self.valid_dataset = self.Inner(dataset=dataset)

    class Inner(torch.utils.data.Dataset):
        def __init__(self, dataset):
            X = dataset[:-1].reshape(-1,1)
            y = dataset[1:]
            self.encode = OneHotEncoder(sparse=False)
            X = self.encode.fit_transform(X)
            # self.y = self.encode.transform(self.y)

            # norm_dataset = (dataset - dataset.min())/(dataset.max() - dataset.min())
            # self.dataset = norm_dataset


            self.data = torch.FloatTensor(X).unsqueeze(dim=1)
            self.labels = torch.LongTensor(y)



        def __len__(self):
            return self.data.__len__()

        def __getitem__(self, ix):
            return self.data[ix, :], self.labels[ix]




class SequenceLearningManyToOne(GenericDataset):
    def __init__(self, seq_len=3, dataset_len=100, onehot=False):
        # todo : add seq_range

        seq7s = list()
        while(True):
            seq7 = list()
            rnd = np.random.randint(0, seq_len)
            sequence = cycle(np.arange(0, seq_len, 1))
            for i in range(seq_len):
                digit = next(sequence)
                seq7.append(digit)
            seq7s.append(seq7)

            if len(seq7s) == dataset_len:
                break

        dataset = np.array(seq7s)
        np.random.shuffle(dataset)
        self.encode = OneHotEncoder(sparse=False)
        dataset = self.encode.fit_transform(dataset.T).T

        self.train_dataset = self.Inner(dataset=dataset, seq_len=seq_len)
        self.valid_dataset = self.Inner(dataset=dataset, seq_len=seq_len)

    class Inner(torch.utils.data.Dataset, GenericDataset):
        def __init__(self, dataset, seq_len):
            # self.encode = OneHotEncoder(sparse=False)
            # X = self.encode.fit_transform(X)
            # self.y = self.encode.transform(self.y)

            # norm_dataset = (dataset - dataset.min())/(dataset.max() - dataset.min())
            # self.dataset = norm_dataset

            X = dataset[:, :].astype(np.float32)
            y = (dataset[:, -1] + 1).__mod__(seq_len).astype(np.float32)

            self.data = torch.FloatTensor(X).unsqueeze(dim=1)
            self.labels = torch.LongTensor(y)





        def __len__(self):
            return self.data.__len__()

        def __getitem__(self, ix):
            return self.data[ix, :], self.labels[ix]









class ValueMemorizationOneToOne:
    def __init__(self):
        dataset = np.random.randn(100)
        self.X = dataset[:-1]
        self.y = dataset[1:]

    def __getitem__(self, ix):
        return self.X[ix], self.y[ix]

    def __len__(self):
        return self.X.__len__()

    def random_sample(self):
        ix = np.random.randint(self.__len__())
        return self.__getitem__(ix)
class ValueMemorizationManyToOne:
    def __init__(self):
        pass
    def __getitem__(self, ix):
        pass
    def __len__(self):
        pass
class EchoRandomInteger:
    def __init__(self, lag=0):
        self.lag = lag
    def __getitem__(self, ix):
        sequence = np.random.randn(10)
        return sequence, sequence[sequence.__len__()-self.lag-1]
    def __len__(self):
        return 100

class EchoRandomSubsequences:
    def __init__(self, lag=0):
        self.lag = lag

    def __getitem__(self, ix):
        sequence = np.random.randn(10)
        return sequence, sequence[sequence.__len__() - self.lag - 1]

    def __len__(self):
        return 100

class MNISTDataset():
    def __init__(self, config):

        self.train_dataset = datasets.MNIST('../dataset', train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))
                                            ])
                                            )
        self.valid_dataset = datasets.MNIST('../dataset', train=False, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))
                                            ])
                                            )
        # todo: check for transformation later


    def random_sample(self, n):
        perm = np.random.randint(0, self.train_dataset.__len__(), size=n)
        data = self.train_dataset.train_data.numpy()[perm]
        labels = self.train_dataset.train_labels.numpy()[perm]

        return torch.Tensor(data).unsqueeze(dim=1), torch.Tensor(labels).long()

class IndicatorDataset():
    """

    """

    def __init__(self, dataset_name, input_path, save_dataset, train_valid_ratio, seq_len, label_type='classification'):

        self.dataset_name = dataset_name
        self.input_path = input_path
        self.train_valid_ratio = train_valid_ratio
        self.save_dataset = save_dataset
        self.seq_len = seq_len
        self.label_type = label_type

        self.standardizer = IndicatorDataset.IndicatorStandardizer()

        raw_dataset = pd.read_csv(input_path)
        raw_dataset['name'] = 'spy'
        train_len = int(raw_dataset.shape[0] * train_valid_ratio)
        self.raw_train_dataset = raw_dataset.iloc[:train_len, :]
        self.raw_valid_dataset = raw_dataset.iloc[train_len:, :]

        self.preprocessed_train_dataset = self.preprocess_dataset(dataset=self.raw_train_dataset, kind='train', label_type=self.label_type)
        self.preprocessed_valid_dataset = self.preprocess_dataset(dataset=self.raw_valid_dataset, kind='validation', label_type=self.label_type)

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

        self.train_dataset = IndicatorDataset.InnerIndicatorDataset(dataset=self.preprocessed_train_dataset, seq_len=self.seq_len, problem_type=self.label_type)
        self.valid_dataset = IndicatorDataset.InnerIndicatorDataset(dataset=self.preprocessed_valid_dataset, seq_len=self.seq_len, problem_type=self.label_type)
        print()

    def preprocess_dataset(self, dataset, kind='train', label_type='classification'):

        dataset['date'] = dataset['date'].astype('datetime64[ns]')
        dataset['high'] = dataset['high'].values.astype(np.float)
        dataset['low'] = dataset['low'].values.astype(np.float)
        dataset['adjusted_close'] = dataset['adjusted_close'].values.astype(np.float)
        dataset['volume'] = dataset['volume'].values.astype(np.float)



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
                                    neg_subset=['date', 'name',
                                                'raw_adjusted_close'], kind=kind)

        if label_type == 'classification':
            # labelize with up,down,hold
            dataset = self.label_top_bot_mid(dataset, window=15)
            dataset = self.dilate(dataset, window=3)
        if label_type == 'regression':
            # labelize with respect to distance
            # dataset = self.label_wrt_distance(dataset, window=15)

            dataset = dataset.reset_index(drop=True)
            dataset = plots.zigzag(stock=dataset, on='adjusted_close', window=15)
            dataset['label'] = dataset['zigzag']
            dataset = dataset.drop('zigzag', axis=1)

            dataset = dataset.drop(['volume', 'year', 'month', 'week', 'weekday', 'day'], axis=1)

            print('Features: {}'.format(list(dataset.columns)))

        # sort dataset
        dataset = dataset.sort_values(by=['date', 'name']).reset_index(drop=True)


        # # equalize up,down and hold labels
        # if kind == 'train':
        #     dataset = self.updown_scaling(dataset)

        return dataset

    @staticmethod
    def is_center_max(window_data):
        return np.max(window_data) == window_data[len(window_data)//2]
    @staticmethod
    def is_center_min(window_data):
        return np.min(window_data) == window_data[len(window_data)//2]

    @staticmethod
    def plot_top_bot_turning_point(stock_data):
        close = stock_data['adjusted_close'].values
        labels = stock_data['label'].values
        x = range(len(close))

        # (r,g,b,a)
        colormap = {'mid':(0.2, 0.4, 0.6, 0), 'top':(1, 0, 0, 0.7), 'bot':(0, 0, 1, 0.7)}
        colors = [colormap[label] for label in labels]

        plt.scatter(x=x, y=close, c=colors)
        plt.plot(x, close, lw=1, label='close')
        plt.legend()


    def label_top_bot_mid(self, stocks, window=7):

        def inner_func(stock_data):
            stock_data['maxs'] = stock_data['adjusted_close'].rolling(window, center=True, min_periods=window).apply(IndicatorDataset.is_center_max)
            stock_data['mins'] = stock_data['adjusted_close'].rolling(window, center=True, min_periods=window).apply(IndicatorDataset.is_center_min)

            stock_data['label'] = 'mid'
            stock_data.loc[stock_data['maxs'] == 1, 'label'] = 'top'
            stock_data.loc[stock_data['mins'] == 1, 'label'] = 'bot'



            stock_data = stock_data.drop(['maxs','mins'], axis=1)

            return stock_data

        return stocks.groupby('name').apply(inner_func).dropna()

    def label_wrt_distance(self, stocks, window=7):

        # point turning points then process for distance
        # after this line, stocks has 'label' column which has top-mid-bot values.
        stocks = self.label_top_bot_mid(stocks=stocks, window=window)
        stocks = self.filter_consequtive_same_label(stocks=stocks)
        stocks = self.crop_firstnonbot_and_lastnontop(stocks=stocks)

        def distance(idxs, turning_points):
            """
            Assumes turning_points start with increasing segment.
            :param idxs:
            :param turning_points:
            :return:
            """
            segments = np.array(list(zip(turning_points[:-1], turning_points[1:])))

            def calc_dist(x, lower, upper):
                return (x-lower)/(upper-lower)

            state = True
            dist = np.zeros_like(idxs, dtype=np.float)
            for (lower,upper) in segments:
                for i in range(lower,upper):
                    if state:
                        dist[i] = calc_dist(i, lower, upper)
                    else:
                        dist[i] = 1 - calc_dist(i, lower, upper)

                state = not state

            return dist

        def inner_func(stock_data):
            mid_idxs = stock_data.loc[stock_data['label'] == 'mid'].index.values
            top_idxs = stock_data.loc[stock_data['label'] == 'top'].index.values
            bot_idxs = stock_data.loc[stock_data['label'] == 'bot'].index.values

            turning_points = np.sort(np.concatenate((bot_idxs, top_idxs)))
            stock_data['label'] = distance(stock_data.index.values, turning_points)

            # at this point "label" values are between 0-1 range.
            # let me add -0.5 bias
            stock_data['label'] = stock_data['label'] - 0.5

            return stock_data

        return stocks.groupby('name').apply(inner_func).dropna()

    def filter_consequtive_same_label(self, stocks):

        def inner_func(stock_data):
            state = None
            for i,row in stock_data.iterrows():
                if row['label'] == 'mid':
                    continue
                if state is None:
                    state = row['label']
                    continue
                if state == row['label']:
                    stock_data.loc[i,'label'] = 'mid'
                else:
                    state = row['label']
            return stock_data

        return stocks.groupby('name').apply(inner_func).dropna()

    def crop_firstnonbot_and_lastnontop(self, stocks):

        def inner_func(stock_data):
            first_bot_idx = stock_data.loc[stock_data['label'] == 'bot'].index.values[0]
            last_top_idx = stock_data.loc[stock_data['label'] == 'top'].index.values[-1]

            return stock_data.loc[first_bot_idx:last_top_idx, :]

        return stocks.groupby('name').apply(inner_func).dropna().reset_index(drop=True)

    @staticmethod
    def nearest_neighbour(arr, search_space):
        """
        requires sorted search_space
        :return (np.ndarray) nearest neighbours of arr in search_space
        """
        ret = []
        for a in arr:
            min_idx = np.argmin(np.abs(search_space - a))
            ret.append(min_idx)
        return np.array(ret)




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
        # todo: this function should be grouped by 'name'
        def inner_func(stock_data):
            stock_data_subset = stock_data[subset]
            stock_data_neg_subset = stock_data.drop(subset, axis=1)
            stock_data_subset = stock_data_subset.pct_change()

            return pd.concat((stock_data_subset, stock_data_neg_subset), axis=1)

        return inner_func(stocks).dropna()

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
        slowk, slowd = STOCH(high, low, close, fastk_period=18, slowk_period=3, slowk_matype=0, slowd_period=3,
                             slowd_matype=0)
        dataframe['kddiff_18'] = slowk - slowd
        slowk, slowd = STOCH(high, low, close, fastk_period=22, slowk_period=3, slowk_matype=0, slowd_period=3,
                             slowd_matype=0)
        dataframe['kddiff_22'] = slowk - slowd

        return dataframe

    class InnerIndicatorDataset(torch.utils.data.Dataset):
        """

        Args:
            dataset(pd.DataFrame):
        """

        def __init__(self, dataset, seq_len, problem_type):
            self.dataset = dataset

            self.X = self.dataset.drop(['date', 'name', 'open', 'high', 'low', 'close',
                                        'label', 'raw_adjusted_close'], axis=1)

            self.y = self.dataset[['label']]

            if problem_type == 'classification':
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

        def get_all_data(self, transforms=None):
            xs, ys, _ = self.__getitem__(0)
            for ix in range(1, self.__len__()):
                X, y, info = self.__getitem__(ix)
                xs = np.append(xs, X, axis=0)
                ys = np.append(ys, y, axis=0)

            # tranform
            # example:
            # transforms=[FloatTensor, Variable])
            # xs = Variable(FloatTensor(xs))
            if transforms is not None:
                for transform in transforms:
                    xs, ys = transform(xs), transform(ys)

            return xs.unsqueeze_(-1), ys.unsqueeze_(-1)

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

        self.train_dataset = LoadDataset.InnerLoadDataset(train_values, seq_length=seq_length, raw_dataset=raw_train_dataset)
        self.valid_dataset = LoadDataset.InnerLoadDataset(valid_values, seq_length=seq_length, raw_dataset=raw_valid_dataset)

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


if __name__ == "__main__":
    config = Config()

    dataset = IndicatorDataset(dataset_name='IndicatorDataset',
                               input_path='../dataset/finance/stocks/stocks_sample.csv',
                               train_valid_ratio=0.9)
    print(dataset.train_dataset.__getitem__(14))
    print()
