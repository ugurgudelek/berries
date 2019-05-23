from generate_AR_data import ARData, fixed_ar_coefficients  # to generate AR data for easy debug
import torch
import torch.utils.data

import pandas as pd
import numpy as np

from itertools import cycle
from plotly.offline import plot
import plotly.graph_objs as go

import matplotlib.pyplot as plt

import math
import torchvision


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw

    Examples:
        train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=ImbalancedDatasetSampler(train_dataset),
        batch_size=args.batch_size,
        **kwargs
    )
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.Tensor(weights).double()

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        elif dataset_type is FinanceDataset.InnerIndicatorDataset:
            return dataset[idx][1]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class GenericDataset:

    def __init__(self):
        self.sampler = self.Sampler()

    class Normalizer:
        def __init__(self, dataset):
            self.dataset = dataset
            self.max = dataset.max()
            self.min = dataset.min()

        def denormalize(self, arr):
            return arr * (self.max - self.min) + self.max

        def normalized(self):
            return 2.*((self.dataset - self.min) / (self.max - self.min)) - 1.

    class Sampler:
        def __init__(self):
            pass

        def random_train_sample(self, n, sequence_sample=False):
            return self._random_sample(self.train_dataset, n, sequence_sample=sequence_sample)

        def random_valid_sample(self, n, sequence_sample=False):
            return self._random_sample(self.valid_dataset, n, sequence_sample=sequence_sample)

        @staticmethod
        def _random_sample(dataset, n, sequence_sample):
            perm_ixs = sorted(np.random.randint(0, dataset.__len__(), size=n))
            if sequence_sample:  # drop perm_ix and recreate them again in the order of sequence
                random_ix = np.random.randint(0, dataset.__len__() - n, size=1)[0]
                perm_ixs = np.array(range(random_ix, random_ix + n))

            X = np.array([dataset.__getitem__(perm_ix)[0] for perm_ix in perm_ixs])
            y = np.array([dataset.__getitem__(perm_ix)[1] for perm_ix in perm_ixs])
            return torch.Tensor(X).float(), torch.Tensor(y).float(), perm_ixs

        @staticmethod
        def get_data(dataset, n=None):
            if n is None:
                n = dataset.__len__()

            X, y = [], []
            for i in range(n):
                _x, _y = dataset[i]
                X.append(_x)
                y.append(_y)
            X, y = np.array(X), np.array(y)

            return X, y

    @staticmethod
    def to_categorical(y, num_classes=None, dtype='float32'):
        """Converts a class vector (integers) to binary class matrix.

        E.g. for use with categorical_crossentropy.

        # Arguments
            y: class vector to be converted into a matrix
                (integers from 0 to num_classes).
            num_classes: total number of classes.
            dtype: The data type expected by the input, as a string
                (`float32`, `float64`, `int32`...)

        # Returns
            A binary matrix representation of the input. The classes axis
            is placed last.
        """
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=dtype)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical

    @staticmethod
    def noisy_sin(steps_per_cycle=50, number_of_cycles=500, random_factor=0.4):
        '''
        random_factor    : amont of noise in sign wave. 0 = no noise
        number_of_cycles : The number of steps required for one cycle

        Return :
        pd.DataFrame() with column sin_t containing the generated sin wave
        '''
        df = pd.DataFrame(np.arange(steps_per_cycle * number_of_cycles + 1), columns=["t"])
        df["sin_t"] = df['t'].apply(
            lambda x: math.sin(x * (2 * math.pi / steps_per_cycle) + np.random.uniform(-1.0, +1.0) * random_factor))
        df["sin_t_clean"] = df['t'].apply(lambda x: math.sin(x * (2 * math.pi / steps_per_cycle)))
        print("create period-{} sin wave with {} cycles".format(steps_per_cycle, number_of_cycles))
        print("In total, the sin wave time series length is {}".format(steps_per_cycle * number_of_cycles + 1))
        return (df)

    @staticmethod
    def normalize(arr, min=None, max=None):
        min = arr.min() if min is None else min
        max = arr.max() if max is None else max
        return (arr - min) / (max - min), min, max





class FinanceDataset(GenericDataset):
    """

    """

    def __init__(self, path, train_test_ratio, window_size, stride_size, look_ahead_size, classification):
        GenericDataset.__init__(self)

        self.path = path
        self.train_test_ratio = train_test_ratio
        self.window_size = window_size
        self.stride_size = stride_size
        self.look_ahead_size = look_ahead_size
        self.classification = classification

        # self.standardizer = IndicatorDataset.IndicatorStandardizer()

        raw_dataset = pd.read_csv(self.path)
        train_len = int(raw_dataset.shape[0] * self.train_test_ratio)
        self.raw_train_dataset = raw_dataset.iloc[:train_len, :]
        self.raw_valid_dataset = raw_dataset.iloc[train_len:, :]

        self.preprocessed_train_dataset = self.preprocess_dataset(dataset=self.raw_train_dataset)
        self.preprocessed_valid_dataset = self.preprocess_dataset(dataset=self.raw_valid_dataset)

        print((f"Train ----\n"
               f"Shape: {self.preprocessed_train_dataset.shape} \n"
               f"First date: {self.preprocessed_train_dataset['date'].iloc[0]} \n"
               f"Last date: {self.preprocessed_train_dataset['date'].iloc[-1]}"))

        print((f"Valid ----\n"
               f"Shape: {self.preprocessed_valid_dataset.shape} \n"
               f"First date: {self.preprocessed_valid_dataset['date'].iloc[0]} \n"
               f"Last date: {self.preprocessed_valid_dataset['date'].iloc[-1]}"))

        self.train_dataset = FinanceDataset.InnerIndicatorDataset(dataset=self.preprocessed_train_dataset,
                                                                  window_size=self.window_size,
                                                                  stride_size=self.stride_size,
                                                                  look_ahead_size=self.look_ahead_size,
                                                                  classification=self.classification)
        self.valid_dataset = FinanceDataset.InnerIndicatorDataset(dataset=self.preprocessed_valid_dataset,
                                                                  window_size=self.window_size,
                                                                  stride_size=self.stride_size,
                                                                  look_ahead_size=self.look_ahead_size,
                                                                  classification=self.classification)

    # region ZigZag
    @staticmethod
    def is_center_max(window_data):
        return np.max(window_data) == window_data[len(window_data) // 2]

    @staticmethod
    def is_center_min(window_data):
        return np.min(window_data) == window_data[len(window_data) // 2]

    @staticmethod
    def point2label(data, on, window=15):
        """
        Finds turning points while iterating over data[on]

        Args:
            data: (pd.DataFrame) applied dataset
            on: applied column name
            window: rolling window size

        Returns: (pd.Series) labels

        """
        data = data.copy()
        data['maxs'] = data[on].rolling(window, center=True, min_periods=window).apply(
            FinanceDataset.is_center_max, raw=True)
        data['mins'] = data[on].rolling(window, center=True, min_periods=window).apply(
            FinanceDataset.is_center_min, raw=True)

        data['label'] = 'mid'
        data.loc[data['maxs'] == 1, 'label'] = 'top'
        data.loc[data['mins'] == 1, 'label'] = 'bot'

        data = data.drop(['maxs', 'mins'], axis=1)

        return data['label']

    @staticmethod
    def filter_consequtive_same_label(labels):
        """

        Args:
            labels: (pd.Series)

        Returns:

        """
        state = None
        for i, label in enumerate(labels):
            if label == 'mid':
                continue
            if state is None:
                state = label
                continue
            if state == label:
                labels[i] = 'mid'
            else:
                state = label

        return labels

    @staticmethod
    def crop_firstnonbot_and_lastnontop(labels):
        """

        Args:
            labels: (pd.Series)

        Returns:( pd.Series)

        """

        first_bot_idx = labels[(labels == 'bot')].index.values[0]
        last_top_idx = labels[(labels == 'top')].index.values[-1]

        return labels[first_bot_idx:last_top_idx + 1]

    @staticmethod
    def plot_data_zigzag_tpoints(x, data, point_labels, zigzag_distances, label, colormap=None, linestyles=None):
        """
        Plots data & zigzag data & turning points

        Args:
            data: (pd.Series)
            labels: (pd.Series)
            label: (str) label of plot
            colormap: (dict) e.g : {'mid': (0.2, 0.4, 0.6, 0), 'top': (1, 0, 0, 0.7), 'bot': (0, 1, 0, 0.7)}
            linestyle: (str) e.g : --b

        Returns: fig

        """

        buy_trace = go.Scatter(x=x[point_labels == 'bot'], y=data[point_labels == 'bot'],
                               name='buy', mode='markers', opacity=1.)
        sell_trace = go.Scatter(x=x[point_labels == 'top'], y=data[point_labels == 'top'],
                                name='sell', mode='markers', opacity=1.)
        data_trace = go.Scatter(x=x, y=data,
                                name='data', mode='lines', opacity=0.5)
        zigzag_trace = go.Scatter(x=x, y=zigzag_distances,
                                  name='zigzag', mode='lines', line=dict(dash='dash', width=2), opacity=1.)

        layout = go.Layout()
        fig = go.Figure(data=[buy_trace, sell_trace, data_trace, zigzag_trace], layout=layout)

        plot(fig, filename='zigzag.html')

        # plt.scatter(x=x, y=data, c=[colormap[plabel] for plabel in point_labels], label=None, marker='^')
        # plt.plot(x, data, linestyles['data'], alpha=0.6, label=label)
        # plt.plot(x, zigzag_distances, linestyles['zigzag'], label='zigzag', alpha=0.6)
        # plt.legend()

        # return plt

    @staticmethod
    def label2distance(data, labels):
        # point turning points then process for distance
        # after this line, stocks has 'label' column which has top-mid-bot values.

        def distance(idxs, turning_points):
            """
            Assumes turning_points start with increasing segment.
            Args:
                idxs: used for creation of dist array
                turning_points:

            Returns:

            """
            segments = np.array(list(zip(turning_points[:-1], turning_points[1:])))

            def calc_dist(data, current_ix, lower_ix, upper_ix):
                delta_x = upper_ix - lower_ix
                delta_y = data[upper_ix] - data[lower_ix]
                slope = delta_y / delta_x

                d = data[lower_ix] + slope * (current_ix - lower_ix)
                return d, slope

            dist = np.zeros_like(idxs, dtype=np.float)
            for (lower_ix, upper_ix) in segments:
                for current_ix in range(lower_ix, upper_ix):
                    dist[current_ix], slope = calc_dist(data, current_ix, lower_ix, upper_ix)

            return dist

        mid_idxs = labels[labels == 'mid'].index.values
        top_idxs = labels[labels == 'top'].index.values
        bot_idxs = labels[labels == 'bot'].index.values

        turning_points = np.sort(np.concatenate((bot_idxs, top_idxs)))
        distances = distance(labels.index.values, turning_points)

        # at this point "label" values are between 0-1 range.
        # let me add -0.5 bias
        # distances = distances - 0.5

        return distances

    @staticmethod
    def zigzag(stock, on, window):
        labels = FinanceDataset.point2label(stock, on=on, window=window)
        labels = FinanceDataset.filter_consequtive_same_label(labels=labels)
        labels = FinanceDataset.crop_firstnonbot_and_lastnontop(labels=labels)

        stock['label'] = labels
        stock = stock.dropna(axis=0).reset_index(drop=True)

        zigzag_distances = FinanceDataset.label2distance(stock[on], stock['label'])

        stock['zigzag'] = zigzag_distances

        return stock

    # endregion

    def preprocess_dataset(self, dataset):

        dataset: pd.DataFrame = dataset.loc[:, ['adjusted_close', 'date']]

        # make stationary
        dataset['adjusted_close'] = dataset['adjusted_close'].diff(periods=1)
        dataset = dataset.dropna(axis=0).reset_index(drop=True)

        dataset = FinanceDataset.zigzag(stock=dataset, on='adjusted_close', window=15)

        dataset['date'] = dataset['date'].astype('datetime64[ns]').dt.date

        # todo: plotly should add new axis type for missing dates.
        FinanceDataset.plot_data_zigzag_tpoints(x=dataset['date'].index.values,
                                                data=dataset['adjusted_close'].values,
                                                point_labels=dataset['label'].values,
                                                zigzag_distances=dataset['zigzag'].values,
                                                label='spy')
        dataset['label'] = dataset['label'].astype(pd.api.types.CategoricalDtype(['bot','mid','top'])).cat.rename_categories([0,1,2])


        # convolve over buy-sell points to reduce noise
        dataset['label_conv_0'] = np.convolve((dataset['label'].values == 0), np.array([1, 1, 1]), mode='same')
        dataset['label_conv_2'] = np.convolve((dataset['label'].values == 2), np.array([1, 1, 1]), mode='same')
        return dataset

    class InnerIndicatorDataset(torch.utils.data.Dataset):
        """

        Args:
            dataset(pd.DataFrame):
        """

        def __init__(self, dataset, window_size, stride_size, look_ahead_size, classification):


            self.window_size = window_size
            self.stride_size = stride_size
            self.look_ahead_size = look_ahead_size
            self.classification = classification

            self.dataset = dataset.iloc[:(dataset.shape[0] - dataset.shape[0] % window_size), :]  # drop last non-complete sequence
            indexes = np.arange(self.dataset.shape[0])

            sequence_start_indexes = indexes[indexes % self.stride_size == 0]
            sequence_end_indexes = sequence_start_indexes + self.window_size

            while sequence_end_indexes[-1] > indexes[-1] - look_ahead_size:  # drop overflow indexes
                sequence_start_indexes = sequence_start_indexes[:-1]
                sequence_end_indexes = sequence_end_indexes[:-1]


            X = self.dataset['adjusted_close'].values

            # regression
            y = self.dataset['zigzag'].values

            # classification
            if classification:
                y = self.dataset['label'].values
                y[y == 'bot'] = 0
                y[y == 'mid'] = 1
                y[y == 'top'] = 2

            self.X = np.array([X[start:end] for (start, end) in zip(sequence_start_indexes, sequence_end_indexes)])
            self.y = np.array([y[end-1+look_ahead_size] for (start, end) in zip(sequence_start_indexes, sequence_end_indexes)])
            print()

        def __len__(self):
            return self.X.shape[0] - self.window_size - 1

        def __getitem__(self, ix):
            # in-window normalization
            x, _min, _max = GenericDataset.normalize(self.X[ix, :])
            x = x.reshape((-1, 1)) # add input dim
            y = self.y[ix]
            if not self.classification:
                y,_,_ = GenericDataset.normalize(y, _min, _max)
            return x, y


class LongMemoryDebugDataset(GenericDataset):
    def __init__(self, window_size=20, sample_size=1000, train_test_ratio=0.8, subseq_size=1):
        GenericDataset.__init__(self)

        dataset = np.zeros((2, window_size, 1))
        dataset[0, 0, 0] = 1.

        dataset = np.repeat(dataset, repeats=sample_size // 2, axis=0)
        np.random.shuffle(dataset)

        self.train_dataset = self.Inner(dataset=dataset[:int(sample_size * train_test_ratio)], subseq_size=subseq_size)
        self.valid_dataset = self.Inner(dataset=dataset[int(sample_size * train_test_ratio):], subseq_size=subseq_size)

    class Inner(torch.utils.data.Dataset):
        def __init__(self, dataset, subseq_size):
            self.subseq_size = subseq_size
            self.x = dataset.reshape((-1, subseq_size, 1))
            self.y = np.repeat(dataset[:, 0, :], repeats=self.x.shape[0] // dataset.shape[0], axis=0)

        def __getitem__(self, ix):
            return self.x[ix, :, :], self.y[ix, 0]

        def __len__(self):
            return self.x.shape[0]

    def __str__(self):
        return (f"Dataset Name: {self.__class__}\n"
                f"Train Dataset Length: {self.train_dataset.__len__()}\n"
                f"Valid Dataset Length: {self.valid_dataset.__len__()}\n")


class StatefulTimeseriesDataset(GenericDataset):
    def __init__(self, path=None, input_col=0, dataset=None, window_size=4, train_test_ratio=0.8, squeeze=False,
                 normalize=False):
        GenericDataset.__init__(self)

        if path is None and dataset is None:
            raise Exception('path or dataset should be defined.')

        if dataset is None:  # path is valid
            dataset = pd.read_csv(path).iloc[:, input_col].values

        # drop_last_count = dataset.shape[0] % window_size  # drop some for reshape
        # dataset = dataset[:(dataset.shape[0]-drop_last_count)].reshape((-1, window_size, 1)) # batch, seq, input

        # dataset = dataset.reshape((-1, window_size, 1)) # batch, seq, input

        # self.normalizer = self.Normalizer(dataset=dataset)
        # dataset = self.normalizer.normalized()

        if normalize:
            dataset = self.normalize(dataset)

        x = dataset[:-1].reshape((-1, window_size, 1))  # batch, seq, input  # drop last
        y = dataset[1:].reshape((-1, window_size, 1))[:, -(window_size - 1), :]

        if squeeze:
            x = np.tanh(x)
            y = np.tanh(y)

        self.train_dataset = self.Inner(x=x[:int(x.shape[0] * train_test_ratio), :, :],
                                        y=y[:int(y.shape[0] * train_test_ratio), :])
        self.valid_dataset = self.Inner(x=x[int(x.shape[0] * train_test_ratio):, :, :],
                                        y=y[int(y.shape[0] * train_test_ratio):, :])

    class Inner(torch.utils.data.Dataset):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __getitem__(self, ix):
            return self.x[ix, :, :], self.y[ix, 0]

        def __len__(self):
            return self.x.shape[0]


class StatelessTimeseriesDataset(GenericDataset):
    def __init__(self, path=None, input_col=0, dataset=None, window_size=4, train_test_ratio=0.8):
        GenericDataset.__init__(self)

        if path is None and dataset is None:
            raise Exception('path or dataset should be defined.')

        if dataset is None:  # path is valid
            dataset = pd.read_csv(path).iloc[:, input_col].values

        dataset = dataset.reshape((-1, 1))  # add input dim
        # self.normalizer = self.Normalizer(dataset=dataset)
        # dataset = self.normalizer.normalized()

        self.train_dataset = self.Inner(dataset=dataset[:int(dataset.shape[0] * train_test_ratio), :],
                                        window_size=window_size)
        self.valid_dataset = self.Inner(dataset=dataset[int(dataset.shape[0] * train_test_ratio):, :],
                                        window_size=window_size)

    class Inner(torch.utils.data.Dataset):
        def __init__(self, dataset, window_size):
            self.dataset = dataset
            self.window_size = window_size

        def __getitem__(self, ix):
            return self.dataset[ix:ix + self.window_size, :], self.dataset[ix + self.window_size, :]

        def __len__(self):
            return self.dataset.shape[0] - self.window_size


class SequenceLearningManyToOneRotate(GenericDataset):
    def __init__(self, seq_len=3, seq_limit=11, dataset_len=100, onehot=False):
        self.seq_len = seq_len
        self.seq_limit = seq_limit
        self.dataset_len = dataset_len

        # todo : add seq_range
        sequence = cycle(np.arange(0, seq_limit, 1))
        seq7s = list()
        while (True):
            seq7 = list()

            for i in range(seq_len):
                digit = next(sequence)
                seq7.append(digit)
            seq7s.append(seq7)

            if len(seq7s) == dataset_len:
                break

        dataset = np.array(seq7s)
        np.random.shuffle(dataset)

        # data_size, seq_len, input_size

        self.train_dataset = self.Inner(dataset=dataset[:int(dataset.shape[0] * 0.9)], seq_len=seq_len,
                                        seq_limit=seq_limit)
        self.valid_dataset = self.Inner(dataset=dataset[int(dataset.shape[0] * 0.9):], seq_len=seq_len,
                                        seq_limit=seq_limit)

    def denormalize(self, x):
        return (x * self.seq_limit)

    class Inner(torch.utils.data.Dataset, GenericDataset):
        def __init__(self, dataset, seq_len, seq_limit):
            # self.encode = OneHotEncoder(sparse=False)
            # X = self.encode.fit_transform(X)
            # self.y = self.encode.transform(self.y)

            # norm_dataset = (dataset - dataset.min())/(dataset.max() - dataset.min())
            # self.dataset = norm_dataset

            X = dataset[:, :].astype(np.float32)
            y = (dataset[:, -1] + 1).__mod__(seq_limit).astype(np.float32)

            # seq_len, dataset_len, input_size
            # X = to_categorical(X, seq_limit).transpose([1,0,2])

            X = np.expand_dims(X.T, axis=2)
            y = y.T
            self.data = torch.FloatTensor(X) / seq_limit
            self.labels = torch.FloatTensor(y) / seq_limit

        def __len__(self):
            return self.data.shape[1]

        def __getitem__(self, ix):
            return self.data[:, ix, :], self.labels[ix]


class SequenceLearningManyToOne(GenericDataset):
    def __init__(self, seq_len=7, seq_limit=1000):
        self.seq_len = seq_len
        self.seq_limit = seq_limit

        # todo : add seq_range
        sequences = []
        for ix in range(0, seq_limit - seq_len):
            sequences += [list(range(ix, ix + seq_len))]

        dataset = np.array(sequences)
        np.random.shuffle(dataset)

        # data_size, seq_len, input_size

        self.train_dataset = self.Inner(dataset=dataset[:int(dataset.shape[0] * 0.9)], seq_len=seq_len,
                                        seq_limit=seq_limit)
        self.valid_dataset = self.Inner(dataset=dataset[int(dataset.shape[0] * 0.9):], seq_len=seq_len,
                                        seq_limit=seq_limit)

    def denormalize(self, x):
        return x * self.seq_limit

    class Inner(torch.utils.data.Dataset, GenericDataset):
        def __init__(self, dataset, seq_len, seq_limit):
            # self.encode = OneHotEncoder(sparse=False)
            # X = self.encode.fit_transform(X)
            # self.y = self.encode.transform(self.y)

            # norm_dataset = (dataset - dataset.min())/(dataset.max() - dataset.min())
            # self.dataset = norm_dataset

            X = dataset[:, :].astype(np.float32)
            y = (dataset[:, -1] + 1).astype(np.float32)

            # seq_len, dataset_len, input_size
            # X = to_categorical(X, seq_limit).transpose([1,0,2])

            X = np.expand_dims(X.T, axis=2)
            y = y.T
            self.data = torch.FloatTensor(X) / seq_limit
            self.labels = torch.FloatTensor(y) / seq_limit

        def __len__(self):
            return self.data.shape[1]

        def __getitem__(self, ix):
            return self.data[:, ix, :], self.labels[ix]


class TimeSeriesARDataset():

    def __init__(self, num_datapoints=100, test_size=0.2, num_prev=5):
        if num_prev not in [2, 5, 10, 20]:
            raise Exception(f"num_prev should be in [2, 5, 10, 20] but you gave {num_prev}")
        dataset = ARData(num_datapoints, coeffs=fixed_ar_coefficients[num_prev],
                         test_size=test_size, num_prev=num_prev)

        self.train_dataset = TimeSeriesARDataset.InnerTimeSeriesArDataset(X=dataset.X_train, y=dataset.y_train)
        self.valid_dataset = TimeSeriesARDataset.InnerTimeSeriesArDataset(X=dataset.X_test, y=dataset.y_test)

    class InnerTimeSeriesArDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            # expand  (batch, seq) -> (seq, batch, input)

            self.X = X.T
            self.X = np.expand_dims(self.X, axis=2)

            # (batch, input, seq ) vs (seq, batch, input)
            np.testing.assert_equal(np.expand_dims(X, axis=1)[1, :, :].flatten(),
                                    self.X[:, 1, :].flatten())

            self.y = y

        def __getitem__(self, ix):
            return (torch.from_numpy(self.X[:, ix, :]).float(),
                    torch.from_numpy(np.array(self.y[ix])).float())

        def __len__(self):
            return self.X.shape[1]


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

        self.preprocessed_train_dataset = self.preprocess_dataset(dataset=self.raw_train_dataset, kind='train',
                                                                  label_type=self.label_type)
        self.preprocessed_valid_dataset = self.preprocess_dataset(dataset=self.raw_valid_dataset, kind='validation',
                                                                  label_type=self.label_type)

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

        self.train_dataset = IndicatorDataset.InnerIndicatorDataset(dataset=self.preprocessed_train_dataset,
                                                                    seq_len=self.seq_len, problem_type=self.label_type)
        self.valid_dataset = IndicatorDataset.InnerIndicatorDataset(dataset=self.preprocessed_valid_dataset,
                                                                    seq_len=self.seq_len, problem_type=self.label_type)
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
        dataset = self.differentiate(dataset, subset=['open', 'high', 'low', 'close',
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
        return np.max(window_data) == window_data[len(window_data) // 2]

    @staticmethod
    def is_center_min(window_data):
        return np.min(window_data) == window_data[len(window_data) // 2]

    @staticmethod
    def plot_top_bot_turning_point(stock_data):
        close = stock_data['adjusted_close'].values
        labels = stock_data['label'].values
        x = range(len(close))

        # (r,g,b,a)
        colormap = {'mid': (0.2, 0.4, 0.6, 0), 'top': (1, 0, 0, 0.7), 'bot': (0, 0, 1, 0.7)}
        colors = [colormap[label] for label in labels]

        plt.scatter(x=x, y=close, c=colors)
        plt.plot(x, close, lw=1, label='close')
        plt.legend()

    def label_top_bot_mid(self, stocks, window=7):

        def inner_func(stock_data):
            stock_data['maxs'] = stock_data['adjusted_close'].rolling(window, center=True, min_periods=window).apply(
                IndicatorDataset.is_center_max)
            stock_data['mins'] = stock_data['adjusted_close'].rolling(window, center=True, min_periods=window).apply(
                IndicatorDataset.is_center_min)

            stock_data['label'] = 'mid'
            stock_data.loc[stock_data['maxs'] == 1, 'label'] = 'top'
            stock_data.loc[stock_data['mins'] == 1, 'label'] = 'bot'

            stock_data = stock_data.drop(['maxs', 'mins'], axis=1)

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
                return (x - lower) / (upper - lower)

            state = True
            dist = np.zeros_like(idxs, dtype=np.float)
            for (lower, upper) in segments:
                for i in range(lower, upper):
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
            for i, row in stock_data.iterrows():
                if row['label'] == 'mid':
                    continue
                if state is None:
                    state = row['label']
                    continue
                if state == row['label']:
                    stock_data.loc[i, 'label'] = 'mid'
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
            (self.raw_train_dataset['name'] == name) &
            (self.raw_train_dataset['date'] == date)]

    def get_data_seq(self, name, first_date, last_date):
        return self.raw_train_dataset.loc[
            (self.raw_train_dataset['name'] == name) &
            (self.raw_train_dataset['date'] >= first_date) &
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
            stock_data['label2'] = np.convolve((stock_data['label'] == 'top').values, np.ones(window), 'same')
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
            data_subset = data_subset.apply(
                lambda x: self.standardizer.apply_standardization(x, kind=kind, stock_name=stock_name), axis=0)
            return pd.concat((data_neg_subset, data_subset), axis=1)

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

        self.train_dataset = LoadDataset.InnerLoadDataset(train_values, seq_length=seq_length,
                                                          raw_dataset=raw_train_dataset)
        self.valid_dataset = LoadDataset.InnerLoadDataset(valid_values, seq_length=seq_length,
                                                          raw_dataset=raw_valid_dataset)

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


if __name__ == '__main__':
    finance_dataset = FinanceDataset(path='../input/spy.csv',
                                     train_test_ratio=0.8,
                                     window_size=20)
    print()
