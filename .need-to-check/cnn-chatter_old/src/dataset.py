__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

from torch.utils.data import Dataset
import scipy
from collections import defaultdict, OrderedDict
import random
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from nptdms import TdmsFile

import matplotlib.pyplot as plt
import plotly.tools as tls
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio
import seaborn as sns
import os
from scipy import interpolate, signal

import warnings
from tqdm import tqdm
import pickle
from sklearn.preprocessing import OneHotEncoder

import multiprocessing
from functools import reduce

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])


class DatasetNumpyWrapper(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, ix):
        return self.X[ix].numpy(), self.y[ix].item()

    def __len__(self):
        return self.y.__len__()


class VibrationDataset():
    # PATH = Path('../input/preprocessed_data/alu_v1_verification')

    def __init__(self, path, train_ratio=0.75, kind='acoustic', shuffle_mode=0):
        # shuffle_mode = 0:no_shuffle, 1:standard_shuffle, 2:shuffle_in_channel

        self.label_types = {'chatter yok': 0, 'chatter var': 1}

        def read_data(cutpath):
            img_paths = defaultdict(dict)  # kanal10 -> {0:path0, 1:path1 ...}
            for path in cutpath.glob(f'**/spectrogram_{kind}.csv'):
                if path.parent.name != '00':  # do not read a folder which have name '00' cuz full data not necessary
                    img_paths[path.parent.parent.name][path.parent.name] = path

            if shuffle_mode == 2 or shuffle_mode == 1:
                cut_keys = list(img_paths.keys())
                random.shuffle(cut_keys)
                img_paths = {k: img_paths[k] for k in cut_keys}

            if shuffle_mode == 1:
                cut_keys = list(img_paths.keys())
                for k in cut_keys:
                    channel_keys = list(img_paths[k].keys())
                    random.shuffle(channel_keys)
                    img_paths[k] = {c: img_paths[k][c] for c in channel_keys}

            label_excel = pd.read_excel(cutpath / 'labels_acc_batihan.xlsx', index_col=0)
            images = defaultdict(dict)
            labels = defaultdict(dict)
            data = defaultdict(dict)
            for kanalname, cut_dict in tqdm(img_paths.items()):
                for time, img_path in cut_dict.items():
                    img = Image.fromarray(pd.read_csv(img_path)[::-1].values.astype('uint8'), 'L')
                    label = label_excel.loc[int(time), kanalname]
                    if label in self.label_types.keys():
                        images[kanalname][time] = img
                        labels[kanalname][time] = self.label_types[label]
                        data[kanalname][time] = (img_transform(img), self.label_types[label])

            # labels = {f'kanal{ix+1}': (1 if val == 'severe' else 0 if val == 'yok' else -1) for ix, val in
            #                enumerate(pd.read_csv(cutpath / 'labels.csv', header=None)[1].values)}
            r_data = list()
            for kanalname, cut_dict in data.items():
                r_data += list(cut_dict.values())
            return r_data

        self.path = path


        self.data = read_data(self.path)

        train_size = int(self.data.__len__() * train_ratio)

        self.dataset = self.VibrationInnerDataset(self.data)
        self.train_dataset = self.VibrationInnerDataset(self.data[:train_size])
        self.test_dataset = self.VibrationInnerDataset(self.data[train_size:])

        print((f"Dataset Statistics:\n",
               f"All:{self.dataset.describe()}\n",
               f"Train:{self.train_dataset.describe()}\n",
               f"Test :{self.test_dataset.describe()}\n"))

    @staticmethod
    def make_inner_dataset(data):
        return VibrationDataset.VibrationInnerDataset(data)

    class VibrationInnerDataset(Dataset):

        def __init__(self, data):
            self.data = data

            self.features = [X for X, y in self.data]
            self.labels = [y for X, y in self.data]

        def __getitem__(self, ix):
            return self.data[ix]

        def __len__(self):
            return self.data.__len__()

        def describe(self):
            return (f"Dataset Lenght:{self.__len__()}",
                    f"Label_mean:{np.array(self.labels).mean()}")


class TimeSeriesNode:
    def __init__(self, **kwargs):
        self.feature_names = list()
        self.experiment_param_names = list()
        self.data_chunks = defaultdict(OrderedDict)

        for kw, arg in kwargs.items():
            self.__dict__[kw] = arg
            if 'slotname' == kw:
                self.__dict__['slotno'] = int(arg.split('kanal')[-1])

    def add_features(self, chunk_id, features_dict):
        for kw, arg in features_dict.items():
            self.data_chunks[chunk_id][kw] = arg
            if kw not in self.feature_names:
                self.feature_names.append(kw)

    def add_experiment_params(self, params_dict):
        for kw, arg in params_dict.items():
            self.__dict__[kw] = arg
            self.experiment_param_names.append(kw)

    # @property
    # def features(self):
    #     return {name: self.__dict__[name] for name in self.feature_names}

    @property
    def experiment_params(self):
        return {name: self.__dict__[name] for name in self.experiment_param_names}

    # @property
    # def hover_information(self):
    #     return ''.join((
    #         f"Sample {1 if self.slotno % 2 == 1 else 2}<br />",
    #         "-----------<br />",
    #         "Params<br />",
    #         "-------<br />",
    #         '<br />'.join([f'{k}: {v}' for k, v in self.experiment_params.items()]),
    #         "Features<br />",
    #         "-------<br />",
    #         '<br />'.join([f'{k}: {v}' for k, v in self.features.items()])))


class TimeSeriesDataset():
    LABEL_TYPES = {'no chatter': 0, 'medium chatter': 1, 'high chatter': 2,
                   'iptal': 3}

    FEATURES = [
        # 'var',
        'skew',
        # 'kurtosis',
        # 'vpeak',
        'rms',
        'clearance_factor',
        'crest_factor',
        'shape_factor',
        # 'impulse_factor',
        # 'index_of_dispersion',
    ]

    EXPERIMENT_PARAMS = ['kesim_paramsexcel',
                         'n_flutes',
                         'feed_per_tooth',
                         'depth_of_cut',
                         'spindle_speed',
                         'feed_rate',
                         'cut_interval', ]

    # SHUFFLE_MODE = 0

    def __init__(self, path, train_ratio=0.75, shuffle=True):
        self.path = Path(path)
        self.train_ratio = train_ratio
        self.dataframe = pd.read_csv(self.path)

        # self.dataframe = self.dataframe.loc[self.dataframe['label'] != 'iptal']

        # Label Binarizer
        self.dataframe['label'] = self.dataframe['label'].apply(lambda x: self.LABEL_TYPES[x])

        if shuffle:
            # Shuffle while protecting slotname order
            d = {slotname: group for slotname, group in self.dataframe.groupby(by='slotname')}
            perm = np.random.permutation(list(d.keys()))
            temp_df = pd.DataFrame(columns=self.dataframe.columns)
            for key in perm:
                temp_df = temp_df.append(d[key])

            self.dataframe = temp_df.reset_index(drop=True)
        # self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

        # Split dataset into train and test set while protecting slotname order
        train_size = int(self.dataframe['slotname'].nunique() * self.train_ratio)
        self.dataset = self.TimeSeriesInnerDataset(self.dataframe, self.FEATURES)
        self.train_dataset = self.TimeSeriesInnerDataset(
            self.dataframe.loc[self.dataframe['slotname'].isin(self.dataframe['slotname'].unique()[:train_size])],
            self.FEATURES)
        self.test_dataset = self.TimeSeriesInnerDataset(
            self.dataframe.loc[self.dataframe['slotname'].isin(self.dataframe['slotname'].unique()[train_size:])],
            self.FEATURES)

        print('\n'.join(
            (f"Dataset Statistics:",
            f"All:{self.dataset.describe()}",
            f"Train:{self.train_dataset.describe()}",
            f"Test :{self.test_dataset.describe()}")
            ))

    @staticmethod
    def make_inner_dataset(dataframe):
        return TimeSeriesDataset.TimeSeriesInnerDataset(dataframe,
                                                        TimeSeriesDataset.FEATURES)

    @staticmethod
    def get_params_from_experiment(experiment_params_excel, slotname):

        slotno = int(slotname.split('kanal')[-1])
        row = experiment_params_excel.loc[experiment_params_excel.iloc[:, 0] == slotno].iloc[0, :8]
        return {'kesim_paramsexcel': row['Kesim'],
                'n_flutes': row['Number of flutes'],
                'feed_per_tooth': row['Feed per tooth (mm/tooth)'],
                'depth_of_cut': row['Depth of Cut (mm)'],
                'spindle_speed': row['RPM'],
                'feed_rate': row['Feed rate (mm/min)'],
                'cut_interval': row['cut-interval'],
                }, row['Label']  # label

    @classmethod
    def from_readings(cls, reading_path, shuffle=True, kind='acc', train_ratio=0.75):

        reading_path = Path(reading_path)
        experiment_params_excel = pd.read_excel(reading_path / f'alu_v2_parameters_{kind}.xlsx', skiprows=1)


        def read_data(cutpath):
            data_paths = defaultdict(dict)  # kanal10 -> {0:path0, 1:path1 ...}
            for path in cutpath.glob(f'**/data.csv'):
                if path.parent.name == '00':  # read only full data
                    data_paths[path.parent.parent.name][path.parent.name] = path

                    # if shuffle_mode == 2 or shuffle_mode == 1:
                    #     cut_keys = list(data_paths.keys())
                    #     random.shuffle(cut_keys)
                    #     data_paths = {k: data_paths[k] for k in cut_keys}
                    #
                    # if shuffle_mode == 1:
                    #     cut_keys = list(data_paths.keys())
                    #     for k in cut_keys:
                    #         channel_keys = list(data_paths[k].keys())
                    #         random.shuffle(channel_keys)
                    #         data_paths[k] = {c: data_paths[k][c] for c in channel_keys}

            data = list()
            for kanalname, cut_dict in tqdm(data_paths.items()):
                for time, data_path in cut_dict.items():
                    img = pd.read_csv(data_path, index_col=0)[kind].values

                    experiment_params, label = cls.get_params_from_experiment(experiment_params_excel, kanalname)

                    node = TimeSeriesNode(slotname=kanalname,
                                          start_time=int(experiment_params["cut_interval"].split(' ')[0].split('-')[0]),
                                          raw_data=img,
                                          label=label)
                    node.add_experiment_params(experiment_params)
                    data.append(node)

                    # labels = {f'kanal{ix+1}': (1 if val == 'severe' else 0 if val == 'yok' else -1) for ix, val in
                    #                enumerate(pd.read_csv(cutpath / 'labels.csv', header=None)[1].values)}
                    # r_data = list()
                    # for kanalname, cut_dict in data.items():
                    #     r_data += list(cut_dict.values())
                    #
                    #
                    # return np.array([X for X, y in r_data]), np.array([y for X, y in r_data])

            return data

        readings: list = read_data(reading_path)

        for datanode in readings:  # type:TimeSeriesNode
            # Data Cleansing
            datanode.cleansed_data = cls.cleanse(datanode.raw_data)

            # Data augmentation
            for chunk_id, datachunk in cls.augment(x=datanode.cleansed_data, stride=0.1, length=1.,
                                                   sampling_rate=12800):
                # Calculate features and add to datanode for each chunk
                datanode.add_features(chunk_id, cls._feature_dict(datachunk))

        def create_dataframe(condition):
            """

            :param condition: takes datanode as input and returns True or False
            :param save_name:
            :return:
            """
            dataframe = pd.DataFrame(data=[chunk_data
                                           for datanode in readings for chunk_id, chunk_data in
                                           datanode.data_chunks.items()
                                           if condition(datanode)])
            dataframe['label'] = [datanode.label for datanode in readings for chunk_id, chunk_data in
                                  datanode.data_chunks.items() if condition(datanode)]
            dataframe['slotname'] = [datanode.slotname for datanode in readings for chunk_id, chunk_data in
                                     datanode.data_chunks.items() if condition(datanode)]
            dataframe['time'] = [datanode.start_time+chunk_id for datanode in readings for chunk_id, chunk_data in
                                 datanode.data_chunks.items() if condition(datanode)]
            dataframe['kesim_paramsexcel'] = [datanode.kesim_paramsexcel for datanode in readings for
                                              chunk_id, chunk_data in datanode.data_chunks.items() if
                                              condition(datanode)]
            dataframe['n_flutes'] = [datanode.n_flutes for datanode in readings for chunk_id, chunk_data in
                                     datanode.data_chunks.items() if condition(datanode)]
            dataframe['feed_per_tooth'] = [datanode.feed_per_tooth for datanode in readings for chunk_id, chunk_data in
                                           datanode.data_chunks.items() if condition(datanode)]
            dataframe['depth_of_cut'] = [datanode.depth_of_cut for datanode in readings for chunk_id, chunk_data in
                                         datanode.data_chunks.items() if condition(datanode)]
            dataframe['spindle_speed'] = [datanode.spindle_speed for datanode in readings for chunk_id, chunk_data in
                                          datanode.data_chunks.items() if condition(datanode)]
            dataframe['feed_rate'] = [datanode.feed_rate for datanode in readings for chunk_id, chunk_data in
                                      datanode.data_chunks.items() if condition(datanode)]

            return dataframe

        # Create a dataframe and drop time 00 cuz full data not necessary
        dataframe = create_dataframe(condition=lambda datanode: datanode.label != 'iptal')
        dataframe.to_excel(reading_path / f'data_{kind}.xlsx', index=False)
        dataframe.to_csv(reading_path / f'data_{kind}.csv', index=False)
        # # Create a dash-dataframe : get only '00' data
        # dash_dataframe = create_dataframe(condition=lambda datanode: datanode.time == '00')
        # dash_dataframe.to_excel(reading_path / f'dash_data_{kind}.xlsx', index=False)
        # dash_dataframe.to_csv(reading_path / f'dash_data_{kind}.csv', index=False)

        return cls(path=reading_path / f'data_{kind}.csv', train_ratio=train_ratio, shuffle=shuffle)

    @staticmethod
    def augment(x, stride, length, sampling_rate):
        stride = int(stride * sampling_rate)
        length = int(length * sampling_rate)
        for id, start in enumerate(range(0, x.shape[0], stride)):  # 128001
            if start + length > x.shape[0]:
                return
            yield id*(stride/sampling_rate), x[start:start + length]  # index, data_chunk

    @staticmethod
    def feature_importance(feature_names, train_dataset, test_dataset):
        # clf = DecisionTreeClassifier()
        clf = RandomForestClassifier(n_estimators=100)

        train_labels = train_dataset.labels
        test_labels = test_dataset.labels

        train_labels = OneHotEncoder(3).fit_transform(train_labels.reshape((-1, 1))).toarray()
        test_labels = OneHotEncoder(3).fit_transform(test_labels.reshape((-1, 1))).toarray()

        clf.fit(train_dataset.features, train_labels)
        print(f"Train score:{clf.score(train_dataset.features, train_labels)}")
        print(f"Test score:{clf.score(test_dataset.features, test_labels)}")
        print(f"Feature Importances: {clf.feature_importances_}")

        fig = go.Figure(data=[go.Bar(x=feature_names, y=clf.feature_importances_)])
        py.plot(fig)

    @staticmethod
    def feature_heatmap(feature_names, train_dataset):

        plt.figure()
        df = pd.DataFrame(train_dataset.features, columns=feature_names)
        df['label'] = train_dataset.labels
        ax = sns.heatmap(df.corr(), annot=True, fmt='.1f')

        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0] + 0.5, ylim[1] - 0.5)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def cleanse(x):
        return x

    @staticmethod
    def _feature_dict(x):
        return OrderedDict(
            var=TimeSeriesDataset.variance(x),
            skew=TimeSeriesDataset.skewness(x),
            kurtosis=TimeSeriesDataset.kurtosis(x),
            vpeak=TimeSeriesDataset.vpeak(x),
            rms=TimeSeriesDataset.rms(x),
            clearance_factor=TimeSeriesDataset.clearance_factor(x),
            crest_factor=TimeSeriesDataset.crest_factor(x),
            shape_factor=TimeSeriesDataset.shape_factor(x),
            impulse_factor=TimeSeriesDataset.impulse_factor(x),
            index_of_dispersion=TimeSeriesDataset.index_of_dispersion(x)
        )

    # Chatter detection in milling machines by neural network classification and feature selection
    # Mourad Lamraoui1, Mustapha Barakat, Marc Thomas1 and Mohamed El Badaoui
    @staticmethod
    def variance(x):
        return np.var(x)

    @staticmethod
    def skewness(x):
        return scipy.stats.skew(x)

    @staticmethod
    def kurtosis(x):
        return scipy.stats.kurtosis(x)

    @staticmethod
    def vpeak(x):
        return np.max(np.abs(x))

    @staticmethod
    def rms(x):
        return np.sqrt(np.mean(np.square(x)))

    @staticmethod
    def clearance_factor(x):
        return TimeSeriesDataset.vpeak(x) / np.square(np.mean(np.abs(x)))

    @staticmethod
    def crest_factor(x):
        return TimeSeriesDataset.vpeak(x) / TimeSeriesDataset.rms(x)

    @staticmethod
    def shape_factor(x):
        return TimeSeriesDataset.rms(x) / np.mean(np.abs(x))

    @staticmethod
    def impulse_factor(x):
        return TimeSeriesDataset.vpeak(x) / np.mean(np.abs(x))

    @staticmethod
    def mean(x):
        return np.mean(x)

    @staticmethod
    def index_of_dispersion(x):  # variance-to-mean-ratio
        return TimeSeriesDataset.variance(x) / TimeSeriesDataset.mean(x)

    class TimeSeriesInnerDataset(Dataset):

        def normalize(self, features):
            return (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0))

        @staticmethod
        def standardize(features):
            return (features - np.mean(features, axis=0)) / np.std(features, axis=0)

        def __init__(self, dataframe, feature_names):
            self.dataframe = dataframe
            self.features = dataframe.loc[:, feature_names].values
            self.labels = dataframe['label'].values

            self.features = self.standardize(features=self.features)

        def __getitem__(self, ix):
            return self.features[ix, :], self.labels[ix]

        def __len__(self):
            return self.features.shape[0]

        def describe(self):
            if self.__len__() != 0:
                return (f"Dataset Lenght:{self.__len__()}",
                        f"Label_mean:{self.labels.mean()}",
                        {s: np.sum((self.labels == s)) for s in set(self.labels)})
            return "Empty"

        def to_dataframe(self):
            pass



class WearData():      
    
    CUTNO = 4
    HTML_OUTPUT = True
    WARN = False




    BASE_DATAPATH = Path(f'../input/raw_data/304_wear/tool{CUTNO}')
    BASE_OUTPUTPATH = Path(f'../input/preprocessed_data/304_wear/tool{CUTNO}')
    BASE_SAMPLING_FREQ = 12800
    LABELPATH = Path('../input/raw_data/304_wear/Experimental Data Sheet.xlsx')

    @staticmethod
    def read_and_merge():
        with multiprocessing.Pool(processes=4) as pool:
            wear_data_container = pool.map(WearData.worker, list(WearData.BASE_DATAPATH.iterdir()))
        wear_data_container = sorted(wear_data_container, key=lambda cls: cls.measure_len)
        wear_data = reduce(lambda a,b:a+b, wear_data_container)
        wear_data.interpolate_cut_len()
        wear_data.interpolate_wear_len()
        # round all float precision to 5
        wear_data.tdms_data = wear_data.tdms_data.round(5)
        return wear_data


    def __init__(self, fpath=None):
        self.fpath = fpath
        self.fft_ylim = {'acc':0.0042, 'acoustic':0.05}
        
        # update after read()
        self.measure_len = None # read
        self.outputpath = None # read
        self.tdms_data = pd.DataFrame() # read
        self.label_df = pd.DataFrame() # read_labels -> read
        self.interp_f = None # interpolate_labels
        self.sampling_freq = self.BASE_SAMPLING_FREQ

    @staticmethod    
    def worker(fpath):
        with open('log.log', 'a') as log:
            log.write(f'Worker for "{fpath.name}" has been started.\n')
        return WearData(fpath=fpath).read()

    def read(self):
        if self.fpath is not None:
            self.measure_len = int(self.fpath.name.split('_')[-1])
            self.outputpath = self.BASE_OUTPUTPATH / self.fpath.name
        
            tdms_acc:pd.DataFrame = TdmsFile(self.fpath/"ivme1.tdms").as_dataframe().rename(columns={"/'IVME'/'Untitled'":"acc"})
            tdms_sound:pd.DataFrame = TdmsFile(self.fpath/"ses1.tdms").as_dataframe().rename(columns={"/'SES'/'Untitled'":"acoustic"})
        
            self.tdms_data = pd.DataFrame(data={'time':np.arange(tdms_acc.shape[0])*(1/self.BASE_SAMPLING_FREQ),
                                'acc':tdms_acc['acc'],
                              'acoustic':tdms_sound['acoustic'],
                              })
        
            self.tdms_data.index = self.tdms_data['time'].apply(lambda t:f'{int(t//60)}:{(t-60*(t//60)):.6f}')
            self.tdms_data.index = pd.to_datetime(self.tdms_data.index, format='%M:%S.%f')
            self.tdms_data = self.tdms_data.resample('L').mean() # changes sampling freq to 1000 
            self.sampling_freq = 1000

            self.read_labels() # read labels to assign into our data
            self.tdms_data['cut_len'] = np.nan
            with open('log.log', 'a') as f:
                f.write(f"{self.label_df.loc[self.label_df['cut_len'] == self.measure_len]}\n")
            self.tdms_data.loc[:, 'cut_len'].iloc[-1] = self.label_df.loc[self.label_df['cut_len'] == self.measure_len, 'cut_len'].item()

            return self  

    def read_labels(self):
        self.label_df = pd.read_excel(self.LABELPATH, sheet_name=f'Kesim{self.CUTNO}').loc[:, ['cutting length (metre)', 'Ortalama (µm)']]
        self.label_df = self.label_df.rename(columns={'cutting length (metre)':'cut_len',
                                        'Ortalama (µm)':'wear_len'})
        self.label_df['cut_len'] = self.label_df['cut_len']*1000 # meter to milimeter

    def interpolate_cut_len(self):
        """ Call this function after total merge!
        """
        y = self.tdms_data['cut_len'].dropna()
        x = self.tdms_data.loc[y.index, 'time']

        y = y.values
        x = x.values

        self.interp_cut_len = interpolate.interp1d(x,y, fill_value="extrapolate")
        # Interpolate
        new_x = self.tdms_data['time'].values
        self.tdms_data['raw_cut_len'] = self.tdms_data['cut_len'].copy()
        self.tdms_data['cut_len'] = self.interp_cut_len(new_x)

    def interpolate_wear_len(self):
        """ Call this function after total merge 
        and interpolate_cut_len!
        """
        x = self.label_df['cut_len'].values 
        y = self.label_df['wear_len'].values
        self.interp_wear_len = interpolate.interp1d(x,y, fill_value="extrapolate")

        # Interpolate
        new_x = self.tdms_data['cut_len'].values  # should be filled already
        self.tdms_data['wear_len'] = self.interp_wear_len(new_x)

    def plot_timeseries(self, data, whichone='acc', fig=None, save=True, plot=True):
                
        trace = go.Scatter(x=data['time'], y=data[whichone], name=whichone)

        data = [trace]
        layout = go.Layout(
            xaxis={'title': 'Time(sec)'},
            yaxis={'title': 'Amplitude'}
        )

        fig = go.Figure(data=data, layout=layout)

        if save:
            pio.write_image(fig, os.path.join(self.outputpath,f"plot_{whichone}.png"))
            if self.HTML_OUTPUT:
                pio.write_html(fig, os.path.join(self.outputpath,f"plot_{whichone}.html"))
        if plot:
            py.iplot(fig)
    
    @staticmethod
    def fft(data, whichone, sampling_freq):
        Fs = sampling_freq # sampling freq
        Ts = 1/Fs # sampling interval
        t = data['time'] # time vector

        n = data.shape[0] # data lenght
        k = np.arange(n)
        T = n/Fs
        freq = k/T

        S = data[whichone].values
        S_fft = np.fft.fft(S)


        freq = freq[:n//2] # one side frequency range
        freq = freq[freq<=2500] # cut unnecessary frequencies
        
        S_fft = S_fft.real[:freq.shape[0]] / n # fft computing and normalization
        S_fft = np.abs(S_fft) # y-axis symetric correction
        
        fft_data = S_fft
        fft_freqs = freq
        return fft_data, fft_freqs
         
    def plot_fft(self,  data, whichone='acc', fig=None, save=True, plot=True):
        
        
        fft_data, fft_freqs = self.fft(data=data, whichone=whichone, sampling_freq=self.sampling_freq)

        if (fft_data.max() > self.fft_ylim[whichone]) and self.WARN:
            warnings.warn(f"fft_ylim should be increased to {fft_data.max()} from {self.fft_ylim[whichone]}")

        trace = go.Scatter(x=fft_freqs, y=fft_data, name='fft')

        data = [trace]
        layout = go.Layout(
            xaxis={'title': 'Freq(Hz)'},
            yaxis={'title': 'Amplitude', 'range':[0, self.fft_ylim[whichone]]}
        )

        fig = go.Figure(data=data, layout=layout)

        if save:
            pio.write_image(fig, os.path.join(self.outputpath,f"plot_fft_{whichone}.png"))
            if self.HTML_OUTPUT:
                pio.write_html(fig, os.path.join(self.outputpath,f"plot_fft_{whichone}.html"))
        if plot:
            py.iplot(fig)
    
    @staticmethod
    def spectrogram(data, whichone, sampling_freq):
       
        S = data[whichone]
        spect_freqs, spect_times, spectrogram_data = signal.spectrogram(S, sampling_freq)
        spect_freqs = spect_freqs[spect_freqs<=2500]  # cut unnecessary frequencies
        spect_times = spect_times
        spectrogram_data = spectrogram_data[:spect_freqs.shape[0], :]
        
        return spect_freqs, spect_times, spectrogram_data

    def plot_spectrogram(self, data, whichone='acc', fig=None, save=True, plot=True, csv=False):
        
  
        spect_freqs, spect_times, spectrogram_data = self.spectrogram(data=data, whichone=whichone, sampling_freq=self.sampling_freq)
        trace = go.Heatmap(z=10*np.log10(spectrogram_data), x=spect_times, y=spect_freqs, name='spectrogram')

        data = [trace]
        layout = go.Layout(
            xaxis={'title': 'Time(sec)'},
            yaxis={'title': 'Freq(Hz)'})

        fig = go.Figure(data=data, layout=layout)

        if save:
            pio.write_image(fig, os.path.join(self.outputpath,f"plot_spectrogram_{whichone}.png"))
            if self.HTML_OUTPUT:
                pio.write_html(fig, os.path.join(self.outputpath,f"plot_spectrogram_{whichone}.html"))
            if csv:
                np.savetxt(os.path.join(self.outputpath,f"spectrogram_{whichone}.csv"),
                           fig['data'][0]['z'], delimiter=',')

        if plot:
            py.iplot(fig)
        
        
    def save(self):
        
        os.makedirs(self.outputpath, exist_ok=True)
        for whichone in ['acc', 'acoustic']:
            self.plot_timeseries(self.tdms_data,  whichone=whichone, plot=False, save=True)
            self.plot_fft(self.tdms_data, whichone=whichone, plot=False, save=True)
            self.plot_spectrogram(self.tdms_data, whichone=whichone, plot=False, save=True, csv=False)
        self.tdms_data.to_csv(os.path.join(self.outputpath, f"data.csv"))
        
    def __add__(self, other):
        w0 = self.tdms_data.copy()
        w1 = other.tdms_data.copy()
        
         # update index
        interval = w0.index[1] - w0.index[0]
        elapsed_time = w0.index[-1] - w0.index[0] + interval
        w1.index += elapsed_time

        # update time column
        interval = w0.loc[:, 'time'].iloc[1] - w0.loc[:, 'time'].iloc[0]
        elapsed_time = w0.loc[:, 'time'].iloc[-1] - w0.loc[:, 'time'].iloc[0] + interval
        w1['time'] += elapsed_time # update index
        
        
        r_data = WearData(fpath=None)
        # update paths       
        r_data.fpath = Path(os.path.commonpath([self.fpath, other.fpath]))
        r_data.outputpath = Path(os.path.commonpath([self.outputpath, other.outputpath]))
        # update tdms data
        r_data.tdms_data = w0.append(w1, verify_integrity=True)
        # pass label_df
        r_data.label_df = self.label_df.copy()

        return r_data


if __name__ == '__main__':
    wear_data = WearData.read_and_merge()
    wear_data.save()
