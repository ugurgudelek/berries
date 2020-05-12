# Libraries
from pathlib import Path
import os
import pickle
import numpy as np
import pandas as pd
from nptdms import TdmsFile
from tqdm import tqdm

# Signal libraries
from scipy import signal, interpolate
import pywt  # wavelet analysis
import scaleogram as scg  # wavelet plot

# Plot libraries
import plotly
import plotly.offline as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.tools as tls
import plotly.express as px
import ipywidgets
# py.init_notebook_mode(connected=True)
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection, LineCollection
import matplotlib.ticker as ticker

import seaborn as sns
# plt.style.use('seaborn-whitegrid')

# Deep learning libraries
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms

# Local libraries
from dataset.generic import Standardizer, TimeSeriesDatasetWrapper
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import gc
import multiprocessing
import itertools

# from utils.plot_utils import camera_ready_matplotlib_style


from PIL import Image
from datetime import datetime

# dask
import dask
import dask.dataframe as dd
import dask.array as da

import datashader as ds
from datashader import transfer_functions as tf

import colorcet
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import datashade, rasterize, shade

import xarray as xr

import dask
from dask.distributed import Client, LocalCluster, Worker


class ExcelData:
    """
    In order to read cutting dataset excel
    """

    def __init__(self, root):
        self.root = root
        self.excel = pd.read_excel(self.root / 'Experimental Data Sheet.xlsx')
        self.excel = self.excel.drop([0], axis=0).reset_index(
            drop=True)  # drop mm|m/min like that row

    def get_attribute(self, cut_no, slot_no, kind):
        subset = self.excel.loc[
            (self.excel['cut_no'] == cut_no) & (self.excel['slot_no'] == slot_no) & (self.excel['kind'] == kind)]
        attr = subset.iloc[0].to_dict()

        attr['feed_rate'] = attr['feed_per_tooth'] * \
            attr['n_flute'] * attr['spindle_speed']
        attr['cutting_sec'] = 60 * attr['cutting_length'] / attr['feed_rate']
        attr['cutting_sec'] = np.floor(attr['cutting_sec'])
        return attr


class Toolwear:
    """
    Provides
        - raw vibration, accoustic data read from tdms file
        - toolwear analysis (fft, spectrogram, wavelet)
        - apply any filter in apply_aggregation method(savgol, lowpass)
        - save to and load from pickle options
        - making subset and plot them (plotly and matplotlib integration)
    """
    PLOT_TEMPLATE = 'plotly_white'

    def __init__(self, root, attributes, n_cycle=3, start_sec=3, fake=False, add_noise=False):

        self.root = root
        self.attributes = attributes

        self.n_cycle = n_cycle
        self.start_sec = start_sec
        self.add_noise = add_noise
        self.interpolated = False

        self.reading = None
        self.description = None

        self.wavelet_path = None
        self.scales = None
        # self.wavelet_xarr = None

        # add ref signal: signal_freq*2 Hz
        # self.reading['data'] += 0.01*np.sin(2*np.pi*self.expected_tooth_freq*2*self.reading['time'])

        # add ref signal: signal_freq*0.5 Hz
        # self.reading['data'] += 0.005*np.sin(2*np.pi*self.expected_tooth_freq*0.5*self.reading['time'])

        # create aggregation columns
        # self.apply_aggregation()

    @property
    def rpm(self):
        return self.attributes['spindle_speed']

    @property
    def n_flute(self):
        return self.attributes['n_flute']

    @property
    def sampling_freq(self):
        return self.attributes['sampling_freq']

    @property
    def sampling_period(self):
        return 1 / self.attributes['sampling_freq']

    @property
    def T(self):
        """
        T: interval : 1/Fs
        :return:
        """
        return 1 / self.sampling_freq

    @property
    def expected_tooth_freq(self):
        return self.n_flute * self.rpm / 60

    @property
    def dt_tooth(self):
        return 1 / self.expected_tooth_freq

    @property
    def num_points_in_one_period(self):
        """
        Return the number of data points for 1 revolution.
        """
        return int(self.sampling_freq / (self.rpm / 60))

    @property
    def num_points_per_tooth(self):
        return self.num_points_in_one_period // self.n_flute

    def create_fake_data(self):
        raise Exception("You should fix self.rpm line!")
        # sec = 10
        # rpm = 120
        # n_flute = 4
        # expected_freq = int((rpm/60)*n_flute)
        # samplig_freq = 6000 # Hz
        # num_points_in_one_period = samplig_freq//expected_freq

        # skip_sin = 2 # number of skip sine wave

        # steps = np.linspace(0, sec, sec*samplig_freq)
        # orig = np.sin(2*np.pi*expected_freq*steps)
        # s = np.sin(2*np.pi*expected_freq*steps)
        # for i in range(0, len(steps), (skip_sin+1)*(num_points_in_one_period)):
        #     s[i:i+num_points_in_one_period*skip_sin]= 0.
        # s[s<0] = 0.

        skip_sin = 9  # number of skip sine wave

        steps = np.linspace(0, 10, 10 * self.sampling_freq)
        #             orig = np.sin(2*np.pi*self.expected_tooth_freq*steps)
        s = np.sin(2 * np.pi * self.expected_tooth_freq * steps)
        for i in range(0, len(steps), (skip_sin + 1) * (self.num_points_per_tooth)):
            s[i:i + self.num_points_per_tooth * (skip_sin)] = 0.
        s[s < 0] = 0.

        self.reading = pd.DataFrame()
        self.reading['time'] = steps
        self.reading['data'] = s
        self.rpm = self.rpm // (skip_sin + 1)

        if self.add_noise:
            self.reading['data'] += np.random.randn(self.reading.shape[0])

    def second2index(self, sec):
        return int(sec * self.sampling_freq)

    @staticmethod
    def denoise(subsets,
                threshold_mul=0.2, threshold_epsilon=0.0,
                method='envelope',
                inplace=False, plot=False, save=False, figsize=None, filename=None, verbose=False):
        """

        :param subsets:
        :param threshold_mul:
        :param method: savgol_difference, envelope
        :param plot:
        :return:
        """

        _islist = (type(subsets) is list) or (type(subsets) is tuple)

        subsets = subsets if _islist else [subsets]

        if plot:
            fig, axes = plt.subplots(nrows=len(subsets) if _islist else 2, ncols=2 if _islist else len(subsets),
                                     squeeze=False, sharex=True, figsize=figsize)

        denoiseds = list()
        for i, subset in enumerate(subsets):
            t = np.arange(0, subset['time'].values.shape[0]
                          ) if _islist else subset['time'].values
            raw_data = subset['data'].values
            data = raw_data.copy()

            if method == 'basic':
                threshold = threshold_epsilon
                nan_valid_bool = np.abs(data) > threshold

            elif method == 'envelope':

                envelope_result = Toolwear.envelope(subset)
                pos_index = envelope_result['pos_index']
                pos_envelope = envelope_result['pos_envelope']
                neg_index = envelope_result['neg_index']
                neg_envelope = envelope_result['neg_envelope']

                nan_valid_bool = np.empty(data.shape, dtype=np.bool)

                # axes[i, 0].plot(t[pos_index], pos_envelope, label='pos_envelope', alpha=1., c='b', linewidth=0.7)
                # axes[i, 0].plot(t[neg_index], neg_envelope, label='neg_envelope', alpha=1., c='b', linewidth=0.7)

                pos_envelope = signal.savgol_filter(pos_envelope,
                                                    window_length=51,
                                                    polyorder=1)

                neg_envelope = signal.savgol_filter(neg_envelope,
                                                    window_length=51,
                                                    polyorder=1)

                pos_threshold = pos_envelope + \
                    (threshold_epsilon + threshold_mul *
                     data[pos_index].mean())
                neg_threshold = neg_envelope - \
                    (threshold_epsilon - threshold_mul *
                     data[neg_index].mean())

                nan_valid_bool[pos_index] = (data[pos_index] > pos_threshold)
                nan_valid_bool[neg_index] = (data[neg_index] < neg_threshold)

                nan_valid_index = np.where(nan_valid_bool)

                drop_count = np.count_nonzero(nan_valid_bool)
                drop_ratio = np.count_nonzero(
                    nan_valid_bool) / nan_valid_bool.shape[0]
                if verbose:
                    print(f'Pos_mean:{data[pos_index].mean()}')
                    print(f'Neg_mean:{data[neg_index].mean()}')
                    print(
                        f'Pos_threshold_coef:{(threshold_epsilon + threshold_mul * data[pos_index].mean())}')
                    print(
                        f'Neg_threshold_coef:{(threshold_epsilon - threshold_mul * data[neg_index].mean())}')
                    print(
                        f'Dropped data number:{drop_count} (%{drop_ratio:.5f})')

                axes[i if _islist else 0, 0 if _islist else i].scatter(t[nan_valid_index], data[nan_valid_index],
                                                                       label=f'dropped {drop_count}(%{drop_ratio})',
                                                                       alpha=1., c='crimson', s=10, marker='+')
                axes[i if _islist else 0, 0 if _islist else i].plot(t[pos_index], pos_threshold, label='pos_threshold',
                                                                    alpha=.5, c='r', linewidth=0.7, linestyle='dashed')
                axes[i if _islist else 0, 0 if _islist else i].plot(t[neg_index], neg_threshold, label='neg_threshold',
                                                                    alpha=.5, c='r', linewidth=0.7, linestyle='dashed')

            elif method == 'envelope_mean':
                threshold = Toolwear.envelope(
                    subset).mean() * threshold_mul + threshold_epsilon
                nan_valid_bool = np.abs(data) > threshold

            elif method == 'savgol_difference':
                threshold = 1 * threshold_mul + threshold_epsilon
                savgol = signal.savgol_filter(data,
                                              window_length=9,
                                              polyorder=3)

                diff = np.abs(data - savgol)

                nan_valid_bool = diff >= threshold

            else:
                raise ValueError(f'{method} not an option.')

            # data[nan_valid_bool] = np.nan
            # denoised = pd.Series(data).interpolate(method='cubic')
            data[nan_valid_bool] = 0
            denoised = data
            denoiseds.append(denoised)

            if inplace:
                subset.loc[:, 'data'] = denoised

            if plot:
                # axes[i, 0].scatter(t, raw_data, label='original', c='b', marker='+', s=20.)
                axes[i if _islist else 0, 0 if _islist else i].scatter(t, denoised, label='denoised', alpha=1., c='y',
                                                                       marker='.', s=10)

                # axes[i, 0].plot(threshold, c='r', linestyle='dashed', linewidth=0.4,  label='threshold')
                # axes[i, 0].plot(-threshold, c='r', linestyle='dashed', linewidth=0.4,)

                axes[i if _islist else 1, 1 if _islist else i].scatter(t, denoised, label='denoised', alpha=1., c='y',
                                                                       marker='.', s=10)

                ylim = (-2, 2)
                axes[i if _islist else 0, 0 if _islist else i].set_ylim(*ylim)
                axes[i if _islist else 1, 1 if _islist else i].set_ylim(*ylim)

                axes[0, 0].legend(loc='upper center', bbox_to_anchor=(
                    0.5, 1.25), fancybox=True, shadow=True, ncol=5)

                if save:
                    os.makedirs('denoised', exist_ok=True)
                    plt.figure(fig.number)
                    plt.savefig(
                        filename or f'denoised/denoised_{subset["time"].iloc[0]}.jpg')

        if plot:
            plt.close()

        return denoiseds if _islist else denoiseds[0]

    @staticmethod
    def envelope(subsets, plot=False):
        _type = type(subsets)

        subsets = subsets if _type is list else [subsets]

        if plot:
            fig, axes = plt.subplots(
                nrows=len(subsets), ncols=1, squeeze=False, sharex=True)

        amplitude_envelopes = list()
        for i, subset in enumerate(subsets):
            s = subset['data'].values
            # t = subset['time'].values
            t = np.arange(0, subset['time'].values.shape[0])

            pos_s_index = np.where(s >= 0)[0]
            neg_s_index = np.where(s < 0)[0]

            pos_s = s[pos_s_index]
            neg_s = s[neg_s_index]

            pos_envelope = np.abs(signal.hilbert(pos_s))
            neg_envelope = -np.abs(signal.hilbert(np.abs(neg_s)))

            amplitude_envelopes.append({'pos_index': pos_s_index,
                                        'pos_envelope': pos_envelope,
                                        'neg_index': neg_s_index,
                                        'neg_envelope': neg_envelope})
            if plot:
                axes[i, 0].plot(t, s, '.', label='signal')
                axes[i, 0].plot(t[pos_s_index], pos_envelope,
                                '-', label='pos_envelope')
                axes[i, 0].plot(t[neg_s_index], neg_envelope,
                                '-', label='neg_envelope')
                axes[i, 0].axhline(y=pos_envelope.mean(), c='r', linestyle='dashed',
                                   label='pos_env.mean')
                axes[i, 0].axhline(y=neg_envelope.mean(), c='r', linestyle='dashed',
                                   label='neg_env.mean')

                axes[i, 0].axhline(y=0, c='k', linestyle='dashed')

                ylim = max(pos_envelope.max(), -neg_envelope.min())
                axes[i, 0].set_ylim(-ylim, ylim)
                fig.legend(
                    *axes[0, 0].get_legend_handles_labels(), loc='upper left')
                fig.suptitle('Envelope')
        return amplitude_envelopes if _type is list else amplitude_envelopes[0]

    def apply_aggregation(self):

        # apply savgol filter
        SAVGOL_WINDOW_LEN = int(self.num_points_in_one_period / 1)
        SAVGOL_WINDOW_LEN = SAVGOL_WINDOW_LEN if SAVGOL_WINDOW_LEN % 2 == 1 else SAVGOL_WINDOW_LEN + 1
        self.reading['savgol'] = signal.savgol_filter(self.reading['data'],
                                                      window_length=SAVGOL_WINDOW_LEN,
                                                      polyorder=1)
        # apply lowpass filter
        LOWPASS_ORDER = 2
        CUTOFF = 2500  # desired cutoff frequency of the filter, Hz
        self.reading['lowpass'] = self.butter_lowpass_filter(
            self.reading['data'], CUTOFF, self.sampling_freq, LOWPASS_ORDER)

        self.reading['lowpass+savgol'] = signal.savgol_filter(self.reading['lowpass'],
                                                              window_length=SAVGOL_WINDOW_LEN,
                                                              polyorder=1)

    def __len__(self):
        return self.reading.shape[0]

    #  PLOT METHODS
    @staticmethod
    def _make_trace(data, colname, mode='lines'):
        return go.Scattergl(name=colname,
                            x=data['time'],
                            y=data[colname],
                            mode=mode,
                            marker=dict(line=dict(width=1)))

    def make_subset_after(self, sec):
        """make subset starting at 'sec' seconds"""
        self.start_sec = sec
        return self._make_subset(start_idx=int(self.sampling_freq * self.start_sec))

    def _make_subset(self, start_idx):
        return self.reading.iloc[start_idx:start_idx + self.num_points_in_one_period * self.n_cycle, :].copy()

    def static_plot(self, subset=None, bound=None, figsize=None, save=False, fpath=None):

        if subset is None:
            subset = self.reading
        if bound is not None:
            subset = subset.iloc[
                int(self.sampling_freq * bound[0]):int(self.sampling_freq * bound[1]), :]

        plt.figure(figsize=figsize)
        ax = plt.plot(subset['time'], subset['data'], '.')

        if save:
            # self.save_fig(ax.get_figure(), suffix='signal-full', kind='matplotlib', fpath=fpath)
            os.makedirs(fpath, exist_ok=True)
            plt.savefig(f"{fpath}.jpg")

    def plot(self, plot=True, save=True):
        def add_vline_per_period(fig):
            # add vertical line per period
            for i in range(self.n_cycle):
                fig.add_shape(
                    # Line Vertical
                    dict(
                        type="line",
                        x0=subset['time'].iloc[self.num_points_in_one_period * i],
                        y0=subset['data'].max(),
                        x1=subset['time'].iloc[self.num_points_in_one_period * i],
                        y1=subset['data'].min(),
                        line=dict(
                            color="RoyalBlue",
                            width=3,
                            dash="dashdot"
                        )
                    ))
            return fig

        def init():

            # Create initial figure
            colnames = ['data', 'lowpass', 'savgol', 'lowpass+savgol']
            traces = [self._make_trace(
                data=subset, colname=colname, mode='lines') for colname in colnames]
            fig = go.FigureWidget(data=traces,
                                  layout=go.Layout(xaxis={'title': 'Time(sec)'},
                                                   yaxis={
                                                       'title': 'Amplitude'},
                                                   template=Toolwear.PLOT_TEMPLATE))
            fig = add_vline_per_period(fig)
            return fig

        # Create initial subset
        subset = self.make_subset_after(sec=self.start_sec)
        fig = init()

        # create min,max dictionary to faster update_range
        y_min_max_lookup = {
            d.name: {'min': d['y'].min(), 'max': d['y'].max()} for d in fig.data}

        def update_subset(start_sec):
            subset = self.make_subset_after(sec=start_sec)

            @ipywidgets.interact(xaxis_range=ipywidgets.widgets.FloatRangeSlider(min=subset['time'].iloc[0],
                                                                                 max=subset['time'].iloc[-1],
                                                                                 step=(
                subset['time'].iloc[-1] -
                subset['time'].iloc[
                    0]) / self.sampling_freq,
                description='Time',
                continuous_update=False,
                readout=True,
                readout_format='.3f',
                layout=ipywidgets.Layout(
                                                                                     align_items='center',
                                                                                     width='90%', height='80px')))
            def update_plot(xaxis_range):
                if len(fig.data) <= 3:
                    return

                # update traces with new subset
                for d in fig.data:
                    d['x'] = subset['time'].values
                    d['y'] = subset[d.name].values

                # update xaxis range wrt range slider
                fig.layout.xaxis.range = [xaxis_range[0], xaxis_range[1]]

                # update shapes wrt selected traces' max amplitude
                global_active_min_list = [
                    y_min_max_lookup[d.name]['min'] for d in fig.data if d.visible == True]
                global_active_max_list = [
                    y_min_max_lookup[d.name]['max'] for d in fig.data if d.visible == True]
                if len(global_active_min_list) and len(global_active_max_list):
                    _min = min(global_active_min_list)
                    _max = max(global_active_max_list)
                    for s in fig['layout']['shapes']:
                        s['y0'] = _max * 1.1
                        s['y1'] = _min * 1.1

        #         def update_n_cycle(n_cycle):
        #             self.n_cycle = n_cycle
        #             subset = self.make_subset_after(sec=self.start_sec)
        #             print(subset.shape)
        #             fig['layout']['shapes'] = None
        #             add_vline_per_period(fig)

        #         n_cycle_slider_widget = interactive(update_n_cycle, n_cycle=widgets.IntSlider(value=self.n_cycle,
        #                                                                               min=1,
        #                                                                               max=100,
        #                                                                               step=1,
        #                                                                               description='Number of Cycle:',
        #                                                                               disabled=False,
        #                                                                               continuous_update=False,
        #                                                                               readout=True,
        #                                                                               readout_format='d'))

        control_widget = ipywidgets.interactive(update_subset,
                                                start_sec=ipywidgets.widgets.BoundedFloatText(
                                                    # means start_sec
                                                    value=subset['time'].iloc[0],
                                                    min=self.reading['time'].iloc[0],
                                                    max=self.reading['time'].iloc[-1],
                                                    step=0.5,
                                                    description='Start Second:',
                                                    disabled=False,
                                                    layout=ipywidgets.Layout(align_items='center',
                                                                             width='50%',
                                                                             )
                                                ))

        form = ipywidgets.Box((control_widget, fig),
                              layout=ipywidgets.Layout(display='flex',
                                                       flex_flow='column',
                                                       border='solid 2px',
                                                       align_items='stretch',
                                                       width='100%'))

        if save:
            self.save_fig(fig, suffix='signal', kind='plotly')
        # if plot:
        # display(form)

    # @camera_ready_matplotlib_style
    def plot_3d(self, subsets, kind, plot=True):
        """
        :param subsets: should be list containing subsets
        :param plot:
        :return:
        """

        measurement_points = list()
        rep = list()
        zmin, zmax = None, None

        for i, subset in enumerate(subsets):
            if kind == 'fft':
                fft_data, fft_freqs = self.fft(
                    data=subset, plot=False, save=False)
                x = fft_freqs
                y = fft_data
                xlabel = 'Frequencies [Hz]'
                ylabel = 'Measurement Point'
                zlabel = 'Amplitude'
                xlim = (0, 2500)

            elif kind == 'signal':
                y = subset['data'].values
                x = np.arange(0, y.shape[0])
                xlabel = 'Time'
                ylabel = 'Measurement Point'
                zlabel = 'Amplitude'
                xlim = (x.min(), x.max())

            else:
                raise NotImplementedError(
                    f"{kind} not implemented for 3d plot.")

            if zmax is not None:
                if zmax < y.max():
                    y.max()

            zmax = y.max() if zmax is None or zmax < y.max() else zmax
            zmin = y.min() if zmin is None or zmin > y.min() else zmin

            measurement_points.append(subset['time'].iloc[0])
            rep.append(list(zip(x, y)))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        poly = LineCollection(rep)
        poly.set_alpha(0.7)

        zs = measurement_points
        ax.add_collection3d(poly, zs=zs, zdir='y')

        ax.set_xlabel(xlabel)
        ax.set_xlim3d(*xlim)
        ax.set_ylabel(ylabel)
        ax.set_ylim3d(min(zs), max(zs))
        ax.set_zlabel(zlabel)
        if zmax == 0:
            raise ValueError('zmax should be higher than 0.')
        ax.set_zlim3d(zmin, zmax)
        return ax

    def plot_toolwear(self):
        plt.plot(self.reading['cutting_length'], self.reading['toolwear'])
        plt.xlabel('Cutting Length')
        plt.ylabel('Toolwear')

    def save_fig(self, fig, suffix, kind='plotly', fpath=None):
        # path : 'D:/machining/wear_data/raw/1/acc7000.tdms'

        if fpath is None:
            try:
                filename = self.path.stem  # 'acc7000'
                foldername = self.path.parent.name  # '1'
                images_path = self.path.parent.parent.parent / \
                    'images'  # 'D:/machining/wear_data/images'
                # 'D:/machining/wear_data/images/1'
                specific_folder_path = images_path / foldername
            except:
                print(
                    "You cannot save because path is not valid. This is probably because you call 'append' function.")
                return

        else:
            specific_folder_path = fpath

        os.makedirs(specific_folder_path, exist_ok=True)
        save_path = specific_folder_path

        # save_path = specific_folder_path / filename  # 'D:/machining/wear_data/images/1/acc7000.png

        if kind == 'plotly':
            # fig.write_image(str(save_path.with_suffix('.png')))
            fig.write_html(str(save_path.with_suffix(f'.{suffix}.html')))
        if kind == 'matplotlib':
            plt.savefig(str(save_path.with_suffix(f'.{suffix}.png')))

    def fft(self, data, bound=(0, 2500), plot=False, save=True):
        Fs = self.sampling_freq  # sampling freq
        Ts = 1 / Fs  # sampling interval
        t = data['time']  # time vector

        n = data.shape[0]  # data lenght
        k = np.arange(n)
        T = n / Fs
        freq = k / T

        S = data['data'].values
        S_fft = np.fft.fft(S)

        freq = freq[:n // 2]  # one side frequency range
        freq = freq[freq < bound[1]]  # cut unnecessary frequencies

        S_fft = S_fft.real[:freq.shape[0]] / \
            n  # fft computing and normalization
        S_fft = np.abs(S_fft)  # y-axis symetric correction

        fft_data = S_fft
        fft_freqs = freq

        if save or plot:
            fig = go.Figure(data=[go.Scattergl(x=fft_freqs, y=fft_data, name='fft', mode='markers+lines')],
                            layout=go.Layout(xaxis={'title': 'Freq(Hz)'},
                                             yaxis={'title': 'Amplitude'},
                                             template=Toolwear.PLOT_TEMPLATE))
        if save:
            self.save_fig(fig, suffix='fft', kind='plotly')
        if plot:
            fig.show()

        return fft_data, fft_freqs

    def spectrogram(self, data, bound=(0, 2500), plot=False, save=True, engine='plotly'):

        window = self.num_points_per_tooth * 3
        nftt = self.num_points_per_tooth * 100
        #         noverlap=0
        #         print(f"window:{window}  nftt:{nftt}")

        spect_freqs, spect_times, spectrogram_data = signal.spectrogram(
            x=data['data'],
            fs=self.sampling_freq,
            # increase in window : decrease in precision on time axis
            window=signal.windows.tukey(window, alpha=0.25, sym=False),
            # increase in nfft : increase in precision on freq axis
            nfft=nftt,
            #             noverlap=0,
            scaling='spectrum',
            mode='magnitude')

        # cut unnecessary frequencies
        spect_freqs = spect_freqs[spect_freqs < bound[1]]
        spect_times = spect_times
        spectrogram_data = spectrogram_data[:spect_freqs.shape[0], :]

        fig = go.Figure(data=[go.Heatmap(z=spectrogram_data,
                                         x=spect_times,
                                         y=spect_freqs,
                                         name='spectrogram',
                                         hoverongaps=False,
                                         hovertemplate='X: %{x:.4f}h <br>Y: %{y} <br>Z: %{z}', )],
                        layout=go.Layout(xaxis={'title': 'Time(sec)'},
                                         yaxis={'title': 'Freq(Hz)'},
                                         template=Toolwear.PLOT_TEMPLATE),
                        )

        if engine == 'matplotlib':
            fig, ax = plt.subplots(nrows=1)
            Pxx, freqs, bins, im = ax.specgram(data['data'], NFFT=self.num_points_per_tooth,
                                               Fs=self.sampling_freq, noverlap=0)
            # The `specgram` method returns 4 objects. They are:
            # - Pxx: the periodogram
            # - freqs: the frequency vector
            # - bins: the centers of the time bins
            # - im: the matplotlib.image.AxesImage instance representing the data in the plot
            plt.show()

        if save:
            self.save_fig(fig, suffix='spectrogram', kind='plotly')
        if plot:
            fig.show()

    @staticmethod
    def butter_lowpass_filter(data, cutoff, fs, order=5):
        # butter_lowpass
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        y = signal.lfilter(b, a, data)
        return y

    @staticmethod
    def wavelet(subset, wavelet='cmor6-1.5',
                plot_func='own',  # plot_func can be 'lib' to use more robust scaleogram plot
                fpath='wavelet-results',
                cwt_save=True,
                plot=True,
                clim=None,
                ):
        """
        X-axis: time
        Y-axis: scale - The scale correspond to the signal periodicity to which the transform is sentitive to.
        wavelets does not detect a single frequency but rather a band.
        fuzziness is proportional to the scale on y-axis
        wavelet_fun = <name><B>-<center-freq>
                    B: inversly proportional to bandwidth(smoothing) (The bandwidth parameter allow to tune the sensitivity on the period axis (Y). Or in other words the visibl amount of smoothing.)

        With logscale on Y axis, the bandwith will have the same height at all scales which may be helpful for data interpretation.
        ----------------------------------------------------------------------
        time   = np.arange(200, dtype=np.float16)-100
        data   =  np.exp(-0.5*((time)/0.2)**2)  # insert a gaussian at the center
        scales = np.arange(1,101) # scaleogram with 100 rows
        # compute ONCE the Continuous Wavelet Transform
        cwt    = scg.CWT(time, data, scales)

        # plot 1 with full range
        scg.cws(cwt)

        # plot 2 with a zoom
        scg.cws(cwt, xlim=(-50, 50), ylim=(20, 1))

        """

        os.makedirs(fpath, exist_ok=True)
        os.makedirs(f"{fpath}/figures", exist_ok=True)
        os.makedirs(f"{fpath}/data", exist_ok=True)

        t = subset['time'].values
        signal = subset['data'].values
        output_filename = f"{t[0]:.3f}"

        # scales = np.arange(1, 500, 1)
        scales = scg.periods2scales(np.logspace(start=0, stop=2.5, num=200))

        # Calculate wavelet
        cwt = scg.CWT(time=t,
                      signal=signal,
                      scales=scales,
                      wavelet=wavelet)

        if cwt_save:
            with open(f"{fpath}/data/{output_filename}.pickle", "wb") as f:
                pickle.dump(cwt, f)

        if plot:
            fig, axes = plt.subplots(
                nrows=2, ncols=1, figsize=None, squeeze=False, frameon=False, sharex=True)
            # Vibration subplot
            time_plot = axes[1, 0].plot(cwt.time, cwt.signal)
            axes[1, 0].set_ylabel('Magnitude')

            if plot_func == "own":

                # Wavelet subplot
                z = np.abs(cwt.coefs).astype(np.float)
                qmesh = axes[0, 0].pcolormesh(
                    cwt.time, range(cwt.scales_freq.shape[0]), z)
                if clim is not None:
                    qmesh.set_clim(*clim)
                axes[0, 0].set_ylabel('Frequency')
                colorbar = plt.colorbar(
                    qmesh, orientation='vertical', ax=axes[0, 0])

            elif plot_func == "lib":
                scg.cws(cwt,
                        spectrum='power',
                        figsize=(12, 6),
                        yscale='log',
                        ylabel="Period [seconds]",
                        xlabel='Time [seconds]',
                        ax=axes[0, 0])

                """
                text = ax.annotate("found freq",
                                   xy=(time[100], 1 / self.expected_tooth_freq),
                                   xytext=None,
                                   bbox=dict(boxstyle="round", facecolor="y", edgecolor="0.5"),
                                   arrowprops=dict(facecolor='yellow', shrink=0.05))
        
                text = ax.annotate(f"Ref freq x2.0:{self.expected_tooth_freq * 2}",
                                   xy=(time[100], 1 / (self.expected_tooth_freq * 2)),
                                   xytext=None,
                                   bbox=dict(boxstyle="round", facecolor="y", edgecolor="0.5"),
                                   arrowprops=dict(facecolor='yellow', shrink=0.05))
                text = ax.annotate(f"Ref freq x0.5:{self.expected_tooth_freq * 0.5}",
                                   xy=(time[100], 1 / (self.expected_tooth_freq * 0.5)),
                                   xytext=None,
                                   bbox=dict(boxstyle="round", facecolor="y", edgecolor="0.5"),
                                   arrowprops=dict(facecolor='yellow', shrink=0.05))
        
                #         ax.use_sticky_edges = False
                #         p2f = lambda x:1/x
                #         f2p = lambda x:1/x
                #         secax = ax.secondary_yaxis('right', functions=(p2f, f2p))
                #         secax.set_ylabel('Freq [Hz]')
                #         pseudo_frequencies = pywt.scale2frequency(wavelet=wavelet, scale=scales)
                #          for i in range(self.n_cycle):
                #             ax.axvline(x=subset['time'].values[i*self.num_points_in_one_period], ymin=0.1, ymax=0.9, c='w')
        
                #         # remove colorbar
                #         ax.collections[0].colorbar.remove()
        
                if save:
                    self.save_fig(ax.get_figure(), suffix='wavelet', kind='matplotlib')
                #         if plot:
                plt.show()
                
                """

            else:
                raise ValueError(f"plot_func:{plot_func} not implemented.")

            plt.figure(fig.number)
            plt.suptitle('Vibration Wavelet Analysis')
            plt.xlabel('Time [seconds]')
            plt.savefig(f"{fpath}/figures/{output_filename}.jpg")
            plt.close()

        return cwt

    # END OF PLOT METHODS

    def append(self, other):

        # UPDATE READING
        _other_reading = other.reading.copy()

        # update index
        interval = self.reading.index[1] - self.reading.index[0]
        elapsed_time = self.reading.index[-1] - \
            self.reading.index[0] + interval
        _other_reading.index += elapsed_time

        # update time column
        interval = self.reading.loc[:, 'time'].iloc[1] - \
            self.reading.loc[:, 'time'].iloc[0]
        elapsed_time = self.reading.loc[:, 'time'].iloc[-1] - \
            self.reading.loc[:, 'time'].iloc[0] + interval
        _other_reading['time'] += elapsed_time

        # update tdms data
        self.reading = self.reading.append(
            _other_reading, verify_integrity=True)

        # UPDATE ATTRIBUTES
        for attr_name, attr_val in self.attributes.items():
            if type(attr_val) is list:
                self.attributes[attr_name].append(other.attributes[attr_name])
            else:
                if attr_val != other.attributes[attr_name]:  # create new list
                    self.attributes[attr_name] = [
                        attr_val, other.attributes[attr_name]]
                # else continue - if they are same value, no need to do anything

    def fix_time(self):
        self.reading['time'] = np.arange(self.__len__()) * self.T
        self.reading['time'] = self.reading['time'].astype(np.float32)

    def raw_tdms_data(self, tdms_path):
        reading: pd.DataFrame = TdmsFile(tdms_path).as_dataframe()
        reading = reading.rename(
            columns={reading.columns[0]: "data"})

        reading['data'] = reading['data'].astype(np.float32)

        # add time column
        reading['time'] = np.arange(len(reading)) * self.T
        reading['time'] = reading['time'].astype(np.float32)

        # add cutting_length column
        reading['cutting_length'] = self.attributes['cutting_length']
        reading['cutting_length'] = reading['cutting_length'].astype(
            np.float32)

        # add toolwear column
        reading['toolwear'] = self.attributes['toolwear']
        reading['toolwear'] = reading['toolwear'].astype(np.float32)

        return reading

    @staticmethod
    def raw_batch_read(root, cut_no, kind, n_cycle,
                       cut=True,
                       cut_dropping_seq=True,
                       interpolate=True,
                       save_parquet=True):

        # read excel file for attributes
        exceldata = ExcelData(root=root)

        vib = None
        tdms_glob = list((root / f'raw/{cut_no}').glob(f'{kind}*.tdms'))
        with tqdm(total=len(tdms_glob), desc='Batch File Read') as pbar:
            for slot_no, tdms_path in enumerate(tdms_glob, start=1):
                vib_current = Toolwear(root=root,
                                       attributes=exceldata.get_attribute(cut_no=cut_no,
                                                                          slot_no=slot_no,
                                                                          kind=kind),
                                       n_cycle=n_cycle)

                vib_current.reading = vib_current.raw_tdms_data(
                    tdms_path=tdms_path)

                if vib is None:
                    vib = vib_current
                else:
                    vib.append(vib_current)

                pbar.update(1)

        if cut:
            # cut nan-valid data
            cut_excel = pd.read_excel(
                root / "verilerin_ayiklanmasi.xlsx", sheet_name='cut')
            cut_excel = cut_excel.loc[cut_excel['cutno'] == cut_no]

            for i, (_, start, stop) in cut_excel.iterrows():
                if stop == -1:
                    # for machine learning model, cut_dropping_seq should be True.
                    # this will used for plot only
                    if not cut_dropping_seq:
                        continue
                    stop = len(vib.reading)
                vib.reading.iloc[vib.second2index(
                    start):vib.second2index(stop)] = np.nan
            vib.reading = vib.reading.dropna(axis=0).reset_index(drop=True)

            # fix time-index before interpolating
            # bacause it uses 'time' index
            vib.fix_time()

        if interpolate:
            # custom interpolation
            vib.interpolate('cutting_length')
            vib.interpolate('toolwear')
            vib.interpolated = True

        # save reading to parquet
        if save_parquet:
            # calculate dataset description
            vib.description = vib.reading.describe()
            vib.to_parquet()
        return vib

    @staticmethod
    def from_parquet(root, client, cut_no):
        """[summary]

        Arguments:
            root {[Path]} -- [description]
            client {[dask.distributed.Client]} -- [description]
            cut_no {[int]} -- [description]

        Returns:
            [Toolwear] -- [description]
        """

        parquet_path = root / f'parquet/{cut_no}'
        with open(parquet_path / 'attributes.pickle', 'rb') as f:
            attributes = pickle.load(f)

        with open(parquet_path / 'description.pickle', 'rb') as f:
            description = pickle.load(f)

        vib = Toolwear(root=root,
                       attributes=attributes)
        vib.reading = dd.read_parquet(parquet_path, engine='fastparquet')
        vib.description = description
        vib.client = client

        wavelet_path = root / f'wavelet/{cut_no}'
        if os.path.exists(wavelet_path):
            try:
                vib.wavelet_path = wavelet_path
                vib.wavelet_image = Toolwear.read_wavelet_image(
                    wavelet_path / f'wavelet_{cut_no}_log_thin.png')

                # if labels are not recorded. record them once.
                if not os.path.exists(wavelet_path / 'label.csv'):
                    toolwear = vib.reading['toolwear'].compute()
                    inc = toolwear.shape[0] // vib.wavelet_image.shape[0]
                    ix = np.array(
                        [i for i in range(0, toolwear.shape[0] - inc, inc)])
                    label = toolwear[ix].values
                    pd.Series(label, name='label').to_csv(
                        wavelet_path / 'label.csv')

                vib.wavelet_label = pd.read_csv(
                    wavelet_path / 'label.csv', index_col=0)['label'].values

                vib.wavelet_description = pd.read_csv(
                    wavelet_path / 'desc.csv', index_col=0).T  # [mean, std, max, min] columns
                vib.wavelet_aux = vib.attributes['cutting_speed']
            except:
                pass

        return vib

    @staticmethod
    def read_wavelet_image(img_path):
        img = Image.open(img_path).convert('L')
        img = np.array(img) / 255
        return pd.DataFrame(img.T)

    def interpolate(self, yname='cutting_length'):
        """ 
        Call this function after total merge!
        """

        xname = 'time'

        # quick fix for proper interpolation (to not go below 0)
        # below line extract proper interpolation points.
        from itertools import accumulate

        def successive2impulse(series):
            # target_indexs = <class 'dict(index, value)'>: {   4633414: 4400.0, 5683199: 8800.0,
            #                                                   4915199: 13200.0, 5299199: 17600.0,
            #                                                   4671999: 22000.0, 4953599: 26400.0,
            #                                                   0: 0}
            counts = [(series == t).sum() for t in series.unique()]
            indices = [0] + list(map(lambda x: x - 1, accumulate(counts)))
            values = [0] + list(series.unique())

            return indices, values

        target_indices, target_values = successive2impulse(self.reading[yname])
        self.reading[yname] = np.nan
        self.reading.loc[target_indices, yname] = target_values

        y = self.reading[yname].dropna()
        x = self.reading.loc[y.index, xname]

        interp_f = interpolate.interp1d(x.values, y.values,
                                        kind='cubic',
                                        fill_value="extrapolate")
        new_x = self.reading[xname].values
        self.reading[yname] = interp_f(new_x)

    def to_parquet(self):
        os.makedirs(self.root / 'parquet', exist_ok=True)
        parquet_path = self.root / f'parquet/{int(self.attributes["cut_no"])}'
        if os.path.exists(parquet_path):
            raise Exception(f'{parquet_path} already exists.')
        os.makedirs(parquet_path, exist_ok=True)

        # save attributes to pickle
        with open(parquet_path / 'attributes.pickle', 'wb') as f:
            pickle.dump(self.attributes, f)

        # save description to pickle
        with open(parquet_path / 'description.pickle', 'wb') as f:
            pickle.dump(self.description, f)

        chunksize = int(self.sampling_freq * 60)  # 60 sec chunksize
        dataframe = dd.from_pandas(self.reading.loc[:, ['time', 'data', 'cutting_length', 'toolwear']],
                                   chunksize=chunksize)
        # dataframe.set_index('time', sorted=True)
        print('Saving parquet file...')
        dataframe.to_parquet(parquet_path, engine='fastparquet')

    def plot_holoviews(self, plot_height, plot_width, xname=None, yname=None, save_path=None):
        xname = xname or 'time'
        yname = yname or 'data'

        points = hv.Points(self.reading, kdims=[
            xname, yname], label=f'Cut {self.attributes["cut_no"]}')
        img = shade(rasterize(points))

        img.opts(height=plot_height, width=plot_width,
                 backend='bokeh',
                 fontsize={'title': 16, 'labels': 14, 'xticks': 12, 'yticks': 12})

        return img

    def plot_datashader(self, plot_height, plot_width, xname=None, yname=None, save_path=None):
        xname = xname or 'time'
        yname = yname or 'data'

        canvas = ds.Canvas(plot_height=plot_height, plot_width=plot_width,
                           x_range=None, y_range=None,
                           x_axis_type='linear', y_axis_type='linear')
        points = canvas.points(self.reading, xname, yname)

        img = tf.shade(points, cmap=colorcet.bmy)
        if save_path is not None:
            img.to_pil().save(save_path)

        # mpl_image = matplotlib.image.pil_to_array(img.to_pil())
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.imshow(mpl_image)
        return img

    def dask_wavelet_init(self, scales=np.arange(1, 129), wavelet='cmor1-25'):

        cut_no = self.attributes['cut_no']
        self.wavelet_path = self.root / f'wavelet/{int(cut_no)}'
        os.makedirs(self.wavelet_path, exist_ok=True)
        self.scales = scales

        signal = self.reading['data'].to_dask_array()
        t = self.reading['time'].to_dask_array()

        # make iterable signal delayed to construct dask pipeline
        signal = dask.delayed(signal)
        t = dask.delayed(t)

        powers, frequencies = Toolwear.dask_wavelet_calculation(signal, scales, wavelet, self.sampling_period,
                                                                shape=(self.description.loc['count', 'data'],))
        powers = np.log(powers)

        wavelet_xarr = xr.DataArray(powers,
                                    dims=['scales', 'time'],
                                    coords=[('scales', scales[::-1]),
                                            ('time', t.compute()[:-1])],
                                    # last element dropped because costly function uses np.diff()
                                    name='wavelet')

        # wavelet_xarr = wavelet_xarr.chunk({'scales': len(scales), 'time': 12800 * 60})
        return wavelet_xarr

    def dask_wavelet_description(self):

        # generate dask dataframe
        wavelet_xarr = self.dask_wavelet_init()
        if self.client is None:
            raise ValueError(f"self.client should be passed!")
        _wavelet_xarr = self.client.persist(wavelet_xarr)
        _mean = _wavelet_xarr.mean(dim='time')
        _std = _wavelet_xarr.std(dim='time')
        _max = _wavelet_xarr.max(dim='time')
        _min = _wavelet_xarr.min(dim='time')
        desc_df = pd.DataFrame({'mean': _mean,
                                'std': _std,
                                'max': _max,
                                'min': _min})

        desc_df.to_csv(self.wavelet_path / 'desc.csv')

        return desc_df

    def dask_wavelet_compute(self):
        wavelet_xarr = self.dask_wavelet_init()

        cut_no = int(self.attributes['cut_no'])
        plot_height = len(self.scales)
        plot_width = int(
            self.description.loc['count', 'data'] // self.sampling_freq)
        canvas = ds.Canvas(plot_height=plot_height, plot_width=plot_width)
        # raster maps all input points to canvas pixels.
        # computed here because created array will be very small in size.
        print(f"Rastering...[{cut_no}:{plot_height}x{plot_width}]")
        rastered = canvas.raster(wavelet_xarr).compute()

        # ! quadmesh does not support dask objects.
        # qmesh = canvas.quadmesh(xarr, x='time', y='scales', agg=ds.max('wavelet'))
        wavelet_img = tf.shade(rastered, cmap=matplotlib.cm.jet, how='linear')
        wavelet_img.to_pil().save(self.wavelet_path /
                                  f'wavelet_{cut_no}_log_thin.png')

        timeseries_img = self.plot_datashader(
            plot_height=plot_height, plot_width=plot_width)
        timeseries_img.to_pil().save(self.wavelet_path /
                                     f'timeseries_{cut_no}_thin.png')

        return tf.Images(*[wavelet_img, timeseries_img]).cols(1)

    @staticmethod
    def dask_wavelet_calculation(signal, scales, wavelet, sampling_period, shape):

        # Fake wavelet calculation
        @dask.delayed
        def fake_cwt_iter(signal, scales, wavelet, method='conv'):
            return np.random.rand(12800 - 1).astype(np.float32)

        # Wavelet costly part
        @dask.delayed
        def cwt_cost(signal, int_psi_scale, scale):
            conv = np.convolve(signal, int_psi_scale, mode='same')
            conv = - np.sqrt(scale) * np.diff(conv)
            return np.power(np.abs(conv), 2)

        wavelet = pywt.DiscreteContinuousWavelet(wavelet)
        int_psi, x = pywt.integrate_wavelet(wavelet, precision=10)
        int_psi = np.conj(int_psi)
        # convert int_psi, x to the same precision as the data
        int_psi = int_psi.astype(np.complex64)
        x = x.astype(np.float32)

        frequencies = pywt.scale2frequency(
            wavelet, scales, 10) / sampling_period

        int_psi_scales = list()
        for i, scale in enumerate(scales):
            step = x[1] - x[0]
            j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step)
            j = j.astype(int)  # floor
            if j[-1] >= int_psi.size:
                j = dask.delayed(np.extract)(j < int_psi.size, j)
            int_psi_scale = int_psi[j][::-1]
            int_psi_scales.append(int_psi_scale)

        powers = list()
        for i, int_psi_scale in enumerate(int_psi_scales):
            # costly part called here!
            power = cwt_cost(signal, int_psi_scale, scales[i])

            # change it to array, we know shape of N
            power = da.from_delayed(power, shape=(
                shape[0] - 1,), dtype=np.float32)
            powers.append(power)
        powers = da.stack(powers)
        return powers, frequencies

    def timeseries_matplotlib_image(self, datashader_img, start=0, stop=None):
        stop = stop or self.description.loc['count', 'data']
        mpl_image = matplotlib.image.pil_to_array(datashader_img.to_pil())
        fig, ax = plt.subplots(figsize=(12, 6))

        ts = start // self.sampling_freq  # tick start
        te = stop // self.sampling_freq  # tick end
        td = (te - ts) / 8  # tick interval
        ax.set_xticklabels(np.arange(ts - td, te + 2 * td, td).astype(np.int))

        _min = self.description.loc['min', 'data']
        _max = self.description.loc['max', 'data']
        ts = -max(abs(_min), _max)
        te = max(abs(_min), _max)
        td = 2 * te / 5

        print(ts, te, td)
        print(
            [f"{2 * labelf:.2f}" for labelf in np.arange(ts - td, te + 2 * td, td)[::-1]])
        ax.set_yticklabels(
            [f"{2 * labelf:.2f}" for labelf in np.arange(ts - td, te + 2 * td, td)[::-1]])
        ax.imshow(mpl_image)
        return ax

    @staticmethod
    def normalized(img):
        _min = img.min().min()
        _max = img.max().max()
        return (img - _min) / (_max - _min)

    @staticmethod
    def reversed(img):
        return img.max().max() - img

    @staticmethod
    def scaled(img, img_desc, ref_desc):
        coef = ref_desc.T['mean'] / img_desc.T['mean']
        # print(f"Scale-coef:{coef.mean()}")
        return coef * img

    def normalize(self, global_wavelet_description, inplace=False):
        norm_img = (self.wavelet_image - global_wavelet_description.loc['min', :]) / (
            global_wavelet_description.loc['max', :] - global_wavelet_description.loc['min', :])
        if inplace:
            self.wavelet_image = norm_img
        return norm_img

    def standardize(self):
        pass

    def __str__(self):
        return f"""
                Path:{self.path}
                Shape:{self.reading.shape}
                RPM:{self.rpm}
                Sampling Frequency:{self.sampling_freq}
                Number of Cycle:{self.n_cycle}
                Number of Point in 1 Cycle:{self.num_points_in_one_period}
                T:{self.T}
                dt tooth: {self.dt_tooth * 1000} ms
                expected_tooth_freq : {self.expected_tooth_freq} Hz
                -----------------------------------------
                Shape/Sampling Frequency =?: Elapsed Time -> {self.reading.shape[0] / self.sampling_freq} =? {self.reading['time'].iloc[-1]}
                
                
                """


class ToolwearBag:
    def __init__(self, root, client=None):

        self.client = client

        self.root = root
        self.root_parquet_path = self.root / 'parquet'

        self.cut_nos = [int(file) for file in os.listdir(self.root_parquet_path) if
                        os.path.isdir(self.root_parquet_path / file)]

        self.parquet_paths = [self.root_parquet_path /
                              str(cut_no) for cut_no in self.cut_nos]

        self.datasets = {cut_no: Toolwear.from_parquet(root=self.root, client=self.client, cut_no=cut_no)
                         for cut_no in self.cut_nos}

        # assign first wavelet description as reference
        self.ref_wavelet_desc = self.datasets[self.cut_nos[0]
                                              ].wavelet_description
        self.global_wavelet_description = self.compute_global_wavelet_description()

    @property
    def scales(self):
        return np.arange(1, 129)  # todo: do not hard-code

    def compute_global_wavelet_description(self):

        # num_cuts x [mean, std, max, min] x scales
        description_ndarr = np.zeros((len(self.cut_nos), 4, len(self.scales)))
        for i, cut_no in enumerate(self.cut_nos):
            description_ndarr[i, :,
                              :] = self.datasets[cut_no].wavelet_description.values

        return pd.DataFrame({'mean': np.mean(description_ndarr[:, 0, :], axis=0),  # 0:mean, axis=0 -> num_cuts
                             # 1:std, axis=0 -> num_cuts
                             'std': np.mean(description_ndarr[:, 1, :], axis=0),
                             # 2:max, axis=0 -> num_cuts
                             'max': np.max(description_ndarr[:, 2, :], axis=0),
                             # 3:min, axis=0 -> num_cuts
                             'min': np.min(description_ndarr[:, 3, :], axis=0),
                             }).T

    # def normalize(self):
    #     for cut_no, dataset in self.datasets.items():
    #         dataset.normalize(self.global_wavelet_description)

    # def standardize(self):
    #     pass

    def __getitem__(self, key):
        return self.datasets[key]

    def plot_wavelet_vs_toolwear(self, scaled=True, cbar=True, vmin=0., vmax=1.):

        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 20))
        for i, (cut_no, dataset) in enumerate(self.datasets.items()):

            img = dataset.reversed(dataset.normalized(dataset.wavelet_image))
            if scaled:
                img = dataset.scaled(img=img,
                                     img_desc=dataset.wavelet_description,
                                     ref_desc=self.ref_wavelet_desc)

            sns.heatmap(img.T, vmin=vmin, vmax=vmax, ax=axes[i, 0], cbar=cbar)
            axes[i, 1].plot(dataset.wavelet_label,
                            label=f'Cut No {cut_no} - Cutting Speed: {dataset.attributes["cutting_speed"]}')
            axes[i, 1].legend()

    def plot_toolwear(self, vmax=400):
        fig, axes = plt.subplots(
            nrows=1, ncols=1, figsize=(12, 6), squeeze=False)
        for i, (cut_no, dataset) in enumerate(self.datasets.items()):
            attr = [t for t in dataset.attributes['toolwear'] if t < vmax]

            axes[0, 0].plot(dataset.attributes['cutting_length'][:len(attr)], dataset.attributes['toolwear'][:len(attr)], '-*',
                            label=f'Cut No {cut_no} - Cutting Speed: {dataset.attributes["cutting_speed"]}')
            axes[0, 0].legend()

    def plot_holoviews(self):
        images = dict()
        for i, (cut_no, dataset) in enumerate(self.datasets.items()):
            img = dataset.plot_holoviews(plot_height=300, plot_width=800,
                                         xname='cutting_length', yname='data')

            images[cut_no] = img

        return (images[13] + images[14] + images[15] + images[16]).cols(1)

    def to_torch_wavelet_dataset(self, seq_len, train_cuts=(14, 15, 16), test_cut=13):

        # ! normalized -> reversed -> scaled image
        images = {cut_no: dataset.scaled(img=dataset.reversed(dataset.normalized(dataset.wavelet_image)),
                                         img_desc=dataset.wavelet_description,
                                         ref_desc=self.ref_wavelet_desc)
                  for cut_no, dataset in self.datasets.items()}

        targets = {cut_no: dataset.wavelet_label / 1000.
                   for cut_no, dataset in self.datasets.items()}
        auxs = {cut_no: (dataset.wavelet_aux - 73.) / (130. - 73.)  # todo: do not hard-code
                for cut_no, dataset in self.datasets.items()}
        wavelet_descriptions = {cut_no: dataset.wavelet_description
                                for cut_no, dataset in self.datasets.items()}

        return ToolwearWaveletDataset(root=self.root,
                                      data={
                                          'train_cut_no': train_cuts,
                                          'test_cut_no': test_cut,
                                          'images': images,
                                          'targets': targets,
                                          'auxs': auxs,
                                          'wavelet_description': wavelet_descriptions
                                      },
                                      seq_len=seq_len)


class ToolwearWaveletDataset:

    def __init__(self, root, data, seq_len):
        self.root = root

        self.train_cut_nos = data['train_cut_no']
        self.test_cut_no = data['test_cut_no']
        self.cut_nos = list(self.train_cut_nos) + [self.test_cut_no]

        self.seq_len = seq_len
        self.images = data['images']
        self.targets = data['targets']
        self.auxs = data['auxs']
        self.wavelet_descriptions = data['wavelet_description']

        self.train_datasets = list()
        for cut_no in self.train_cut_nos:
            img = self.images[cut_no]
            target = self.targets[cut_no]
            aux = self.auxs[cut_no]

            dataset = ToolwearWaveletInnerDataset(data=img, targets=target,
                                                  seq_len=self.seq_len,
                                                  aux=aux)
            self.train_datasets.append(dataset)

        self.trainset = ConcatDataset(self.train_datasets)

        self.testset = ToolwearWaveletInnerDataset(data=self.images[self.test_cut_no],
                                                   targets=self.targets[self.test_cut_no],
                                                   seq_len=self.seq_len,
                                                   aux=self.auxs[self.test_cut_no])


class ToolwearWaveletInnerDataset(Dataset):
    def __init__(self, data, targets, aux, seq_len=60):
        self.data = data.values
        self.targets = targets
        self.seq_len = seq_len
        self.aux = aux
        print(self.data.shape)

        _shape = len(self.data)
        aug_start_ix = int(_shape * 0.75)

        # # ! regular last part copy augmentation
        # aug_data = self.data[aug_start_ix:, :]
        # self.data = np.concatenate((self.data, aug_data, aug_data))
        # aug_targets = targets[aug_start_ix:]
        # self.targets = np.concatenate((self.targets, aug_targets, aug_targets))

        # # ! expand last part
        # aug_data = self.data[aug_start_ix:, :].repeat(5, axis=0)
        # self.data = self.data[:aug_start_ix]
        # self.data = np.concatenate((self.data, aug_data))
        #
        # aug_targets = targets[aug_start_ix:].repeat(5)
        # self.targets = self.targets[:aug_start_ix]
        # self.targets = np.concatenate((self.targets, aug_targets))

        print()

    def __getitem__(self, ix):
        seq, target = self.data[ix:ix + self.seq_len,
                                :], self.targets[ix + self.seq_len]
        seq = (seq - seq.mean()) / (seq.std())
        return (torch.from_numpy(seq),  # data
                torch.from_numpy(np.array([target], dtype=float)),  # label
                torch.from_numpy(np.array([self.aux], dtype=float)),  # aux
                )

    def __len__(self):
        # todo: data has 2 more item than label. need to investigate
        return len(self.data) - self.seq_len - 5


# class ToolwearTorchInnerDataset(Dataset):
#     def __init__(self, vibration, toolwear, seq_len):
#         self.data = vibration.reshape(-1, 1)
#         self.labels = toolwear.reshape(-1, 1)
#         self.seq_len = seq_len
#
#     def __len__(self):
#         return self.data.shape[0] - self.seq_len
#
#     def __getitem__(self, ix):
#         # x : [batch, seq, feature]
#         # y : [batch, seq]
#
#         if isinstance(ix, slice):
#             xs, ys = [], []
#             for ii in range(*ix.indices(len(self))):  # *ix.indices(len(self)): (start, stop, step)
#                 x, y = self[ii]
#                 xs.append(x)
#                 ys.append(y)
#             return torch.stack(xs), torch.stack(ys)
#         return (torch.DoubleTensor(self.data[ix:ix + self.seq_len, :]),
#                 torch.DoubleTensor(self.labels[ix + self.seq_len, :]))


# class ToolwearTorchDataset:
#     PATH = Path("D:/YandexDisk/machining/data/raw/1/data-timeseries.csv")
#
#     def __init__(self, seq_length, train_split, cut_lim=None):
#         self.seq_length = seq_length
#         self.train_split = train_split
#
#         timeseries = pd.read_csv(self.PATH).values
#
#         # sub-sample some data
#         if cut_lim is None:
#             bottom, top = (0.650, 0.660)
#         else:
#             bottom, top = cut_lim
#         timeseries = timeseries[int(timeseries.shape[0] * bottom):int(timeseries.shape[0] * top), :]
#
#         self.scaler = MinMaxScaler()
#         self.scaler.fit(timeseries)
#
#         timeseries = self.transform(timeseries)
#
#         print(timeseries.max(), timeseries.min())
#
#         self.train_size = int(timeseries.shape[0] * self.train_split)
#         self.test_size = timeseries.shape[0] - self.train_size
#
#         self.trainset = ToolwearTorchInnerDataset(vibration=timeseries[:self.train_size, 0],
#                                                   toolwear=timeseries[:self.train_size, 1],
#                                                   seq_len=self.seq_length)
#         self.testset = ToolwearTorchInnerDataset(vibration=timeseries[self.train_size:, 0],
#                                                  toolwear=timeseries[self.train_size:, 1],
#                                                  seq_len=self.seq_length)
#
#     def plot(self):
#         # x, y = self.trainset[:1000]
#         # y = y[:, -1, :]
#
#         plt.plot(self.trainset.labels, label='y')
#         plt.legend()
#         plt.show()
#
#     def transform(self, x):
#         return self.scaler.transform(x)
#
#     def inverse_transform(self, x):
#         return self.scaler.inverse_transform(x)


# class ToolwearWaveletConcatDataset:
#     def __init__(self, seq_len):
#         self.seq_len = seq_len
#
#         self.train_datasets = list()
#         for cutno in [14, 15, 16]:
#             dataset = ToolwearWaveletDataset(cutno=cutno, seq_len=self.seq_len)
#             self.train_datasets.append(dataset.trainset)
#
#         self.trainset = ConcatDataset(self.train_datasets)
#         self.testset = ToolwearWaveletDataset(cutno=13, seq_len=self.seq_len).trainset


if __name__ == "__main__":
    # for cut_no in [16]:
    #     vib = Toolwear.raw_batch_read(root=Path('D:/YandexDisk/machining/data'),
    #                                   cut_no=cut_no, kind='acc', n_cycle=200)

    dataset = ToolwearBag(
        root=Path('D:/YandexDisk/machining/data')).to_torch_wavelet_dataset(seq_len=60)
    print(dataset.trainset[3])
