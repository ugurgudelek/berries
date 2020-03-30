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
from ipywidgets import interactive, HBox, VBox, widgets, Output, interact, interact_manual, Layout, Box
# py.init_notebook_mode(connected=True)
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection, LineCollection
import matplotlib.ticker as ticker

# plt.style.use('seaborn-whitegrid')

# Deep learning libraries
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Local libraries
from dataset.generic import Standardizer, TimeSeriesDatasetWrapper
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import gc
import multiprocessing
import itertools


# from utils.plot_utils import camera_ready_matplotlib_style


class ExcelData:
    """
    In order to read cutting dataset excel
    """
    EXCEL_PATH = Path('D:/YandexDisk/machining/data/raw/Experimental Data Sheet.xlsx')

    def __init__(self):
        self.excel = pd.read_excel(self.EXCEL_PATH)
        self.excel = self.excel.drop([0], axis=0).reset_index(drop=True)  # drop mm|m/min like that row

    def get_attribute(self, cut_no, slot_no, kind):
        subset = self.excel.loc[
            (self.excel['cut_no'] == cut_no) & (self.excel['slot_no'] == slot_no) & (self.excel['kind'] == kind)]
        attr = subset.iloc[0].to_dict()
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

    def __init__(self, path, attributes, n_cycle=3, start_sec=3, fake=False, add_noise=False):
        self.path = path
        self.attributes = attributes
        self.n_cycle = n_cycle
        self.start_sec = start_sec
        self.add_noise = add_noise

        if not fake:
            # read raw tdms data
            self.reading: pd.DataFrame = TdmsFile(path).as_dataframe()
            self.reading = self.reading.rename(
                columns={self.reading.columns[0]: "data"})

            self.reading['raw_data'] = self.reading['data'].copy()

            # add time column
            self.reading['time'] = np.arange(self.__len__()) * self.T

            # add cutting_length column
            self.reading['cutting_length'] = np.nan
            self.reading['cutting_length'].iloc[-1] = self.attributes['cutting_length']

            # add toolwear column
            self.reading['toolwear'] = np.nan
            self.reading['toolwear'].iloc[-1] = self.attributes['toolwear']

            # add ref signal: signal_freq*2 Hz
            # self.reading['data'] += 0.01*np.sin(2*np.pi*self.expected_tooth_freq*2*self.reading['time'])

            # add ref signal: signal_freq*0.5 Hz
            # self.reading['data'] += 0.005*np.sin(2*np.pi*self.expected_tooth_freq*0.5*self.reading['time'])
        else:
            self.create_fake_data()

        # create aggregation columns
        # self.apply_aggregation()

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

    @property
    def T(self):
        """
        T: interval : 1/Fs
        :return:
        """
        return 1 / self.sampling_freq

    @property
    def dt_tooth(self):
        return 1 / ((self.rpm / 60) * self.n_flute)

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
            t = np.arange(0, subset['time'].values.shape[0]) if _islist else subset['time'].values
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

                pos_threshold = pos_envelope + (threshold_epsilon + threshold_mul * data[pos_index].mean())
                neg_threshold = neg_envelope - (threshold_epsilon - threshold_mul * data[neg_index].mean())

                nan_valid_bool[pos_index] = (data[pos_index] > pos_threshold)
                nan_valid_bool[neg_index] = (data[neg_index] < neg_threshold)

                nan_valid_index = np.where(nan_valid_bool)

                drop_count = np.count_nonzero(nan_valid_bool)
                drop_ratio = np.count_nonzero(nan_valid_bool) / nan_valid_bool.shape[0]
                if verbose:
                    print(f'Pos_mean:{data[pos_index].mean()}')
                    print(f'Neg_mean:{data[neg_index].mean()}')
                    print(f'Pos_threshold_coef:{(threshold_epsilon + threshold_mul * data[pos_index].mean())}')
                    print(f'Neg_threshold_coef:{(threshold_epsilon - threshold_mul * data[neg_index].mean())}')
                    print(f'Dropped data number:{drop_count} (%{drop_ratio:.5f})')

                axes[i if _islist else 0, 0 if _islist else i].scatter(t[nan_valid_index], data[nan_valid_index],
                                                                       label=f'dropped {drop_count}(%{drop_ratio})',
                                                                       alpha=1., c='crimson', s=10, marker='+')
                axes[i if _islist else 0, 0 if _islist else i].plot(t[pos_index], pos_threshold, label='pos_threshold',
                                                                    alpha=.5, c='r', linewidth=0.7, linestyle='dashed')
                axes[i if _islist else 0, 0 if _islist else i].plot(t[neg_index], neg_threshold, label='neg_threshold',
                                                                    alpha=.5, c='r', linewidth=0.7, linestyle='dashed')

            elif method == 'envelope_mean':
                threshold = Toolwear.envelope(subset).mean() * threshold_mul + threshold_epsilon
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

                axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), fancybox=True, shadow=True, ncol=5)

                if save:
                    os.makedirs('denoised', exist_ok=True)
                    plt.figure(fig.number)
                    plt.savefig(filename or f'denoised/denoised_{subset["time"].iloc[0]}.jpg')

        if plot:
            plt.close()

        return denoiseds if _islist else denoiseds[0]

    @staticmethod
    def envelope(subsets, plot=False):
        _type = type(subsets)

        subsets = subsets if _type is list else [subsets]

        if plot:
            fig, axes = plt.subplots(nrows=len(subsets), ncols=1, squeeze=False, sharex=True)

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
                axes[i, 0].plot(t[pos_s_index], pos_envelope, '-', label='pos_envelope')
                axes[i, 0].plot(t[neg_s_index], neg_envelope, '-', label='neg_envelope')
                axes[i, 0].axhline(y=pos_envelope.mean(), c='r', linestyle='dashed',
                                   label='pos_env.mean')
                axes[i, 0].axhline(y=neg_envelope.mean(), c='r', linestyle='dashed',
                                   label='neg_env.mean')

                axes[i, 0].axhline(y=0, c='k', linestyle='dashed')

                ylim = max(pos_envelope.max(), -neg_envelope.min())
                axes[i, 0].set_ylim(-ylim, ylim)
                fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper left')
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

    @property
    def expected_tooth_freq(self):
        return self.n_flute * self.rpm / 60

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

    def static_plot(self, subset=None, bound=None, figsize=None, save=False):

        if subset is None:
            subset = self.reading
        if bound is not None:
            subset = subset.iloc[
                     int(self.sampling_freq * bound[0]):int(self.sampling_freq * bound[1]), :]

        plt.figure(figsize=figsize)
        ax = plt.plot(subset['time'], subset['data'], '.')

        if save:
            self.save_fig(ax.get_figure(), suffix='signal-full', kind='matplotlib')

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

            @interact(xaxis_range=widgets.FloatRangeSlider(min=subset['time'].iloc[0],
                                                           max=subset['time'].iloc[-1],
                                                           step=(
                                                                        subset['time'].iloc[-1] - subset['time'].iloc[
                                                                    0]) / self.sampling_freq,
                                                           description='Time',
                                                           continuous_update=False,
                                                           readout=True,
                                                           readout_format='.3f',
                                                           layout=Layout(align_items='center',
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

        control_widget = interactive(update_subset,
                                     start_sec=widgets.BoundedFloatText(value=subset['time'].iloc[0],  # means start_sec
                                                                        min=self.reading['time'].iloc[0],
                                                                        max=self.reading['time'].iloc[-1],
                                                                        step=0.5,
                                                                        description='Start Second:',
                                                                        disabled=False,
                                                                        layout=Layout(align_items='center',
                                                                                      width='50%',
                                                                                      )
                                                                        ))

        form = Box((control_widget, fig),
                   layout=Layout(display='flex',
                                 flex_flow='column',
                                 border='solid 2px',
                                 align_items='stretch',
                                 width='100%'))

        if save:
            self.save_fig(fig, suffix='signal', kind='plotly')
        # if plot:
        # display(form)

    def save_fig(self, fig, suffix, kind='plotly'):
        # path : 'D:/machining/wear_data/raw/1/acc7000.tdms'

        try:
            filename = self.path.stem  # 'acc7000'
            foldername = self.path.parent.name  # '1'
        except:
            print("You cannot save because path is not valid. This is probably because you call 'append' function.")
            return

        images_path = self.path.parent.parent.parent / 'images'  # 'D:/machining/wear_data/images'
        specific_folder_path = images_path / foldername  # 'D:/machining/wear_data/images/1'
        os.makedirs(specific_folder_path, exist_ok=True)

        save_path = specific_folder_path / filename  # 'D:/machining/wear_data/images/1/acc7000.png

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

    @property
    def num_points_in_one_period(self):
        """
        Return the number of data points for 1 revolution.
        """
        return int(self.sampling_freq / (self.rpm / 60))

    @property
    def num_points_per_tooth(self):
        return self.num_points_in_one_period // self.n_flute

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
                plot_func='own',
                fpath='wavelet-results',
                cwt_save=True,
                plot=True,
                ):  # plot_func can be 'lib' to use more robust scaleogram plot
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
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=None, squeeze=False, frameon=False, sharex=True)
            # Vibration subplot
            time_plot = axes[1, 0].plot(cwt.time, cwt.signal)
            axes[1, 0].set_ylabel('Magnitude')

            if plot_func == "own":

                # Wavelet subplot
                z = np.abs(cwt.coefs).astype(np.float)
                qmesh = axes[0, 0].pcolormesh(cwt.time, range(cwt.scales_freq.shape[0]), z)
                axes[0, 0].set_ylabel('Frequency')
                colorbar = plt.colorbar(qmesh, orientation='vertical', ax=axes[0, 0])

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




    def append(self, other):

        ## UPDATE READING
        _other_reading = other.reading.copy()

        # update index
        interval = self.reading.index[1] - self.reading.index[0]
        elapsed_time = self.reading.index[-1] - \
                       self.reading.index[0] + interval
        _other_reading.index += elapsed_time

        # update time column
        interval = self.reading.loc[:, 'time'].iloc[1] - self.reading.loc[:, 'time'].iloc[0]
        elapsed_time = self.reading.loc[:, 'time'].iloc[-1] - self.reading.loc[:, 'time'].iloc[0] + interval
        _other_reading['time'] += elapsed_time

        # update tdms data
        self.reading = self.reading.append(_other_reading, verify_integrity=True)

        # UPDATE ATTRIBUTES
        for attr_name, attr_val in self.attributes.items():
            if type(attr_val) is list:
                self.attributes[attr_name].append(other.attributes[attr_name])
            else:
                if attr_val != other.attributes[attr_name]:  # create new list
                    self.attributes[attr_name] = [attr_val, other.attributes[attr_name]]
                # else continue - if they are same value, no need to do anything           

        # UPDATE PATH
        # todo: make it correct.
        self.path = Path(os.path.commonpath((self.path, other.path)))

    @staticmethod
    def batch_read(fpath, cut_no, kind, n_cycle):

        # read excel file for attributes
        exceldata = ExcelData()

        vib = None
        tdms_glob = list((fpath / str(cut_no)).glob(f'{kind}*.tdms'))
        with tqdm(total=len(tdms_glob), desc='Batch File Read') as pbar:
            for slot_no, path in enumerate(tdms_glob, start=1):
                vib_current = Toolwear(path=path,
                                       n_cycle=n_cycle,
                                       attributes=exceldata.get_attribute(cut_no=cut_no, slot_no=slot_no, kind=kind))

                if vib is None:
                    vib = vib_current
                else:
                    vib.append(vib_current)

                pbar.update(1)

        vib.interpolate('cutting_length')
        vib.interpolate('toolwear')
        return vib

    @property
    def rpm(self):
        return self.attributes['spindle_speed']

    @property
    def n_flute(self):
        return self.attributes['n_flute']

    @property
    def sampling_freq(self):
        return self.attributes['sampling_freq']

    def interpolate(self, yname='cutting_length'):
        """ 
        Call this function after total merge!
        """

        xname = 'time'

        # quick fix for propor interpolation (to not go below 0)
        self.reading[yname].iloc[0] = 0.
        self.reading[xname].iloc[0] = 0.

        y = self.reading[yname].dropna()
        x = self.reading.loc[y.index, xname]

        interp_f = interpolate.interp1d(x.values, y.values, fill_value="extrapolate")
        new_x = self.reading[xname].values
        self.reading[yname] = interp_f(new_x)

    def plot_toolwear(self):
        plt.plot(self.reading['cutting_length'], self.reading['toolwear'])
        plt.xlabel('Cutting Length')
        plt.ylabel('Toolwear')

    def to_pickle(self):
        with open(f'{self.path}/data.pickle', 'wb') as f:
            pickle.dump(self, f)

    def to_torch_csv(self):
        path = f'{self.path}/data-timeseries.csv'
        self.reading.loc[int(self.reading.shape[0] * 0.):, ['data', 'toolwear']].to_csv(path, index=False)

    @staticmethod
    def from_pickle(path):
        return pickle.load(open(path, 'rb'))

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
                fft_data, fft_freqs = self.fft(data=subset, plot=False, save=False)
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
                raise NotImplementedError(f"{kind} not implemented for 3d plot.")

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


class ToolwearTorchInnerDataset(Dataset):
    def __init__(self, vibration, toolwear, seq_len):
        self.data = vibration.reshape(-1, 1)
        self.labels = toolwear.reshape(-1, 1)
        self.seq_len = seq_len

    def __len__(self):
        return self.data.shape[0] - self.seq_len

    def __getitem__(self, ix):
        # x : [batch, seq, feature]
        # y : [batch, seq]

        if isinstance(ix, slice):
            xs, ys = [], []
            for ii in range(*ix.indices(len(self))):  # *ix.indices(len(self)): (start, stop, step)
                x, y = self[ii]
                xs.append(x)
                ys.append(y)
            return torch.stack(xs), torch.stack(ys)
        return (torch.DoubleTensor(self.data[ix:ix + self.seq_len, :]),
                torch.DoubleTensor(self.labels[ix + self.seq_len, :]))


class ToolwearTorchDataset:
    PATH = Path("D:/YandexDisk/machining/data/raw/1/data-timeseries.csv")

    def __init__(self, seq_length, train_split, cut_lim=None):
        self.seq_length = seq_length
        self.train_split = train_split

        timeseries = pd.read_csv(self.PATH).values

        # sub-sample some data
        if cut_lim is None:
            bottom, top = (0.650, 0.660)
        else:
            bottom, top = cut_lim
        timeseries = timeseries[int(timeseries.shape[0] * bottom):int(timeseries.shape[0] * top), :]

        self.scaler = MinMaxScaler()
        self.scaler.fit(timeseries)

        timeseries = self.transform(timeseries)

        print(timeseries.max(), timeseries.min())

        self.train_size = int(timeseries.shape[0] * self.train_split)
        self.test_size = timeseries.shape[0] - self.train_size

        self.trainset = ToolwearTorchInnerDataset(vibration=timeseries[:self.train_size, 0],
                                                  toolwear=timeseries[:self.train_size, 1],
                                                  seq_len=self.seq_length)
        self.testset = ToolwearTorchInnerDataset(vibration=timeseries[self.train_size:, 0],
                                                 toolwear=timeseries[self.train_size:, 1],
                                                 seq_len=self.seq_length)

    def plot(self):
        # x, y = self.trainset[:1000]
        # y = y[:, -1, :]

        plt.plot(self.trainset.labels, label='y')
        plt.legend()
        plt.show()

    def transform(self, x):
        return self.scaler.transform(x)

    def inverse_transform(self, x):
        return self.scaler.inverse_transform(x)


def toolwear_class_usage():
    # ======= DATA READ ==========
    NUM_CYCLE = 200

    # vib = Toolwear.from_pickle(Path('D:/YandexDisk/machining/data/raw/1/data.pickle'))

    # with tqdm(total=len(subsets), desc='Processing subsets..:') as pbar:

    # ======= DATA PLOT ==========
    # START_SEC = 4000
    # vib.apply_aggregation() # apply several filters
    # subset = vib.make_subset_after(sec=START_SEC)

    # denoised_data = vib.denoise(subsets=subsets[-1], method='basic', threshold_epsilon=1, plot=True)

    # vib.to_pickle()
    # vib.to_torch_csv()

    # # # plot data
    # vib.static_plot()
    # vib.plot(plot=True, save=False)

    # # plot fft
    # vib.fft(data=subset,
    #         plot=True)

    # # plot spectrogram
    # vib.spectrogram(data=subset,
    #                 plot=True)

    # plot wavelet
    # vib.wavelet(data=subset, wavelet='cmor3-1.5', clim=None, batch_size=8000*60)
    vib.wavelet(data=vib.reading, wavelet='cmor3-1.5', clim=None, batch_size=8000 * 23)  # 23 sec data at once

    # vib.plot_toolwear()

    # ======= FAKE DATA ==========
    # NUM_CYCLE =4
    # # read data
    # vib = Toolwear(path='',
    #                     rpm=5*10000, # create (rpm/60)*(n_flute) Hz vibration
    #                     sampling_freq=100000, n_cycle=NUM_CYCLE,
    #                     n_flute=3,
    #                     fake=True, add_noise=False)
    # vib.apply_aggregation() # apply several filters
    # subset = vib.make_subset_after(sec=1)
    # print(vib)
    # # vib.static_plot(save=False)
    # vib.plot(save=False)
    # # vib.fft(data=subset, plot=True, save=False)
    # # vib.spectrogram(data=subset, plot=True, save=False, engine='plotly')
    # ax, cwt = vib.wavelet(data=subset, wavelet='cgau5', save=False)


def pytorch_dataset_usage():
    # ======= PYTORCH DATASET ===========
    dataset = ToolwearTorchDataset(seq_length=8000, train_split=0.95)
    print()


def costly_func(subset):
    denoised_data = Toolwear.denoise(subsets=subset, method='envelope',
                                     threshold_epsilon=0.1, threshold_mul=1.,
                                     plot=True, figsize=(12, 6),
                                     save=True, filename=None,
                                     verbose=False)


if __name__ == "__main__":
    vib = Toolwear.batch_read(fpath=Path('D:/YandexDisk/machining/data/raw'),
                              cut_no=1, kind='acc', n_cycle=200)

    subset_min = vib.reading['time'].iloc[0]  # sec
    subset_max = vib.reading['time'].iloc[-1]  # sec
    subset_len = 1  # sec
    subset_stride = 0.2
    print(f"Subset creation starting: [{subset_min}{subset_max}) stride:{subset_stride} len:{subset_len}")
    subsets = [vib.reading.iloc[vib.second2index(sec0):vib.second2index(sec1), :]
               for sec0, sec1 in zip(np.arange(subset_min, subset_max - subset_len + 1, subset_stride),
                                     np.arange(subset_min + subset_len, subset_max + 1, subset_stride))]

    with multiprocessing.Pool(processes=8) as pool:
        pool.map(costly_func, subsets)

    from utils.plot_utils import image_folder_to_gif

    image_folder_to_gif('denoised', glob='*.jpg')
