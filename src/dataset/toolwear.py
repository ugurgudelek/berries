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
from ipywidgets import interactive, HBox, VBox, widgets, Output, interact, interact_manual, Layout, Box
# py.init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Deep learning libraries
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Local libraries
from dataset.generic import Standardizer, TimeSeriesDatasetWrapper


class ExcelData:
    """
    In order to read cutting dataset excel
    """
    EXCEL_PATH = Path('D:/machining/data/raw/Experimental Data Sheet.xlsx')

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
        self.subset = None

        if not fake:
            # read raw tdms data
            self.reading: pd.DataFrame = TdmsFile(path).as_dataframe()
            self.reading = self.reading.rename(
                columns={self.reading.columns[0]: "data"})

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
        return 1 / self.sampling_freq

    @property
    def dt_tooth(self):
        return 1 / ((self.rpm / 60) * self.n_flute)

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
        self.subset = self.reading.iloc[start_idx:start_idx + self.num_points_in_one_period * self.n_cycle, :]
        return self.subset

    def static_plot(self, save=True):
        plt.figure(figsize=(20, 10))
        plt.plot(self.reading['time'], self.reading['data'], '.')
        ax = plt.gca()
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

        fig = go.Figure(data=[go.Scattergl(x=fft_freqs, y=fft_data, name='fft', mode='markers+lines')],
                        layout=go.Layout(xaxis={'title': 'Freq(Hz)'},
                                         yaxis={'title': 'Amplitude'},
                                         template=Toolwear.PLOT_TEMPLATE))
        if save:
            self.save_fig(fig, suffix='fft', kind='plotly')
        if plot:
            fig.show()

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

    def wavelet(self, data, wavelet='cmor6-1.5', xlim=None, ylim=None, clim=None, save=True):
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

        #         scales = np.arange(1, 500, 1)
        scales = scg.periods2scales(np.logspace(start=0,
                                                stop=2.5,
                                                num=200)  # 10**0 -- 10**2
                                    )

        cwt = scg.CWT(time=data['time'].values,
                      signal=data['data'].values - data['data'].mean(),
                      scales=scales,
                      wavelet=wavelet)

        ax = scg.cws(cwt,
                     spectrum='power',
                     figsize=(12, 6),
                     yscale='log',
                     ylabel="Period [seconds]",
                     xlabel='Time [seconds]',
                     xlim=xlim, ylim=ylim, clim=clim)

        text = ax.annotate("found freq",
                           xy=(data['time'].iloc[100], 1 / self.expected_tooth_freq),
                           xytext=None,
                           bbox=dict(boxstyle="round", facecolor="y", edgecolor="0.5"),
                           arrowprops=dict(facecolor='yellow', shrink=0.05))

        text = ax.annotate(f"Ref freq x2.0:{self.expected_tooth_freq * 2}",
                           xy=(data['time'].iloc[100], 1 / (self.expected_tooth_freq * 2)),
                           xytext=None,
                           bbox=dict(boxstyle="round", facecolor="y", edgecolor="0.5"),
                           arrowprops=dict(facecolor='yellow', shrink=0.05))
        text = ax.annotate(f"Ref freq x0.5:{self.expected_tooth_freq * 0.5}",
                           xy=(data['time'].iloc[100], 1 / (self.expected_tooth_freq * 0.5)),
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
        #             fig.show()

        return (ax, cwt)

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
        for slot_no, path in tqdm(enumerate((fpath / str(cut_no)).glob(f'{kind}*.tdms'), start=1)):
            vib_current = Toolwear(path=path,
                                   n_cycle=n_cycle,
                                   attributes=exceldata.get_attribute(cut_no=cut_no, slot_no=slot_no, kind=kind))

            if vib is None:
                vib = vib_current
            else:
                vib.append(vib_current)

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


class ToolwearTorchDataset:
    CSV_PATH = Path('../input/machining/tool_wear/tool4/raw/data.csv')
    PICKLE_FPATH = Path('../input/machining/tool_wear/tool4/labeled')

    def __init__(self, data, label, train, scaler, params, hyperparams):
        self.params = params
        self.hyperparams = hyperparams
        self.data = data
        self.label = label

        self.train = train
        self.scaler = scaler
        self.augment = params['augment'] if train else False

        self.preprocessing()  # augment data and labels + applies scaler

        self.seq_len = self.hyperparams['seq_len']
        self.batch_size = self.hyperparams['train_batch_size']
        self.input_dim = 1
        self.time_skip = self.data.size(0) // self.batch_size
        self.data = self.data.narrow(0, 0, self.time_skip * self.batch_size)
        self.batched_data = self.data.contiguous().view(self.batch_size, -1, self.input_dim).transpose(0, 1)

    @staticmethod
    def augmentation(data, label, std, noise_ratio=0.05, noise_interval=0.0005, max_length=100000):
        noiseSeq = torch.randn(data.size())
        augmentedData = data.clone()
        augmentedLabel = label.clone()
        for i in np.arange(0, noise_ratio, noise_interval):
            scaled_noiseSeq = noise_ratio * std.expand_as(data) * noiseSeq
            augmentedData = torch.cat([augmentedData, data + scaled_noiseSeq], dim=0)
            augmentedLabel = torch.cat([augmentedLabel, label])
            if len(augmentedData) > max_length:
                augmentedData = augmentedData[:max_length]
                augmentedLabel = augmentedLabel[:max_length]
                break

        return augmentedData, augmentedLabel

    def preprocessing(self):
        if self.train:
            # Train the scaler
            self.scaler.fit(self.data)
            # Augment the data
            if self.augment:
                self.data, self.label = self.augmentation(self.data, self.label, std=self.scaler.std)

        # Apply standardization or normalization
        self.data = self.scaler.transform(self.data)

    @classmethod
    def from_pickle(cls, train=True, **kwargs):
        path = cls.PICKLE_FPATH / ('train' if train else 'test') / 'toolwear.pkl'

        def load_data(path):
            with open(str(path), 'rb') as f:
                df = pickle.load(f)
                # label = torch.FloatTensor(df.loc[:, 'wear_len'].values)
                # data = torch.FloatTensor(df.loc[:, 'acc'].values.reshape(-1, 1))
                df.index = pd.to_datetime(df.index)
                label = torch.FloatTensor(df.loc[:, 'wear_len'].resample('1S').mean().values)
                data = torch.FloatTensor(df.loc[:, 'acc'].resample('1S').std().values.reshape((-1, 1)))
                # data = torch.FloatTensor(df.loc[:, 'acc'].resample('1S').std().diff(1).fillna(0).values.reshape((-1, 1)))
            return data, label

        data, label = load_data(path)
        return cls(data=data, label=label, train=train, **kwargs)

    @classmethod
    def from_file(cls, train=True, **kwargs):
        # 
        #         data = self.raw_data['acc'].values.astype(float)
        #         targets = self.raw_data['wear_len'].values.astype(float)
        raw_path = cls.CSV_PATH
        raw_data = pd.read_csv(raw_path, index_col=0)
        train_path = raw_path.parent.parent.joinpath('labeled', 'train', 'toolwear.pkl').with_suffix('.pkl')

        train_size = int(raw_data.shape[0] * 0.8)
        train_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(train_path), 'wb') as pkl:
            pickle.dump(raw_data[:train_size], pkl)

        test_path = raw_path.parent.parent.joinpath('labeled', 'test', 'toolwear.pkl').with_suffix('.pkl')
        test_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(test_path), 'wb') as pkl:
            pickle.dump(raw_data[train_size:], pkl)

        return cls.from_pickle(train, **kwargs)


def toolwear_class_usage():
    # ======= DATA READ ==========
    NUM_CYCLE = 10
    vib = Toolwear.batch_read(fpath=Path('D:/machining/data/raw'), cut_no=1, kind='acc', n_cycle=NUM_CYCLE)
    # vib = Toolwear.from_pickle(Path('D:/machining/data/raw/1/data.pickle'))

    # ======= DATA PLOT ==========
    START_SEC = 5000
    # vib.apply_aggregation() # apply several filters
    subset = vib.make_subset_after(sec=START_SEC)
    vib.to_pickle()

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
    # (ax,cwt) = vib.wavelet(data=subset, wavelet='cmor3-1.5', clim=[0, 0.1])

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
    import os
    os.chdir('..')
    scaler = Standardizer()

    # ToolwearTorchDataset.from_file()
    train_dataset = ToolwearTorchDataset.from_pickle(train=True, scaler=scaler)
    test_dataset = ToolwearTorchDataset.from_pickle(train=False, scaler=scaler)

    dataset = TimeSeriesDatasetWrapper(trainset=train_dataset,
                                       testset=test_dataset)


if __name__ == "__main__":
    pytorch_dataset_usage()
