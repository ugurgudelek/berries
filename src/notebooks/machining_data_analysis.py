#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[3]:


from nptdms import TdmsFile
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import os
import pickle

from scipy import signal, interpolate
from scipy.signal import butter, lfilter, freqz, savgol_filter

import plotly
import plotly.offline as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.tools as tls
from ipywidgets import interactive, HBox, VBox, widgets, Output, interact, interact_manual, Layout, Box
py.init_notebook_mode(connected=True)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# %matplotlib inline

import scaleogram as scg
import pywt


# # XLSX Data

# In[4]:


class ExcelData():
    EXCEL_PATH = Path('D:/machining/data/raw/Experimental Data Sheet.xlsx')
    def __init__(self):
        self.excel = pd.read_excel(self.EXCEL_PATH)
        self.excel = self.excel.drop([0], axis=0).reset_index(drop=True) # drop mm|m/min like that row
        
    def get_attribute(self, cut_no, slot_no, kind):
        subset = self.excel.loc[(self.excel['cut_no']==cut_no)& (self.excel['slot_no']==slot_no) & (self.excel['kind']==kind)]
        attr = subset.iloc[0].to_dict()
        return attr


# # VibrationData Class

# In[5]:


class VibrationData():
    PLOT_TEMPLATE = 'plotly_white'

    def __init__(self, path, attributes, n_cycle=3, start_sec=3, fake=False, add_noise=False):
        self.path = path
        self.attributes = attributes
        self.n_cycle = n_cycle
        self.start_sec = start_sec
        self.add_noise=add_noise

        
        if fake:
                       
            
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

            skip_sin = 9 # number of skip sine wave

            steps = np.linspace(0, 10, 10*self.sampling_freq)
            # orig = np.sin(2*np.pi*self.expected_tooth_freq*steps)
            s = np.sin(2*np.pi*self.expected_tooth_freq*steps)
            for i in range(0, len(steps), (skip_sin+1)*(self.num_points_per_tooth)):
                s[i:i+self.num_points_per_tooth*(skip_sin)]= 0.
            s[s<0] = 0.
            
            self.reading = pd.DataFrame()
            self.reading['time'] = steps
            self.reading['data'] = s
            self.rpm = self.rpm //(skip_sin+1)
                           
            
            if self.add_noise:
                self.reading['data'] += np.random.randn(self.reading.shape[0])


        else:
            # read raw tdms data
            self.reading: pd.DataFrame = TdmsFile(path).as_dataframe()
            self.reading = self.reading.rename(
                columns={self.reading.columns[0]: "data"})
            
            # add time column
            self.reading['time'] = np.arange(self.__len__())*self.T
            
            # add cutting_length column
            self.reading['cutting_length'] = np.nan
            self.reading['cutting_length'].iloc[-1] = self.attributes['cutting_length']
            
            
            # add toolwear column
            self.reading['toolwear'] = np.nan
            self.reading['toolwear'].iloc[-1] = self.attributes['toolwear']
            
            
            
            
#             # add ref signal: signal_freq*2 Hz
#             self.reading['data'] += 0.01*np.sin(2*np.pi*self.expected_tooth_freq*2*self.reading['time'])

#              # add ref signal: signal_freq*0.5 Hz
#             self.reading['data'] += 0.005*np.sin(2*np.pi*self.expected_tooth_freq*0.5*self.reading['time'])

        # create aggregation columns
#         self.apply_aggregation()

    @property
    def T(self):
        return 1/self.sampling_freq
    
    @property
    def dt_tooth(self):
        return 1/((self.rpm/60)*self.n_flute)

    def apply_aggregation(self):

        # apply savgol filter
        SAVGOL_WINDOW_LEN = int(self.num_points_in_one_period/1)
        SAVGOL_WINDOW_LEN = SAVGOL_WINDOW_LEN if SAVGOL_WINDOW_LEN % 2 == 1 else SAVGOL_WINDOW_LEN+1
        self.reading['savgol'] = savgol_filter(self.reading['data'],
                                               window_length=SAVGOL_WINDOW_LEN,
                                               polyorder=1)
        # apply lowpass filter
        LOWPASS_ORDER = 2
        CUTOFF = 2500  # desired cutoff frequency of the filter, Hz
        self.reading['lowpass'] = self.butter_lowpass_filter(
            self.reading['data'], CUTOFF, self.sampling_freq, LOWPASS_ORDER)

        self.reading['lowpass+savgol'] = savgol_filter(self.reading['lowpass'],
                                                       window_length=SAVGOL_WINDOW_LEN,
                                                       polyorder=1)

    def __len__(self):
        return self.reading.shape[0]
    
    @property
    def expected_tooth_freq(self):
        return (self.n_flute*self.rpm/60)

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
        return self._make_subset(start_idx=int(self.sampling_freq*self.start_sec))

    def _make_subset(self, start_idx):
        return self.reading.iloc[start_idx:start_idx+self.num_points_in_one_period*self.n_cycle, :]

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
                            x0=subset['time'].iloc[self.num_points_in_one_period*i],
                            y0=subset['data'].max(),
                            x1=subset['time'].iloc[self.num_points_in_one_period*i],
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
                                      template=VibrationData.PLOT_TEMPLATE))
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
                                                               subset['time'].iloc[-1] - subset['time'].iloc[0]) / self.sampling_freq,
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
                        s['y0'] = _max*1.1
                        s['y1'] = _min*1.1

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
        if plot:
            display(form)
        
    
    def save_fig(self, fig, suffix, kind='plotly'):
        # path : 'D:/machining/wear_data/raw/1/acc7000.tdms'
            
        try:
            filename = self.path.stem # 'acc7000'
            foldername = self.path.parent.name # '1'
        except:
            print("You cannot save because path is not valid. This is probably because you call 'append' function.")
            return

        images_path = self.path.parent.parent.parent/'images' # 'D:/machining/wear_data/images'
        specific_folder_path = images_path / foldername # 'D:/machining/wear_data/images/1'
        os.makedirs(specific_folder_path, exist_ok=True)

        save_path = specific_folder_path/filename # 'D:/machining/wear_data/images/1/acc7000.png
        
        if kind == 'plotly':
            # fig.write_image(str(save_path.with_suffix('.png')))
            fig.write_html(str(save_path.with_suffix(f'.{suffix}.html')))
        if kind == 'matplotlib':
            plt.savefig(str(save_path.with_suffix(f'.{suffix}.png')))
        
    def fft(self, data, bound=(0, 2500), plot=False, save=True):
        Fs = self.sampling_freq  # sampling freq
        Ts = 1/Fs  # sampling interval
        t = data['time']  # time vector

        n = data.shape[0]  # data lenght
        k = np.arange(n)
        T = n/Fs
        freq = k/T

        S = data['data'].values
        S_fft = np.fft.fft(S)

        freq = freq[:n//2]  # one side frequency range
        freq = freq[freq < bound[1]]  # cut unnecessary frequencies

        S_fft = S_fft.real[:freq.shape[0]] /             n  # fft computing and normalization
        S_fft = np.abs(S_fft)  # y-axis symetric correction

        fft_data = S_fft
        fft_freqs = freq

        
        
        fig = go.Figure(data=[go.Scattergl(x=fft_freqs, y=fft_data, name='fft', mode='markers+lines')],
                        layout=go.Layout(xaxis={'title': 'Freq(Hz)'},
                                         yaxis={'title': 'Amplitude'},
                                         template=VibrationData.PLOT_TEMPLATE))
        if save:
            self.save_fig(fig, suffix='fft', kind='plotly')
        if plot:
            fig.show()

    def spectrogram(self, data, bound=(0, 2500), plot=False, save=True, engine='plotly'):

        window = self.num_points_per_tooth*3
        nftt = self.num_points_per_tooth*100
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
                                         hovertemplate='X: %{x:.4f}h <br>Y: %{y} <br>Z: %{z}',)],
                        layout=go.Layout(xaxis={'title': 'Time(sec)'},
                                         yaxis={'title': 'Freq(Hz)'},
                                         template=VibrationData.PLOT_TEMPLATE),
                        )
            
        if engine=='matplotlib':

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
        return int(self.sampling_freq/(self.rpm/60))
    
    @property
    def num_points_per_tooth(self):
        return self.num_points_in_one_period//self.n_flute

    @staticmethod
    def butter_lowpass_filter(data, cutoff, fs, order=5):
        # butter_lowpass
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = lfilter(b, a, data)
        return y

    def wavelet(self, data, wavelet='cmor6-1.5', xlim=None, ylim=None,clim=None, save=True):
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
                                                num=200) # 10**0 -- 10**2
                                   )

        cwt = scg.CWT(time = data['time'].values,
                   signal = data['data'].values-data['data'].mean(),
                   scales = scales,
                    wavelet=wavelet)

        ax=scg.cws(cwt,
                     spectrum = 'power',
                     figsize = (12, 6),
                     yscale = 'log',
                     ylabel = "Period [seconds]", 
                        xlabel = 'Time [seconds]',
                  xlim=xlim,ylim=ylim,clim=clim)
        
        text = ax.annotate("found freq", 
                           xy=(data['time'].iloc[100], 1/self.expected_tooth_freq), 
                           xytext=None, 
                           bbox=dict(boxstyle="round", facecolor="y", edgecolor="0.5"), 
                           arrowprops=dict(facecolor='yellow', shrink=0.05))        
        
        text = ax.annotate(f"Ref freq x2.0:{self.expected_tooth_freq*2}", 
                           xy=(data['time'].iloc[100], 1/(self.expected_tooth_freq*2)), 
                           xytext=None, 
                           bbox=dict(boxstyle="round", facecolor="y", edgecolor="0.5"), 
                           arrowprops=dict(facecolor='yellow', shrink=0.05))
        text = ax.annotate(f"Ref freq x0.5:{self.expected_tooth_freq*0.5}", 
                           xy=(data['time'].iloc[100], 1/(self.expected_tooth_freq*0.5)), 
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
     
        return (ax,cwt)

    def append(self, other):
        
        ## UPDATE READING
        _other_reading=other.reading.copy()

        # update index
        interval=self.reading.index[1] - self.reading.index[0]
        elapsed_time=self.reading.index[-1] -             self.reading.index[0] + interval
        _other_reading.index += elapsed_time

        # update time column
        interval=self.reading.loc[:, 'time'].iloc[1] - self.reading.loc[:, 'time'].iloc[0]
        elapsed_time=self.reading.loc[:, 'time'].iloc[-1] - self.reading.loc[:, 'time'].iloc[0] + interval
        _other_reading['time'] += elapsed_time

        # update tdms data
        self.reading=self.reading.append(_other_reading, verify_integrity=True)
        
        
        # UPDATE ATTRIBUTES
        for attr_name, attr_val in self.attributes.items():
            if type(attr_val) is list:
                self.attributes[attr_name].append(other.attributes[attr_name])
            else:
                if attr_val != other.attributes[attr_name]: # create new list
                    self.attributes[attr_name] = [attr_val, other.attributes[attr_name]]
                # else continue - if they are same value, no need to do anything           
                

        # UPDATE PATH
        # todo: make it correct.
        self.path = Path(os.path.commonpath((self.path, other.path)))

    @staticmethod
    def batch_read(fpath, cut_no, kind, n_cycle):
        
        #read excel file for attributes
        exceldata = ExcelData()
        
        vib = None
        for slot_no, path in tqdm(enumerate((fpath/str(cut_no)).glob(f'{kind}*.tdms'), start=1)):
            vib_current = VibrationData(path=path, 
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
    def rpm(self): return self.attributes['spindle_speed']
    
    @property
    def n_flute(self): return self.attributes['n_flute']
    
    @property
    def sampling_freq(self): return self.attributes['sampling_freq']
    

    def interpolate(self, yname='cutting_length'):
        """ 
        Call this function after total merge!
        """

        xname = 'time'
        self.reading[yname].iloc[0] = 0.   # quick fix for propor interpolation   
        self.reading[xname].iloc[0] = 0.         
        y = self.reading[yname].dropna()
        x = self.reading.loc[y.index, xname]
        
        interp_f = interpolate.interp1d(x.values,y.values, fill_value="extrapolate")
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
                dt tooth: {self.dt_tooth*1000} ms
                expected_tooth_freq : {self.expected_tooth_freq} Hz
                -----------------------------------------
                Shape/Sampling Frequency =?: Elapsed Time -> {self.reading.shape[0]/self.sampling_freq} =? {self.reading['time'].iloc[-1]}
                
                
                """
                
    
        
                


# # Real Data

# In[7]:

if __name__=="__main__":
    NUM_CYCLE = 4
    vib = VibrationData.batch_read(fpath=Path('D:/machining/data/raw'), cut_no=1, kind='acc', n_cycle=NUM_CYCLE)
    # vib = VibrationData.from_pickle(Path('D:/machining/data/raw/1/data.pickle'))


    # # Data Plot

    # In[8]:


    START_SEC = 5000
    # vib.apply_aggregation() # apply several filters
    subset = vib.make_subset_after(sec=START_SEC)

    # # # plot data
    # vib.static_plot()
    # vib.plot(plot=True)

    # # plot fft
    # vib.fft(data=subset,
    #         plot=True)

    # # plot spectrogram
    # vib.spectrogram(data=subset,
    #                 plot=True)

    # plot wavelet
    (ax,cwt) = vib.wavelet(data=subset, wavelet='cmor3-1.5', clim=[0, 0.02])
    plt.show()


    # In[9]:
    # plot wavelet
    plt.figure()
    vib.plot_toolwear()
    plt.show()

    # save as pickle
    vib.to_pickle()

# In[ ]:





# In[ ]:





# # # Fake Data

# # In[652]:


# NUM_CYCLE =4
# # read data
# vib = VibrationData(path='',
#                     rpm=5*10000, # create (rpm/60)*(n_flute) Hz vibration
#                     sampling_freq=100000, n_cycle=NUM_CYCLE,
#                     n_flute=3,
#                        fake=True, add_noise=False)
# vib.apply_aggregation() # apply several filters
# subset = vib.make_subset_after(sec=1)
# print(vib)
# # vib.static_plot(save=False)
# vib.plot(save=False)
# # vib.fft(data=subset, plot=True, save=False)
# # vib.spectrogram(data=subset, plot=True, save=False, engine='plotly')
# ax, cwt = vib.wavelet(data=subset, wavelet='cgau5', save=False)


# # In[ ]:





# # In[ ]:





# # In[ ]:





# # # Aux

# # In[ ]:


# subset = vib.make_subset_after(sec=2.0)
# colnames = ['data', 'lowpass', 'savgol', 'lowpass+savgol']
# traces = [vib._make_trace(data=subset, colname=colname, mode='lines') for colname in colnames]
# fig = go.FigureWidget(data=traces,
#                       layout=go.Layout(xaxis={'title': 'Time(sec)'},
#                                        yaxis={'title': 'Amplitude'},
#                                        template=VibrationData.PLOT_TEMPLATE))


# # In[ ]:


# fig.show()


# # In[ ]:


# fig = vib.plot(subset)


# # In[ ]:


# fig.data[0].name


# # In[ ]:


# # load fig
# fig = go.Figure("https://plot.ly/~jordanpeterson/889")

# # find the range of the slider.
# xmin, xmax = fig['layout']['xaxis']['range']

# # create FigureWidget from fig
# f = go.FigureWidget(data=fig.data, layout=fig.layout)

# slider = widgets.FloatRangeSlider(
#     min=xmin,
#     max=xmax,
#     step=(xmax - xmin) / 1000.0,
#     readout=False,
#     description='Time')
# slider.layout.width = '800px'


# # our function that will modify the xaxis range
# def update_range(y):
#     f.layout.xaxis.range = [y[0], y[1]]


# # display the FigureWidget and slider with center justification
# vb = VBox((f, interactive(update_range, y=slider)))
# vb.layout.align_items = 'center'
# vb


# # In[ ]:


# import plotly.graph_objects as go

# import pandas as pd

# # Load data
# df = pd.read_csv(
#     "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")
# df.columns = [col.replace("AAPL.", "") for col in df.columns]

# # Create figure
# fig = go.Figure()

# fig.add_trace(
#     go.Scatter(x=list(df.Date), y=list(df.High)))

# # Set title
# fig.update_layout(
#     title_text="Time series with range slider and selectors"
# )

# # Add range slider
# fig.update_layout(
#     xaxis=dict(
# #         rangeselector=dict(
# #             buttons=list([
# #                 dict(count=1,
# #                      label="1m",
# #                      step="month",
# #                      stepmode="backward"),
# #                 dict(count=6,
# #                      label="6m",
# #                      step="month",
# #                      stepmode="backward"),
# #                 dict(count=1,
# #                      label="YTD",
# #                      step="year",
# #                      stepmode="todate"),
# #                 dict(count=1,
# #                      label="1y",
# #                      step="year",
# #                      stepmode="backward"),
# #                 dict(step="all")
# #             ])
# #         ),
#         rangeslider=dict(
#             visible=True
#         ),
#         type="date"
#     )
# )

# fig.show()


# # In[ ]:


# x = widgets.IntText()
# x


# # In[ ]:


# x.value


# # In[ ]:





# # In[ ]:




# def compare_ffts(subset):
#     fig = make_subplots(rows=4, cols=1)
            

#     fig = plot_fft(fig, data=subset, whichone='acc', row=1, col=1)
#     fig = plot_fft(fig, data=subset, whichone='lowpass', row=2, col=1)
#     fig = plot_fft(fig, data=subset, whichone='savgol', row=3, col=1)
#     fig = plot_fft(fig, data=subset, whichone='lowpass+savgol', row=4, col=1)

#     fig.update_layout(  xaxis={'title': 'Freq(Hz)'},
#                         yaxis={'title': 'Amplitude'},
#                         height=600, width=800)
#     return fig

# def compare_spectrogram(subset, num_point_cycle):
#     fig = make_subplots(rows=4, cols=1)
            

#     fig = plot_spectrogram(fig, data=subset, whichone='acc', row=1, col=1, num_point_cycle=num_point_cycle)
#     fig = plot_spectrogram(fig, data=subset, whichone='lowpass', row=2, col=1, num_point_cycle=num_point_cycle)
#     fig = plot_spectrogram(fig, data=subset, whichone='savgol', row=3, col=1, num_point_cycle=num_point_cycle)
#     fig = plot_spectrogram(fig, data=subset, whichone='lowpass+savgol', row=4, col=1,num_point_cycle=num_point_cycle)

#     shapes = []
#     for i in range(NUM_CYCLE):
#         print(subset['time'].iloc[num_point_cycle*i])
#         shapes.append(
#                 dict(
#                     type="line",
#                     x0=subset['time'].iloc[num_point_cycle*i],
#                     y0=subset['acc'].max(),
#                     x1=subset['time'].iloc[num_point_cycle*i],
#                     y1=subset['acc'].min(),
#                     line=dict(
#                         color="green",
#                         width=3,
#                         dash="dashdot")
#                 )
#         )

#     fig.update_layout(  xaxis={'title': 'Freq(Hz)'},
#                         yaxis={'title': 'Amplitude'},
#                         height=600, width=800,
#                         shapes=shapes)
#     return fig
# # Load image
# subset = make_subset(rpm=12000, path='D:/machining/spectrogram-analysis/kanallar/accy_kanal6.tdms')
# original = subset['acc'].values

# # Wavelet transform of image, and plot approximation and details
# coeffs2 = pywt.dwt(original, 'db1')
# LL, HH = coeffs2
# fig = plt.figure(figsize=(12, 3))
# for i, a in enumerate([LL, HH]):
#     ax = fig.add_subplot(1, 4, i + 1)
#     ax.plot(a, interpolation="nearest", cmap=plt.cm.gray)
#     #ax.set_title(titles[i], fontsize=10)
#     ax.set_xticks([])
#     ax.set_yticks([])

# fig.tight_layout()
# plt.show()
# def plot_freq_response():
#     # Get the filter coefficients so we can check its frequency response.
#     b, a = butter_lowpass(cutoff, SAMPLING_FREQ, order)

#     # Plot the frequency response.
#     w, h = freqz(b, a, worN=8000)
#     plt.subplot(2, 1, 1)
#     plt.plot(0.5*SAMPLING_FREQ*w/np.pi, np.abs(h), 'b')
#     plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
#     plt.axvline(cutoff, color='k')
#     plt.xlim(0, 0.5*SAMPLING_FREQ)
#     plt.title("Lowpass Filter Frequency Response")
#     plt.xlabel('Frequency [Hz]')
#     plt.grid()


# plot_freq_response()





# # Demonstrate the use of the filter.
# # First make some data to be filtered.
# T = 5.0         # seconds
# n = int(T * fs) # total number of samples
# t = np.linspace(0, T, n, endpoint=False)
# # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
# data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)
# # Filter the data, and plot both the original and filtered signals.
# d = data['acc'][START_IDX:START_IDX+NUM_POINTS_1_CYCLE*NUM_CYCLE].values
# t = data['time'][START_IDX:START_IDX+NUM_POINTS_1_CYCLE*NUM_CYCLE].values

# y = butter_lowpass_filter(d, cutoff, fs, order)

# plt.subplot(2, 1, 2)
# plt.plot(t, d, 'b-', label='data')
# plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
# plt.xlabel('Time [sec]')
# plt.grid()
# plt.legend()

# plt.subplots_adjust(hspace=0.35)
# plt.show()import numpy as np
# import pandas as pd
# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly as py


# df = pd.DataFrame({'A':[100, 120, 100, 105, 110], 
#                 'B':[130, 120, 100, 105, 110],
#                 'C':[110, 110, 140, 115, 120],
#                 'D':[140, 120, 160, 120, 130],                   
#                 'E':[150, 130, 100, 105, 150]})
# df2 = pd.DataFrame({'A':[140, 150, 110, 115, 120], 
#                 'B':[150, 120, 100, 105, 110],
#                 'C':[120, 120, 110, 115, 120],
#                 'D':[170, 140, 120, 125, 150],                   
#                 'E':[140, 180, 115, 115, 140]})


# fig = make_subplots(rows=2, cols=1,shared_xaxes=True)

# # Add traces, one for each slider step
# for step in range(len(df.index)):
#     fig.append_trace(
#         go.Scatter(
#             visible=False,
#             line=dict(color="#00CED1", width=2),
#             name="Time = " + str(step),
#             x=df.columns[0:],
#             y=df.loc[step]),row=1, col=1)
    
# #for step in range(len(df2.index)):# Tried this does not work
#     fig.append_trace(
#         go.Scatter(
#             visible=False,
#             line=dict(color="red", width=2),
#             name="Time = " + str(step),
#             x=df2.columns[0:],
#             y=df2.loc[step]),row=2, col=1)


# #fig.data[1].visible = True


# # Create and add slider
  
# steps = []

# for i in range(0, len(fig.data), 2):
#     step = dict(
#         method="restyle",
#         args=["visible", [False] * len(fig.data)],
#     )
#     step["args"][1][i:i+2] = [True, True]
#     steps.append(step)

# sliders = [dict(
#     active=0,
#     currentvalue={"prefix": "Time:  "},
#     pad={"t": 50},
#     steps=steps
# )]

# fig.update_yaxes(title_text="Temperature", range=[-160, 260],nticks=30, row=1, col=1)
# fig.update_yaxes(title_text="Pressure", range=[-169, 260],nticks=30, row=2, col=1)
# fig.update_layout(sliders=sliders, title="Time Series - Interactive", template ="plotly_white")

# fig.update_layout(width=800,height=600,)
# fig.show()
# # In[ ]:




