# -*- coding: utf-8 -*-
# @Time   : 5/27/2020 5:43 PM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : plotter.py
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from ..utils.plot import mpl2pillow


class Plotter():
    def __init__(self):
        pass

    @staticmethod
    def plot_prediction(prediction, targets, title):
        indices = list(range(len(prediction)))

        fig, ax = plt.subplots(nrows=1)

        ax.scatter(indices, prediction, label='Prediction', s=1, c='r')
        ax.plot(indices, targets, label='Target', linestyle='dashed')
        ax.set_ylabel('Amplitude')
        ax.legend(frameon=True, loc='upper left')

        ax.set_xlim(indices[0], indices[-1])
        # ax.set_ylim(0, ax.get_ylim()[1])

        ax.set_xlabel('Time (sec)')
        ax.set_ylabel(r'Tool wear ($\mu{m}$)')

        # plt.title(title)

        # return mpl2pillow(fig)
        return ax

    @staticmethod
    def plot_prediction_and_wavelet(prediction, targets, wavelet_img, title):
        indices = list(range(len(prediction)))

        fig, axes = plt.subplots(nrows=2)

        axes[0].scatter(indices, prediction, label='prediction', s=1, c='r')
        axes[0].plot(indices, targets, label='true')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend()

        axes[0].set_xlim(indices[0], indices[-1])

        axes[1].imshow(wavelet_img[:, indices], interpolation='nearest', aspect='auto')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Scales')
        plt.suptitle(title)

        # save_dir.mkdir(parents=True, exist_ok=True)
        # plt.savefig(save_dir / filename)
        # plt.close()

        return mpl2pillow(fig)

    @staticmethod
    def learning_curve(read_dir, save_dir):
        training_loss = pd.read_csv(read_dir / 'metric' / 'training_loss.csv')
        validation_loss = pd.read_csv(read_dir / 'metric' / 'validation_loss.csv')

        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(training_loss['x'], training_loss['y'], label='training loss', color='orange')
        ax[0].set_ylabel('Magnitude')
        ax[0].legend(loc='upper right')

        ax[1].plot(validation_loss['x'], validation_loss['y'], label='validation loss', color='blue')
        ax[1].set_ylabel('Magnitude')
        ax[1].legend(loc='upper right')
        ax[1].set_xlabel('Epoch')

        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / 'learning_curve.png')
        plt.close()
