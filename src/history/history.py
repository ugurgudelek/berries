__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

from collections import defaultdict
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import os


class History:
    """

    """

    def __init__(self):

        self.storage_keys = []
        # container -> phase -> epoch -> storage_list

        # init for epoch_num
        # self.container = dict(train=list(), test=list())

        self.container = {'train': {key: list() for key in self.storage_keys},
                          'test': {key: list() for key in self.storage_keys}}

    def append(self, phase, log_dict):
        """
        """

        for key, value in log_dict.items():
            self._validate(key)
            self.container[phase][key].append(value)

        return self

    def _validate(self, key):
        if key not in self.storage_keys:
            self.container['train'][key] = list()
            self.container['test'][key] = list()
            self.storage_keys.append(key)

    def to_dataframe(self, phase='both'):
        if phase == 'both':
            return (self.to_dataframe(phase='train'), self.to_dataframe(phase='test'))

        return pd.DataFrame(self.container[phase])

    def save(self, fpath, phase='both'):
        if phase == 'both':
            self.save(fpath=fpath, phase='train')
            self.save(fpath=fpath, phase='test')
            return
        self.to_dataframe(phase=phase).to_csv(f"{fpath}/history_{phase}.csv")

    def plot(self, fpath=None, show=False):
        train_df = self.to_dataframe('train')
        test_df = self.to_dataframe('test')

        fig, axes = plt.subplots(nrows=2, sharex=True)
        axes[0].plot(train_df['epoch'], train_df['loss'], label='training loss')
        axes[0].plot(test_df['epoch'], test_df['loss'], label='test loss')
        axes[0].legend()
        axes[1].plot(train_df['epoch'], train_df['accuracy'], label='training accuracy')
        axes[1].plot(test_df['epoch'], test_df['accuracy'], label='test accuracy')
        axes[1].legend()

        plt.xlabel('Number of epochs')
        plt.suptitle('Accuracy and Loss History')
        if fpath is not None:
            os.makedirs(fpath, exist_ok=True)
            plt.savefig(f"{fpath}/history.jpg")

        if show:
            plt.show()
        plt.close(fig)






if __name__ == "__main__":
    h = History()
    h.append(phase='train', log_dict={'epoch': 0,
                                      'loss': 5,
                                      'accuracy': 2,
                                      })
    h.append(phase='train', log_dict={'epoch': 1,
                                      'loss': 1,
                                      'accuracy': 3,
                                      })

    h.to_dataframe(phase='train').to_csv('test_csv.csv', index=False)
    h.plot()
    print()
