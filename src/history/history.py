__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

from collections import defaultdict
import numpy as np
import pandas as pd


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
    print()
