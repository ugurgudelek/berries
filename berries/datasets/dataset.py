# -*- coding: utf-8 -*-
# @Time   : 6/16/2020 4:42 PM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : dataset.py


from torch.utils.data import Dataset
import torch


class StatelessTimeseriesDataset(Dataset):
    def __init__(self, timeseries, seq_len, look_ahead):
        self.timeseries = timeseries
        self.seq_len = seq_len
        self.look_ahead = look_ahead

    def __len__(self):
        return len(self.timeseries) - self.seq_len - self.look_ahead + 1

    def __getitem__(self, ix):

        return {'data': {'x': torch.from_numpy(self.timeseries[ix:ix + self.seq_len]).float().view(-1, 1)},
                'target': torch.tensor([self.timeseries[ix + self.seq_len + self.look_ahead - 1]]).float()}
