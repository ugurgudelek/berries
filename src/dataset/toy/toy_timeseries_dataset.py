# -*- coding: utf-8 -*-
# @Time   : 3/12/2020 3:56 PM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : toy_timeseries_dataset.py

import torch
from torch.utils.data import Dataset

import numpy as np


class ToyTimeseriesDataset:
    def __init__(self):

        self.data = torch.tensor(np.arange(0, 1000), dtype=torch.float).unsqueeze(dim=1)

        self.feature_size = 1
        self.seq_len = 10
        self.batch_size = 4
        self.stride = 10


        self.time_skip = self.data.size(0) // self.batch_size
        self.data = self.data.narrow(0, 0, self.time_skip * self.batch_size)

        # batched_data shape : [batch_size, seq_len, feature_size]
        self.batched_data = self.data.contiguous().view(self.batch_size, self.seq_len, -1).transpose(0, 1)

        print()



