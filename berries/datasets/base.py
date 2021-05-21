# -*- encoding: utf-8 -*-
# @File    :   base.py
# @Time    :   2021/05/20 11:55:32
# @Author  :   Ugur Gudelek
# @Contact :   ugurgudelek@gmail.com
# @Desc    :   None

import torch
from torch.utils.data import Dataset, Subset


class BaseTorchDataset(Dataset):

    def __init__(self):
        super(BaseTorchDataset, self).__init__()

    def get_random_sample(self, n=1):

        indices = torch.randint(low=0, high=len(self), size=(n,))
        return Subset(self, indices)
