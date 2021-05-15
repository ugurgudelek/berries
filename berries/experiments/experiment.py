# -*- coding: utf-8 -*-
# @Time   : 6/16/2020 4:48 PM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : experiment.py

import numpy as np
import torch


class Experiment:

    def __init__(self):
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(42)

        self.params = None
        self.hyperparams = None
        self.dataset = None
        self.logger = None
        self.model = None
        self.trainer = None

    def validate_attr(self):
        return True

    def run(self):
        if not self.validate_attr():
            raise Exception("Check attributes. Some of them are None.")
        self.trainer.fit()
