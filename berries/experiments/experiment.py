# -*- coding: utf-8 -*-
# @Time   : 6/16/2020 4:48 PM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : experiment.py


import numpy as np
import torch

from berries.logger import MultiLogger, LocalLogger


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

    def set_logger(self, backend):
        self.logger = backend(
            root=self.params['root'],
            project_name=self.params['project_name'],
            experiment_name=self.params['experiment_name'],
            params=self.params,
            hyperparams=self.hyperparams)

    def validate_attr(self):
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.stop()

    def run(self):
        if not self.validate_attr():
            raise Exception("Check attributes. Some of them are None.")
        self.trainer.fit()
