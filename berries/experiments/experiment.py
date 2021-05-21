# -*- coding: utf-8 -*-
# @Time   : 6/16/2020 4:48 PM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : experiment.py

import numpy as np
import torch
from abc import ABC, abstractmethod
from berries.logger.logger import LocalLogger as logger_backend


class Meta(type):

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance.__post_init__()
        return instance


class Experiment(metaclass=Meta):
    SEED = 42

    def __init__(self):
        torch.manual_seed(self.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.SEED)

    def __post_init__(self):
        for attr in ('params', 'hyperparams', 'model', 'dataset', 'logger',
                     'trainer'):
            if not hasattr(self, attr):
                raise AttributeError(
                    f'{self.__class__.__name__}.{attr} is invalid.')

            _has_neptune_id = 'id' in self.params['neptune']
            _resume = self.params['resume']
            if (_has_neptune_id != _resume):
                raise Exception(
                    'if neptune["id"] is given, then resume should be True or vice versa'
                )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace):
        self.logger.stop()
