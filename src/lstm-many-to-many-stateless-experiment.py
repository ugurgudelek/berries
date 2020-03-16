# -*- coding: utf-8 -*-
# @Time   : 3/15/2020 8:32 PM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : lstm-many-to-one-stateless-experiment.py
# @Status : working

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

from model.lstm import LSTM
from dataset.timeseries import TimeSeriesManyToManyDataset
from trainer.trainer import Trainer

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    input_size = 1
    hidden_size = 2
    num_layers = 1

    params = {'seed':42,
        'device': 'cpu',
              'resume': False,
              'pretrained': False,
              'experiment_name':'lstm-many-to-many-stateless',
              'save_interval':100,
              'save_fig': True,
              'problem_type':'many-to-many'}

    hyperparams = {'lr': 0.1,
                   'weight_decay': 0.,
                   'epoch': 10000,
                   'train_batch_size': 16,
                   'test_batch_size': 4,
                   'seq_len':5}

    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    dataset = TimeSeriesManyToManyDataset(path='../input/airline-passengers/airline-passengers.csv', colname='Passengers',
                                         seq_length=hyperparams['seq_len'], train_split=.9)

    model = LSTM(input_size=input_size, hidden_size=hidden_size,
                 output_size=1, num_layers=1,
                 batch_size=hyperparams['train_batch_size'],
                 stateful=False,  hidden_reset_period=None,
                 problem_type=params['problem_type'])
    trainer = Trainer(model=model, dataset=dataset, hyperparams=hyperparams,
                      params=params)


    trainer.fit()

