# -*- coding: utf-8 -*-
# @Time   : 3/12/2020 3:53 PM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : toy_lstm_experiment.py

import torch
import torch.nn as nn
from dataset.toy.toy_timeseries_dataset import ToyTimeseriesDataset
import matplotlib.pyplot as plt

import numpy as np

from model.lstm import LSTM


from trainer.rnntrainer import RNNTrainer
from dataset.generic import Standardizer, TimeSeriesDatasetWrapper

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    hyperparams = {
        # Model params
        'model': 'LSTM',
        'emsize': 32,
        'nhid': 32,
        'nlayers': 2,
        'res_connection': False,  # resudial connections
        'dropout': 0.2,
        'tied': False,  # tie the word embedding and softmax weights (deprecated)
        # Optim params
        'lr': 2e-4,
        'weight_decay': 1e-4,
        'clip': 10,  # gradient clipping
        # Dataloader params
        'train_batch_size': 64,
        'test_batch_size': 64,
        'seq_len': 50,  # sequence length

        'teacher_forcing_ratio': 0.7,  # teacher forcing ratio (deprecated)
        'epoch': 1000,  # upper epoch limits
    }
    params = {'log_interval': 2,
              'save_interval': 2,
              'save_fig': True,
              'resume': False,
              'pretrained': False,
              'prediction_window_size': 10,
              'augment': True,
              'seed': 42,
              'device': 'cuda',
              'experiment_name': 'nyc_taxi',
              'start_point': 3000,
              'recursive_start_point': 3500,
              'end_point': 4000
              }

    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    scaler = Standardizer()

    dataset = ToyTimeseriesDataset()
    dataset = TimeSeriesDatasetWrapper(trainset=dataset.trainset,
                                       testset=dataset.testset)

    model = LSTM(input_size=dataset.feature_dim,
                 hidden_size=hyperparams['nhid'])


    trainer = RNNTrainer(model=model,
                         dataset=dataset,
                         hyperparams=hyperparams,
                         params=params)

    trainer.fit()
