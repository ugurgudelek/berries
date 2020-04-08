# -*- coding: utf-8 -*-
# @Time   : 3/16/2020 3:16 AM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : toolwear.py
# @Status : -

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

from model.lstm import LSTM
from dataset.toolwear import ToolwearTorchDataset
from trainer.trainer import Trainer

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()


    params = {'seed': 42,
              'device': 'cuda' if torch.cuda.is_available() else 'cpu',
              # 'device':'cpu',
              'resume': False,
              'pretrained': False,
              'experiment_name':'toolwear',
              'save_interval':1,
              'save_fig': True,
              'problem_type':'many-to-one'}

    hyperparams = {'lr': 0.001,
                   'weight_decay': 0.,
                   'epoch': 10000,
                   'train_batch_size': 640,
                   'test_batch_size': 640,
                   'seq_len': 8000,
    'input_size': 1,
    'hidden_size': 10,
    'num_layers': 1,
                   'output_size':1}

    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    dataset = ToolwearTorchDataset(seq_length=hyperparams['seq_len'], train_split=.8,
                                   cut_lim=(0.65, 0.66))
    dataset.plot()
    model = LSTM(input_size=hyperparams['input_size'], hidden_size=hyperparams['hidden_size'],
                 output_size=hyperparams['output_size'], num_layers=hyperparams['num_layers'],
                 batch_size=hyperparams['train_batch_size'],
                 stateful=False,  hidden_reset_period=None,
                 problem_type=params['problem_type'])
    trainer = Trainer(model=model, dataset=dataset, hyperparams=hyperparams,
                      params=params)


    trainer.fit()

