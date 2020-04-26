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
from dataset.toolwear import ToolwearWaveletDataset, ToolwearWaveletConcatDataset
from trainer.trainer import Trainer

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()


    params = {'seed': 42,
              'device': 'cuda' if torch.cuda.is_available() else 'cpu',
              'resume': False,
              'pretrained': False,
              'experiment_name':'toolwear_concat_aux_train(14-15-16)_test(13)',
              'save_interval':10,
              'log_interval': 50,
              'save_fig': True,
              'problem_type': 'many-to-one'}

    hyperparams = {'lr': 0.001,
                   'weight_decay': 0.,
                   'epoch': 1000,
                   'train_batch_size': 10,
                   'test_batch_size': 10,
                   'seq_len': 60,
                    'input_size': 128,
                    'hidden_size': 10,
                    'num_layers': 1,
                   'output_size':1,
                   'aux_input_size': 1  # cutting speed
                   }

    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    dataset = ToolwearWaveletConcatDataset(seq_len=hyperparams['seq_len'])
    # dataset.plot()
    model = LSTM(input_size=hyperparams['input_size'], hidden_size=hyperparams['hidden_size'],
                 output_size=hyperparams['output_size'], num_layers=hyperparams['num_layers'],
                 batch_size=hyperparams['train_batch_size'],
                 aux_input_size=hyperparams['aux_input_size'],
                 stateful=False,  hidden_reset_period=None,
                 problem_type=params['problem_type'])
    trainer = Trainer(model=model, dataset=dataset, hyperparams=hyperparams,
                      params=params)


    trainer.fit()

    # Test plot
    prediction = trainer.predict_loader()
    indices = list(range(len(prediction)))

    fig, axes = plt.subplots(nrows=2)

    axes[0].scatter(indices, prediction, label='prediction', s=1, c='r')
    axes[0].plot(indices, dataset.testset.labels[indices], label='true')
    axes[0].legend()

    axes[0].set_xlim(indices[0], indices[-1])

    axes[1].imshow(dataset.testset.data.T[:, indices])
    plt.suptitle('Test')
    plt.show()

    # Train plot
    prediction = trainer.predict_loader(trainer.train_loader)
    indices = list(range(len(prediction)))

    fig, axes = plt.subplots(nrows=2)

    axes[0].scatter(indices, prediction, label='prediction', s=1, c='r')
    axes[0].plot(indices, dataset.train_datasets[0].labels[indices], label='true')
    axes[0].legend()

    axes[0].set_xlim(indices[0], indices[-1])

    axes[1].imshow(dataset.train_datasets[0].data.T[:, indices])
    plt.suptitle('Train')
    plt.show()

