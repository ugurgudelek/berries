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
from dataset.toolwear import Toolwear, ToolwearBag
from trainer.trainer import Trainer

from pathlib import Path
from tqdm import tqdm


def experiment(train_cut_nos, test_cut_no):
    torch.multiprocessing.freeze_support()

    params = {'seed': 42,
              'device': 'cuda' if torch.cuda.is_available() else 'cpu',
              'resume': False,
              'pretrained': False,
              'experiment_name': f'toolwear_std_train({"-".join(map(str,train_cut_nos))})_test({test_cut_no})',
              'save_interval': 10,
              'log_interval': 10,
              'save_fig': True,
              'problem_type': 'many-to-one'}

    hyperparams = {'lr': 0.001,
                   'weight_decay': 0.,
                   'epoch': 200,
                   'train_batch_size': 100,
                   'test_batch_size': 100,
                   'seq_len': 60,
                   'input_size': 128,
                   'hidden_size': 10,
                   'num_layers': 1,
                   'output_size': 1,
                   'aux_input_size': 1  # cutting speed
                   }

    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    dataset = ToolwearBag(root=Path('D:/YandexDisk/machining/data')) \
        .to_torch_wavelet_dataset(seq_len=hyperparams['seq_len'],
                                  train_cuts=train_cut_nos,
                                  test_cut=test_cut_no)

    # dataset.plot()
    model = LSTM(input_size=hyperparams['input_size'], hidden_size=hyperparams['hidden_size'],
                 output_size=hyperparams['output_size'], num_layers=hyperparams['num_layers'],
                 batch_size=hyperparams['train_batch_size'],
                 aux_input_size=hyperparams['aux_input_size'],
                 stateful=False, hidden_reset_period=None,
                 problem_type=params['problem_type'])
    trainer = Trainer(model=model, dataset=dataset, hyperparams=hyperparams,
                      params=params)

    trainer.fit()

if __name__ == "__main__":
    with tqdm() as pbar:
        for train_cut_nos, test_cut_no in [
            [[13],13],
            [[14],14],
            [[15],15],
            [[16],16],
            [[14], 16],
            [[14], 15],
            [[14], 13],
            [[13,14,15], 16],
            [[14,15,16], 13],
            [[15,16,13], 14],
            [[16,13,14], 15]
        ]:
            experiment(train_cut_nos, test_cut_no)
            pbar.update(1)



    # # Train plot
    # prediction = trainer.predict_loader(trainer.train_loader)
    # indices = list(range(len(prediction)))
    #
    # fig, axes = plt.subplots(nrows=2)
    #
    # axes[0].scatter(indices, prediction, label='prediction', s=1, c='r')
    # axes[0].plot(indices, dataset.train_datasets[0].targets[indices], label='true')
    # axes[0].legend()
    #
    # axes[0].set_xlim(indices[0], indices[-1])
    #
    # axes[1].imshow(dataset.train_datasets[0].data.T.loc[:, indices])
    # plt.suptitle('Train')
    # plt.show()

