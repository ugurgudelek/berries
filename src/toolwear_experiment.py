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
from logger.logger import MultiLogger
import metric

from pathlib import Path
from tqdm import tqdm

from itertools import combinations, cycle


# import IPython; IPython.embed(); exit(1)


def experiment(pre_exp_name, train_cut_nos, test_cut_no):
    torch.multiprocessing.freeze_support()

    params = {'project_name': 'machining',
              'experiment_name': f'{pre_exp_name}-({"-".join(map(str, train_cut_nos))})-({test_cut_no})',
              'train_cut_nos': train_cut_nos,
              'test_cut_no': test_cut_no,
              'seed': 42,
              'normalized': True,

              'device': 'cuda' if torch.cuda.is_available() else 'cpu',
              'resume': False,
              'pretrained': False,
              'save_interval': 5,
              'log_interval': 5,
              'problem_type': 'many-to-one'}

    hyperparams = {'lr': 0.001,
                   'weight_decay': 0.,
                   'epoch': 200,
                   'train_batch_size': 128,
                   'test_batch_size': 128,
                   'seq_len': 300,
                   'input_size': 128,
                   'hidden_size': 12,
                   'num_layers': 1,
                   'output_size': 1,
                   'aux_input_size': 2,  # [cutting speed, tool diameter, Fs, tool brand]
                   'stateful': False,
                   'hidden_reset_period': None,
                   }

    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    dataset = ToolwearBag(root=Path('D:/YandexDisk/machining/data'), kind='acc') \
        .to_torch_wavelet_dataset(seq_len=hyperparams['seq_len'],
                                  train_cuts=train_cut_nos,
                                  test_cut=test_cut_no)

    # dataset.plot()
    model = LSTM(input_size=hyperparams['input_size'], hidden_size=hyperparams['hidden_size'],
                 output_size=hyperparams['output_size'], num_layers=hyperparams['num_layers'],
                 batch_size=hyperparams['train_batch_size'],
                 aux_input_size=hyperparams['aux_input_size'],
                 stateful=hyperparams['stateful'], hidden_reset_period=hyperparams['hidden_reset_period'],
                 problem_type=params['problem_type'])

    with Trainer(root=Path('C:/Users/ugur/Documents/GitHub/ai-framework'),
                 model=model, dataset=dataset,
                 metrics=[metric.MAE, metric.MSE, metric.MAPE, metric.RMSE],
                 hyperparams=hyperparams, params=params,
                 logger=MultiLogger(
                     root=Path('C:/Users/ugur/Documents/GitHub/ai-framework'),
                     project_name=params['project_name'],
                     experiment_name=params['experiment_name'],
                     params=params,
                     hyperparams=hyperparams)) as trainer:
        trainer.fit()


if __name__ == "__main__":
    # AVAILABLE_CUT_NOS = [1, 2, 3, 4, 13, 14, 15, 16]
    # train_cut_nos_list = list(combinations(AVAILABLE_CUT_NOS, len(AVAILABLE_CUT_NOS) - 1))
    # test_cut_no_list = [set(AVAILABLE_CUT_NOS).difference(set(train_cut_nos)).pop() for train_cut_nos in
    #                     train_cut_nos_list]


    SCENARIOS=(
        ('all-10-I', (13, 14, 15), 16),
        ('all-10-II', (16, 13, 14), 15),
        ('all-10-III', (15, 16, 13), 14),
        ('all-10-IV', (14, 15, 16), 13),

        ('all-12-I', (1, 2, 3), 4),
        ('all-12-II', (4, 1, 2), 3),
        ('all-12-III', (3, 4, 1), 2),
        ('all-12-IV', (2, 3, 4), 1),

        ('same-dia-add-affect-I', (13, 14, 15, 16), 1),
        ('same-dia-add-affect-I', (13, 14, 15, 16, 2), 1),
        ('same-dia-add-affect-I', (13, 14, 15, 16, 2, 3), 1),
        ('same-dia-add-affect-I', (13, 14, 15, 16, 2, 3, 4), 1),

        ('same-dia-add-affect-II', (1, 2, 3, 4), 16),
        ('same-dia-add-affect-II', (1, 2, 3, 4, 13), 16),
        ('same-dia-add-affect-II', (1, 2, 3, 4, 13, 14), 16),
        ('same-dia-add-affect-II', (1, 2, 3, 4, 13, 14, 15), 16),

        ('same-speed-add-affect-I', (3, 4, 13, 14, 15, 16), 1),
        ('same-speed-add-affect-I', (3, 4, 13, 14, 15, 16, 2), 1),

        ('same-speed-add-affect-II', (1, 2, 13, 14, 15), 16),
        ('same-speed-add-affect-II', (1, 2, 13, 14, 15, 3), 16),
        ('same-speed-add-affect-II', (1, 2, 13, 14, 15, 3, 4), 16),
    )

    with tqdm() as pbar:
        for pre_exp_name, train_cut_nos, test_cut_no in SCENARIOS:
            experiment(pre_exp_name, train_cut_nos, test_cut_no)
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
