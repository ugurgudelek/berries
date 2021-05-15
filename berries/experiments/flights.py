# -*- coding: utf-8 -*-
# @Time   : 6/16/2020 4:10 PM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : flights.py

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from berries.experiments.experiment import Experiment
from berries.logger import MultiLogger, LocalLogger
from berries.trainer import Trainer
from berries.model import LSTM
from berries.metric import metrics
from berries.utils.plot import mpl2pillow

from berries.datasets import Flights


class FlightExperiment(Experiment):
    def __init__(self):
        super(FlightExperiment, self).__init__()

        self.params = {'project_name': 'debug',
                       'experiment_name': 'flight',
                       'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                       'resume': False,
                       'pretrained': False,
                       'log_interval': 1,
                       'stdout_interval': 1,
                       'root': Path('../../'),
                       }

        self.hyperparams = {'lr': 0.1,
                            'weight_decay': 0.,
                            'epoch': 2000,
                            'train_batch_size': 104,
                            'test_batch_size': 32,
                            'seq_len': 4,
                            'input_size': 1,
                            'hidden_size': 2,
                            'num_layers': 1,
                            'output_size': 1,
                            'aux_input_size': 0,
                            'bidirectional': False,
                            'stateful': False,
                            'hidden_reset_period': None,
                            'return_sequences': False,
                            }

        self.dataset = Flights()

        self.set_logger(backend=LocalLogger)

        self.model = LSTM(input_size=self.hyperparams['input_size'], hidden_size=self.hyperparams['hidden_size'],
                          output_size=self.hyperparams['output_size'], num_layers=self.hyperparams['num_layers'],
                          batch_size=self.hyperparams['train_batch_size'],
                          aux_input_size=self.hyperparams['aux_input_size'],
                          bidirectional=self.hyperparams['bidirectional'],
                          stateful=self.hyperparams['stateful'],
                          hidden_reset_period=self.hyperparams['hidden_reset_period'],
                          return_sequences=self.hyperparams['return_sequences'])

        self.trainer = Trainer(root=self.params['root'],
                               model=self.model, dataset=self.dataset,
                               metrics=[metrics.Accuracy],
                               hyperparams=self.hyperparams, params=self.params,
                               logger=self.logger,
                               criterion=torch.nn.MSELoss())

        self.trainer.attach_callback(self.callback)

    @staticmethod
    def prediction_image(prediction, targets):
        indices = list(range(len(prediction)))

        fig, ax = plt.subplots(nrows=1)

        ax.plot(indices, prediction, label='prediction', c='r')
        ax.plot(indices, targets, label='true')
        ax.set_ylabel('Amplitude')
        ax.legend()

        ax.set_xlim(indices[0], indices[-1])

        plt.suptitle("Flights")

        return mpl2pillow(fig)

    @staticmethod
    def callback(instance, epoch):
        # Training prediction images
        prediction, targets = instance.predict_loader(dataloader=DataLoader(instance.dataset.trainset,
                                                                            batch_size=instance.hyperparams[
                                                                                'test_batch_size'],
                                                                            drop_last=False,
                                                                            shuffle=False))

        img = FlightExperiment.prediction_image(prediction, targets)
        instance.logger.log_image(log_name=f'phase-training',
                                  x=epoch, y=img,
                                  image_name=f'epoch-{epoch}.phase-training')

        # Test prediction images
        prediction, targets = instance.predict_loader(DataLoader(instance.dataset.testset,
                                                                 batch_size=instance.hyperparams['test_batch_size'],
                                                                 drop_last=False,
                                                                 shuffle=False))
        img = FlightExperiment.prediction_image(prediction, targets)
        instance.logger.log_image(log_name=f'phase-test',
                                  x=epoch, y=img,
                                  image_name=f'epoch-{epoch}.phase-test')


def main():
    with FlightExperiment() as experiment:
        experiment.run()


if __name__ == "__main__":
    main()
