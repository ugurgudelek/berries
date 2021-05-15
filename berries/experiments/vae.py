__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

from berries.model.fcnn import FCNN
from pathlib import Path

import torch
from berries.datasets.ecg5000 import ECG5000
from berries.datasets.uea_ucr_dataset import BerriesUEAUCRDataset
from berries.experiments.experiment import Experiment
from berries.metric import metrics
from berries.model.vae import VAE
from berries.trainer.trainer import VAEClassifierTrainer, VAETrainer
from berries.utils.plot import plot_clustering_legacy
from berries.metric.metrics import Accuracy
from torch import nn

import numpy as np


class VAEExperiment(Experiment):
    def __init__(self):
        self.params = {
            'project_name': 'debug',
            'experiment_name': 'vae',
            'seed': 42,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'resume': False,
            'pretrained': False,
            'log_interval': 1,
            'stdout_interval': 30,
            'root': Path('./'),
        }
        self.hyperparams = {
            'lr': 0.0005,
            'epoch': 20,
            'clip': 5,
            'block': 'LSTM',
            'hidden_size': 90,
            'hidden_layer_depth': 1,
            'latent_length': 20,
            'batch_size': 32,
            'dropout_rate': 0.2
        }

        self.dataset = ECG5000(forVAE=True)

        sequence_length = self.dataset.trainset[0]['data'].shape[0]
        number_of_features = self.dataset.trainset[0]['data'].shape[1]

        self.model = VAE(
            sequence_length=sequence_length,
            number_of_features=number_of_features,
            hidden_size=self.hyperparams['hidden_size'],
            hidden_layer_depth=self.hyperparams['hidden_layer_depth'],
            latent_length=self.hyperparams['latent_length'],
            batch_size=self.hyperparams['batch_size'],
            block=self.hyperparams['block'],
            dropout_rate=self.hyperparams['dropout_rate'])

    def run(self):

        with VAETrainer(model=self.model,
                        metrics=[],
                        hyperparams=self.hyperparams,
                        params=self.params,
                        optimizer=None,
                        criterion=nn.MSELoss(size_average=False),
                        logger=None) as trainer:

            trainer.fit(dataset=self.dataset.trainset)

            test_latent = trainer.encode(dataset=self.dataset.testset)

            plot_clustering_legacy(test_latent,
                                   self.dataset.testset.get_targets(),
                                   engine='matplotlib',
                                   download=False)


class VAEClassifierExperiment(Experiment):
    def __init__(self):
        self.params = {
            'project_name': 'debug',
            'experiment_name': 'vae-clf',
            'seed': 42,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'resume': False,
            'pretrained': False,
            'log_interval': 1,
            'stdout_interval': 30,
            'root': Path('./'),
        }
        self.hyperparams = {
            'lr': 0.0005,
            'epoch': 20,
            'clip': 5,
            'block': 'LSTM',
            'hidden_size': 90,
            'hidden_layer_depth': 1,
            'latent_length': 20,
            'batch_size': 32,
            'dropout_rate': 0.2,
            'clf_input_size': 20,
            'clf_hidden_size': 20,
            'clf_output_size': 5
        }

        self.dataset = ECG5000(forVAE=False)

        sequence_length = self.dataset.trainset[0]['data'].shape[0]
        number_of_features = self.dataset.trainset[0]['data'].shape[1]

        device = torch.device('cuda:0' if self.params['device'] ==
                              'cuda' else 'cpu')

        self.compressor = VAE(
            sequence_length=sequence_length,
            number_of_features=number_of_features,
            hidden_size=self.hyperparams['hidden_size'],
            hidden_layer_depth=self.hyperparams['hidden_layer_depth'],
            latent_length=self.hyperparams['latent_length'],
            batch_size=self.hyperparams['batch_size'],
            block=self.hyperparams['block'],
            dropout_rate=self.hyperparams['dropout_rate']
        ).load_model_from_path(
            path=Path('projects/debug/vae/checkpoints/20/model-optim.pth'),
            device=device)

        self.model = FCNN(input_size=self.hyperparams['clf_input_size'],
                          hidden_size=self.hyperparams['clf_hidden_size'],
                          output_size=self.hyperparams['clf_output_size'])

        self.trainer = VAEClassifierTrainer(model=self.model,
                                            compressor=self.compressor,
                                            metrics=[Accuracy],
                                            hyperparams=self.hyperparams,
                                            params=self.params,
                                            optimizer=None,
                                            criterion=nn.CrossEntropyLoss(),
                                            logger=None)

    def run(self):
        self.trainer.fit(dataset=self.dataset.trainset)


def main():
    # VAEExperiment().run()
    VAEClassifierExperiment().run()


if __name__ == "__main__":
    main()
