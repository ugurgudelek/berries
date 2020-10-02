__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

from pathlib import Path

import torch
from berries.datasets.ecg5000 import ECG5000
from berries.datasets.uea_ucr_dataset import BerriesUEAUCRDataset
from berries.experiments.experiment import Experiment
from berries.metric import metrics
from berries.model.vae import VAE
from berries.trainer.trainer import VAETrainer
from berries.utils.plot import plot_clustering_legacy
from torch import nn


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

        self.dataset = ECG5000()

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


def main():
    experiment = VAEExperiment()
    experiment.run()


if __name__ == "__main__":
    main()
