__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import torch
from torch import nn
from torchvision import transforms

from pathlib import Path

from berries.experiments.experiment import Experiment
from berries.trainer.trainer import CNNTrainer
from berries.model.cnn import CNN
from berries.datasets.mnist import MNIST
from berries.metric import metrics
from berries.logger import MultiLogger


class MNISTExperiment(Experiment):

    def __init__(self):
        self.params = {
            'project_name': 'debug',
            'experiment_name': 'mnist-float-v2',
            'seed': 42,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'resume': False,
            'pretrained': False,
            'log_interval': 1,
            'stdout_interval': 10,
            'root': Path('./'),
            'neptune_project_name': 'machining/stroke',
        }

        self.hyperparams = {
            'lr': 0.001,
            'batch_size': 10000,
            'validation_batch_size': 10000,
            'epoch': 30,
        }

        self.dataset = MNIST(root='./input/',
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))

        self.model = CNN(in_channels=1, out_channels=10, input_dim=(1, 28, 28))

        self.logger = MultiLogger(
            root=self.params['root'],
            project_name=self.params['project_name'],
            experiment_name=self.params['experiment_name'],
            params=self.params,
            hyperparams=self.hyperparams)

        self.trainer = CNNTrainer(model=self.model,
                                  metrics=[metrics.Accuracy],
                                  hyperparams=self.hyperparams,
                                  params=self.params,
                                  criterion=nn.CrossEntropyLoss(),
                                  logger=self.logger)

    def run(self):
        self.trainer.fit(dataset=self.dataset.trainset,
                         validation_dataset=self.dataset.testset)


def main():
    MNISTExperiment().run()


if __name__ == "__main__":
    main()
