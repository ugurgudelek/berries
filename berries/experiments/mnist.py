__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import torch
from torch import nn
from torch.optim import lr_scheduler, Adam
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
            'experiment_name': 'mnist',
            'seed': 42,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'resume': False,
            'pretrained': False,
            'checkpoint': {
                'on_epoch': 2,
                'metric': metrics.Accuracy.__name__.lower(),
                'trigger': lambda new, old: new > old
            },
            'log': {
                'on_epoch': 2,
            },
            'stdout': {
                'verbose': True,
                'on_batch': 0,
                'on_epoch': 2
            },
            'root': Path('./'),
            'neptune': {
                # 'id': 'DEBUG-12',
                'workspace': 'ugurgudelek',
                'project': 'debug',
                'tags': ['MNIST', 'CNN'],
                'source_files': ['./mnist.py']
            }
        }

        self.hyperparams = {
            'lr': 0.001,
            'batch_size': 10000,
            'validation_batch_size': 10000,
            'epoch': 13,
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

        self.optimizer = Adam(params=self.model.parameters(),
                              lr=self.hyperparams.get('lr', 0.001),
                              weight_decay=self.hyperparams.get(
                                  'weight_decay', 0))

        # Decay LR by a factor of 0.1 every 7 epochs
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer,
                                                    step_size=7,
                                                    gamma=0.1,
                                                    verbose=True)

        self.trainer = CNNTrainer(model=self.model,
                                  criterion=nn.CrossEntropyLoss(),
                                  optimizer=self.optimizer,
                                  scheduler=self.exp_lr_scheduler,
                                  metrics=[metrics.Accuracy],
                                  hyperparams=self.hyperparams,
                                  params=self.params,
                                  logger=self.logger)

    def run(self):
        self.trainer.fit(dataset=self.dataset.trainset,
                         validation_dataset=self.dataset.testset)

        # Log final model
        self.logger.log_model(path=self.trainer._get_last_checkpoint_path())

        # Log prediction dataframe
        prediction_dataframe = self.trainer.to_prediction_dataframe(dataset=self.dataset.testset,
                                                                    classification=True,
                                                                    save=True) # yapf:disable

        self.logger.log_dataframe(key='prediction/validation',
                                  dataframe=prediction_dataframe)

        # Log image
        # Example can be found at trainer.py


def main():
    with MNISTExperiment() as experiment:
        experiment.run()


if __name__ == "__main__":
    main()
