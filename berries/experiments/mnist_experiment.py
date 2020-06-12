__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

from pathlib import Path
import numpy as np

import torch
from torchvision import transforms

from berries.logger import MultiLogger, LocalLogger
from berries.trainer import Trainer
from berries.model import CNN
from berries.metric import metrics
from berries.datasets import MNIST


class MNISTExperiment:
    def __init__(self):
        self.params = {'project_name': 'debug',
                       'experiment_name': 'mnist-float',
                       'seed': 42,
                       'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                       'resume': False,
                       'pretrained': False,
                       'log_interval': 5,
                       'stdout_interval': 5,
                       'root': Path('../../'),
                       }

        self.hyperparams = {'lr': 0.001,
                            'train_batch_size': 1000,
                            'test_batch_size': 1000,
                            'epoch': 100,
                            }

        torch.manual_seed(self.params['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.params['seed'])

        self.dataset = MNIST(root='../../input/', transform=transforms.Compose([transforms.ToTensor(),
                                                                                transforms.Normalize((0.1307,),
                                                                                                     (0.3081,))]))

        self.logger = LocalLogger(
            root=self.params['root'],
            project_name=self.params['project_name'],
            experiment_name=self.params['experiment_name'],
            params=self.params,
            hyperparams=self.hyperparams)

        self.model = CNN(in_channels=1, out_channels=10, input_dim=(1, 28, 28))

        self.trainer = Trainer(root=self.params['root'],
                               model=self.model, dataset=self.dataset,
                               metrics=[metrics.Accuracy],
                               hyperparams=self.hyperparams, params=self.params,
                               logger=self.logger,
                               criterion=torch.nn.CrossEntropyLoss())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.stop()

    def run(self):
        self.trainer.fit()


def main():
    with MNISTExperiment() as experiment:
        experiment.run()


if __name__ == "__main__":
    main()
