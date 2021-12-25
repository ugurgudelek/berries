__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import torch
from torch import nn
from torch.optim import lr_scheduler, Adam
from torchvision import transforms

import albumentations as A
import albumentations.pytorch as Ap

from pathlib import Path
import copy

from berries.experiments.experiment import Experiment
from berries.trainer.trainer import CNNTrainer
from berries.model.cnn import CNN
from berries.datasets.mnist import MNIST
from berries.metric import metrics
from berries.logger import WandBLogger as logger

import torchmetrics


class MNISTExperiment(Experiment):
    def __init__(self):

        self.params = {
            "entity": "ugurgudelek",
            "project": "sample-project",
            "experiment": "mnist",
            "seed": 42,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "resume": False,
            "pretrained": False,
            "checkpoint": {"metric": metrics.Accuracy.__name__.lower(), "trigger": lambda new, old: new > old},
            "log": {
                "on_epoch": 1,
            },
            "stdout": {"verbose": True, "on_epoch": 1, "on_batch": 10},
            "root": Path("./"),
        }

        self.hyperparams = {
            "lr": 0.001,
            "batch_size": 100,
            "validation_batch_size": 100,
            "epoch": 3,
        }

        self.dataset = MNIST(
            root="./input/",
            transform=A.Compose(
                [
                    #  A.ShiftScaleRotate(shift_limit=0.05,
                    #                     scale_limit=0.05,
                    #                     rotate_limit=15,
                    #                     p=0.5),
                    #  A.HorizontalFlip(p=0.5),
                    #  A.RandomBrightnessContrast(p=0.2),
                    A.Normalize(mean=(0.1307,), std=(0.3081,)),
                    Ap.ToTensorV2(),
                ]
            ),
        )

        self.model = CNN(in_channels=1, out_channels=10, input_dim=(1, 28, 28))

        self.logger = logger(
            project=self.params["project"],
            entity=self.params["entity"],
            config=self.params | self.hyperparams,
        )

        self.logger.watch(self.model)

        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.hyperparams.get("lr", 0.001),
            weight_decay=self.hyperparams.get("weight_decay", 0),
        )

        # Decay LR by a factor of 0.1 every 7 epochs
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1, verbose=True)

        self.trainer = CNNTrainer(
            model=self.model,
            criterion=nn.CrossEntropyLoss(reduction="none"),
            optimizer=self.optimizer,
            scheduler=self.exp_lr_scheduler,
            metrics=[torchmetrics.Accuracy()],
            hyperparams=self.hyperparams,
            params=self.params,
            logger=self.logger,
        )

    def run(self):
        self.trainer.fit(dataset=self.dataset.trainset, validation_dataset=self.dataset.testset)

        # Log final model
        self.logger.log_model(path=self.trainer._get_best_checkpoint_path())

        # Log prediction dataframe
        prediction_dataframe = self.trainer.to_prediction_dataframe(
            dataset=self.dataset.testset, classification=True, save=True
        )  # yapf:disable

        # self.logger.log_dataframe(key="prediction/validation", dataframe=prediction_dataframe)

        # Log image
        # Example can be found at trainer.py


def main():
    with MNISTExperiment() as experiment:
        experiment.run()


if __name__ == "__main__":
    main()
