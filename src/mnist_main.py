
from model import class_by_name
from dataset import dataset_by_name
import config
import torch
from torch import nn

from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader

from tqdm import trange, tqdm
import pandas as pd
import numpy as np

import time

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, confusion_matrix

import os
import torchvision



# todo: add pr_cruve
# todo: add confusion_matrix
# todo: add weight_distribution



class Experiment:

    def __init__(self, config):
        self.config = config

        self.load()
        self.save()



    def load(self):

        self.dataset = dataset_by_name(self.config.DATASET_NAME)(config=self.config)  # MNISTDataset, IndicatorDataset, LoadDataset

        self.train_dataloader = DataLoader(self.dataset.train_dataset,
                                      batch_size=self.config.TRAIN_BATCH_SIZE,
                                      shuffle=self.config.TRAIN_SHUFFLE,
                                      drop_last=True)
        self.valid_dataloader = DataLoader(self.dataset.valid_dataset,
                                      batch_size=self.config.VALID_BATCH_SIZE,
                                      shuffle=self.config.VALID_SHUFFLE,
                                      drop_last=True)

        MODEL = class_by_name(self.config.MODEL_NAME)  # CNN, LSTM
        self.model = MODEL(config=self.config).to(self.config.DEVICE)

        self.writer = SummaryWriter(log_dir=os.path.join(self.config.EXPERIMENT_DIR, 'summary'))

    def save(self):
        self.config.save()
        self.model.to_onnx(directory=self.config.EXPERIMENT_DIR)
        self.model.to_txt(directory=self.config.EXPERIMENT_DIR)

    def run_epoch(self, epoch):

        # Fit the model
        training_loss = self.model.fit(dataloader=self.train_dataloader).item()

        # Validate validation set
        validation_loss = self.model.validate(dataloader=self.valid_dataloader).item()

        # Predict
        images, labels = self.dataset.random_sample(n=16)
        prediction_logprob = self.model.predict(xs=images)[0].cpu().detach()
        predicted_labels = prediction_logprob.max(1, keepdim=True)[1].numpy().flatten()

        # Write losses to the tensorboard
        self.writer.add_scalar('training_loss', training_loss, epoch)
        self.writer.add_scalar('validation_loss', validation_loss, epoch)

        # Write random image to the summary writer.
        image_grid = torchvision.utils.make_grid(images, normalize=True, scale_each=True)
        self.writer.add_image(tag="RandomSample y-{} yhat{}".format(
            '.'.join(map(str, labels)), '.'.join(map(str, predicted_labels))),
                              img_tensor=image_grid, global_step=epoch)


        # Write PR Curve to the summary writer.
        self.writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(100), epoch)

        # for name, param in model.named_parameters():
        #     print(name)
        #     print(param)
        #     model.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch, bins=100)
        # x = dict(model.named_parameters())['conv1.weight'].clone().cpu().data.numpy()
        # kernel1= x[0,0]
        # plt.imshow(kernel1)
        # plt.show()
        # needs tensorboard 0.4RC or later


    def run(self):
        epoch = 0
        with trange(epoch, self.config.EPOCH_SIZE) as t:
            for epoch in t:
                self.run_epoch(epoch=epoch)

        self.writer.export_scalars_to_json(self.config.EXPERIMENT_DIR)





if __name__ == "__main__":

    """
    1. Implement Dataset Class
    2. Implement Model Class
    3. Configure configClass
    4. Pass config to experiment
    5. Run
    """
    experiment = Experiment(config=config.ConfigCNN())
    experiment.run()

