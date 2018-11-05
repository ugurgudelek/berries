
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

import model
import dataset

# todo: add pr_cruve
# todo: add confusion_matrix
# todo: add weight_distribution



class Experiment:

    def __init__(self, config, model ,dataset):
        self.config = config
        self.model = model.to(self.config.DEVICE)
        self.dataset = dataset

        self.load()
        # self.save()



    def load(self):

        #self.dataset = dataset_by_name(self.config.DATASET_NAME)(config=self.config)  # MNISTDataset, IndicatorDataset, LoadDataset

        self.train_dataloader = DataLoader(self.dataset.train_dataset,
                                      batch_size=self.config.TRAIN_BATCH_SIZE,
                                      shuffle=self.config.TRAIN_SHUFFLE,
                                      drop_last=True)
        self.valid_dataloader = DataLoader(self.dataset.valid_dataset,
                                      batch_size=self.config.VALID_BATCH_SIZE,
                                      shuffle=self.config.VALID_SHUFFLE,
                                      drop_last=True)

        #MODEL = class_by_name(self.config.MODEL_NAME)  # CNN, LSTM
        #self.model = MODEL(config=self.config).to(self.config.DEVICE)

        self.writer = SummaryWriter(log_dir=os.path.join(self.config.EXPERIMENT_DIR, 'summary'))

    def save(self):
        self.config.save()
        self.model.to_onnx(directory=self.config.EXPERIMENT_DIR)
        self.model.to_txt(directory=self.config.EXPERIMENT_DIR)

    def run_epoch(self, epoch):

        self.model.init_hidden()
        for step, (X, y) in enumerate(self.train_dataloader):
            # Fit the model
            self.model.fit(X, y)
            training_loss = self.model.training_loss

        score = 0.
        self.model.init_hidden()
        for step, (X, y) in enumerate(self.valid_dataloader):

            # Validate validation set
            self.model.validate(X, y)  # todo: current build call .validate when .score is used!

            # Score
            score += self.model.score(X, y)
            validation_loss = self.model.validation_loss

        score = score/self.valid_dataloader.__len__()
        # Predict
        X_sample, y_sample = self.dataset.random_train_sample(n=100)
        predicted_labels = self.model.predict(X_sample).cpu().detach()
        # predicted_labels = prediction_logprob



        # Log
        print("========================================")
        print("Training Loss: {}".format(training_loss))
        print("Validation Loss: {}".format(validation_loss))
        print("Score: {}".format(score))

        print('Actual label:', y_sample[:10])
        print('Predicted label:', predicted_labels[:10])
        print("========================================")

        # Write losses to the tensorboard
        self.writer.add_scalar('training_loss', training_loss, epoch)
        self.writer.add_scalar('validation_loss', validation_loss, epoch)

        # # Write random image to the summary writer.
        # image_grid = torchvision.utils.make_grid(X_sample, normalize=True, scale_each=True)
        # self.writer.add_image(tag="RandomSample y-{} yhat{}".format(
        #     '.'.join(map(str, y_sample)), '.'.join(map(str, predicted_labels))),
        #                       img_tensor=image_grid, global_step=epoch)


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

        return training_loss, validation_loss


    def run(self):
        epoch = 0
        with tqdm(total=self.config.EPOCH_SIZE) as pbar:

            for epoch in range(self.config.EPOCH_SIZE):
                tloss,vloss = self.run_epoch(epoch=epoch)
                pbar.set_description("{}||||{}".format(tloss, vloss))
                pbar.update(1)

        self.writer.export_scalars_to_json(self.config.EXPERIMENT_DIR+'.json')





if __name__ == "__main__":

    """
    1. Implement Dataset Class
    2. Implement Model Class
    3. Configure configClass
    4. Pass config to experiment
    5. Run
    """
    # config = config.ConfigCNN()
    #
    # dataset = dataset.MNISTDataset(config)

    config = config.ConfigLSTM()
    # dataset = dataset.SequenceLearningOneToOne()
    # model = model.LSTM(input_size=10, seq_length=1, num_layers=1,
    #                    out_size=10, hidden_size=10, batch_size=1, device=config.DEVICE)

    config.save()

    dataset = dataset.SequenceLearningManyToOne(seq_len=config.SEQ_LEN, seq_limit=config.INPUT_SIZE, onehot=True, dataset_len=1000)
    model = model.LSTM(input_size=config.INPUT_SIZE, seq_length=config.SEQ_LEN, num_layers=2,
                       out_size=config.OUTPUT_SIZE, hidden_size=20, batch_size=config.TRAIN_BATCH_SIZE,
                       device=config.DEVICE)

    # model = model.CNN(config)

    experiment = Experiment(config=config, model=model, dataset=dataset)
    experiment.run()

