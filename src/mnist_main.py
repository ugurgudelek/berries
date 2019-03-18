# # Import comet_ml in the top of your file
# from comet_ml import Experiment

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

from model import model as modelfile
from dataset import dataset as datasetfile

import collections

# todo: add pr_cruve
# todo: add confusion_matrix
# todo: add weight_distribution


#
# # Create an experiment
# experiment = Experiment(api_key="BVP0cmmwKRcNSwlAqBkEvKbtA",
#                         project_name="general", workspace="ugurgudelek")
#
# # Report any information you need by:
# hyper_params = {"learning_rate": 0.5, "steps": 100000, "batch_size": 50}
# experiment.log_multiple_params(hyper_params)

def custom_collate_fn(batch):
    if isinstance(batch[0], np.ndarray):
        return torch.stack([torch.from_numpy(b) for b in batch], 0)

    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [custom_collate_fn(samples) for samples in transposed]

    elif isinstance(batch[0], dict):
        return pd.DataFrame(list(batch)).to_dict('list')
    else:
        print(type(batch[0]))
        raise Exception('Update custom_collate_fn!!')

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
                                      drop_last=True,
                                           sampler=datasetfile.ImbalancedDatasetSampler(self.dataset.train_dataset))
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

        training_loss, validation_loss = [], []
        scores = []
        for step, (X, y) in enumerate(self.train_dataloader):
            # Fit the model
            self.model.fit(X.to(self.config.DEVICE).float(), y.to(self.config.DEVICE).long())
            training_loss += [self.model.training_loss.item()]


        for step, (X, y) in enumerate(self.valid_dataloader):

            # Validate validation set
            self.model.validate(X.to(self.config.DEVICE).float(), y.to(self.config.DEVICE).long())  # todo: current build call .validate when .score is used!

            # Score
            scores += [self.model.score(X.to(self.config.DEVICE).float(), y.to(self.config.DEVICE).long())]
            validation_loss += [self.model.validation_loss.item()]


        # Predict

        # X_sample, y_sample = self.dataset.random_train_sample(n=100)
        # predicted_labels = self.model.predict(X_sample.to(self.config.DEVICE)).cpu().detach()
        # predicted_labels = prediction_logprob



        # Log
        print("========================================\n")
        print("Training Loss: {}".format(np.array(training_loss).mean()))
        print("Validation Loss: {}".format(np.array(validation_loss).mean()))
        print("Score: {}".format(np.array(scores).mean()))


        return np.array(training_loss).mean(), np.array(validation_loss).mean()


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
    4. Init dataset
    5. Init model
    6. Pass config, model and dataset to Experiment
    7. Run
    """

    # region EXPERIMENT: BookWriter
    config = config.ConfigCNN()
    # config.save()
    dataset = datasetfile.IndicatorDataset(dataset_name='IndicatorDataset',
                                           input_path='../dataset/finance/stocks/raw_stocks/inner',
                                           save_dataset=False,
                                           train_valid_ratio=0.9,
                                           seq_len=20,
                                           label_type='classification')
    model = modelfile.CNN(config=config).to(config.DEVICE)

    # endregion

    # region  EXPERIMENT: SequenceLearningManyToOne
    # config = config.ConfigLSTM()
    # config.save()
    # dataset = dataset.SequenceLearningManyToOne(seq_len=config.SEQ_LEN, seq_limit=config.INPUT_SIZE, dataset_len=1000)
    # model = model.LSTM(input_size=config.INPUT_SIZE, seq_length=config.SEQ_LEN, num_layers=2,
    #                    out_size=config.OUTPUT_SIZE, hidden_size=20, batch_size=config.TRAIN_BATCH_SIZE,
    #                    device=config.DEVICE).to(config.DEVICE)
    # endregion

    experiment = Experiment(config=config, model=model, dataset=dataset)

    experiment.run()

