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
        training_scores, validation_scores = [], []
        for step, (X, y) in enumerate(self.train_dataloader):
            # Fit the model
            self.model.fit(X.to(self.config.DEVICE).float(), y.to(self.config.DEVICE).long())

            training_scores += [self.model.score(X.to(self.config.DEVICE).float(),
                                                   y.to(self.config.DEVICE).long())[0]]

            training_loss += [self.model.training_loss.item()]

        X_train, y_train = X,y

        for step, (X, y) in enumerate(self.valid_dataloader):

            # Validate validation set
            self.model.validate(X.to(self.config.DEVICE).float(), y.to(self.config.DEVICE).long())  # todo: current build call .validate when .score is used!

            # Score
            validation_scores += [self.model.score(X.to(self.config.DEVICE).float(),
                                                   y.to(self.config.DEVICE).long())[0]]
            validation_loss += [self.model.validation_loss.item()]

        X_valid, y_valid = X, y


        # Log
        print("========================================\n")
        print("Training Loss: {}".format(np.array(training_loss).mean()))
        print("Validation Loss: {}".format(np.array(validation_loss).mean()))
        print("Training Score: {}".format(np.array(training_scores).mean()))
        print("Validation Score: {}".format(np.array(validation_scores).mean()))
        print("========================================")

        _,_,train_acc = self.predict_and_verbose(sample=self.dataset.random_train_sample(n=100), title='Train Random')
        _,_,valid_acc = self.predict_and_verbose(sample=self.dataset.random_valid_sample(n=100), title='Valid Random')
        _, _, last_valid_acc = self.predict_and_verbose(sample=(X_valid.to(self.config.DEVICE).float(),
                                                           y_valid.to(self.config.DEVICE).long()), title='Last Valid Sampler')
        _, _, last_train_acc = self.predict_and_verbose(sample=(X_train.to(self.config.DEVICE).float(),
                                                                y_train.to(self.config.DEVICE).long()),
                                                        title='Last Train Sampler')




        self.writer.add_scalar('training_loss', np.array(training_loss).mean(), epoch)
        self.writer.add_scalar('validation_loss', np.array(validation_loss).mean(), epoch)

        self.writer.add_scalar('training_score', np.array(training_scores).mean(), epoch)
        self.writer.add_scalar('validation_score', np.array(validation_scores).mean(), epoch)

        self.writer.add_scalar('train_acc', train_acc, epoch)
        self.writer.add_scalar('valid_acc', valid_acc, epoch)

        self.writer.add_scalar('last_train_acc', last_train_acc, epoch)
        self.writer.add_scalar('last_valid_acc', last_valid_acc, epoch)





        return np.array(training_loss).mean(), np.array(validation_loss).mean()

    # Predict
    def predict_and_verbose(self, sample, title):
        X_sample, y_sample = sample
        prediction_acc, predicted_labels = self.model.score(X_sample.to(self.config.DEVICE).float(),
                                                            y_sample.to(self.config.DEVICE).long())

        y_sample = y_sample.cpu().numpy()
        cm = confusion_matrix(y_true=y_sample,
                              y_pred=predicted_labels, labels=[0, 1, 2])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print(f"\n==================={title}=======================\n"
              f"y:\n{y_sample}\n"
              f"prediction:\n{predicted_labels}\n"
              f"accuracy:\n{prediction_acc}\n"
              f"cm:\n{cm}\n"
              f"cm_norm:\n{cm_norm}\n"
              f"per_class_acc:\n"
              f"0: {((predicted_labels == 0) & (y_sample == 0)).sum()}/{(y_sample == 0).sum()}:{(predicted_labels == 0).sum()}\n"
              f"1: {((predicted_labels == 1) & (y_sample == 1)).sum()}/{(y_sample == 1).sum()}:{(predicted_labels == 1).sum()}\n"
              f"2: {((predicted_labels == 2) & (y_sample == 2)).sum()}/{(y_sample == 2).sum()}:{(predicted_labels == 2).sum()}\n"
              f"per_class_acc: {cm_norm[0,0],cm_norm[1,1],cm_norm[2,2]}\n"              
              f"\n=================================================\n")

        return y_sample, predicted_labels, prediction_acc

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
    config.save()
    dataset = datasetfile.IndicatorDataset(dataset_name='IndicatorDataset',
                                           input_path='../dataset/finance/stocks/raw_stocks/spy',
                                           save_dataset=False,
                                           train_valid_ratio=0.8,
                                           seq_len=20,
                                           label_type='classification',
                                           output_path=f"{config.EXPERIMENT_DIR}")
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

