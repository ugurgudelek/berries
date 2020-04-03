__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import torch
from torch import nn

import torch.nn.functional as F
from pathlib import Path

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dataset import VibrationDataset
from torch.utils.data import DataLoader

from sampler import ImbalancedDatasetSampler
import random

import os

import seaborn as sns

from model import NN, LogisticRegression, CNN1D
import skorch

import signal
import sys
import time

from utils import plot_confusion_matrix









if __name__ == "__main__":

    EXPERIMENT_NAME = 'utis_paper_verification'
    seed = 7
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_epochs = 670
    train_batch_size = 20
    test_batch_size = 20
    learning_rate = 1e-4
    train_ratio = 0.75

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(device)

    verification_dataset = VibrationDataset(path=Path('../input/preprocessed_data/alu_v1_verification'),
                                            train_ratio=0., kind='acoustic', shuffle_mode=1)


    # dataset.feature_heatmap(dataset.FEATURES, dataset.train_dataset)
    # dataset.feature_importance(dataset.FEATURES, dataset.train_dataset, dataset.test_dataset)

    model = CNN1D().to(device)
    model.load_state_dict(torch.load(f'../results/{EXPERIMENT_NAME}/vibration_model.pth'))
    model.eval()


    def make_prediction(slotnames):
        slotname='all'
        if len(slotnames) == 1:
            slotname = slotnames[0]


        inner_dataset = VibrationDataset.make_inner_dataset(verification_dataset.dataset)
        verification_dataloader = DataLoader(inner_dataset,
                                             batch_size=test_batch_size)
        accuracies = []
        predictions = []
        proba0s, proba1s = [], []
        labels = []

        for data in verification_dataloader:
            img, label = data[0].to(device).float(), data[1].to(device).long()
            output = model(img)

            print(output.shape)
            accuracies.append(torch.mean((torch.argmax(output, dim=1) == label).float()).item())

            predictions += list((torch.argmax(output, dim=1).numpy()))
            labels += list(label.numpy())
            proba = torch.nn.functional.softmax(output.detach(), dim=1).numpy()
            proba0s += list(proba[:, 0])
            proba1s += list(proba[:, 1])


        ver_pred_df = pd.DataFrame({'label': labels,
                                     'pred': predictions,
                                     'proba0': proba0s,
                                     'proba1': proba1s})

        ver_pred_df.to_csv(os.path.join(f'../results/{EXPERIMENT_NAME}', f'ver_pred_{slotname}.csv'))

        plot_confusion_matrix(y_true=ver_pred_df['label'],
                              y_pred=ver_pred_df['pred'],
                              classes=('no chatter', 'medium chatter', 'severe chatter'),
                              save_path=os.path.join(f'../results/{EXPERIMENT_NAME}', f'ver_confusion_matrix_{slotname}.png'),
                              title=f'Verification Confusion Matrix {slotname}',
                              cmap=plt.get_cmap('Blues')
                              )


    # make_prediction([f'kanal{i}' for i in range(1, 7)])  # all

    make_prediction(['kanal1'])
    # make_prediction(['kanal1'])
    # make_prediction(['kanal1'])
    # make_prediction(['kanal1'])


