__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import torch
from torch import nn

import torch.nn.functional as F

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dataset import TimeSeriesDataset
from torch.utils.data import DataLoader

from sampler import ImbalancedDatasetSampler
import random

import os

import seaborn as sns

from model import NN, LogisticRegression
import skorch

import signal
import sys
import time

from utils import plot_confusion_matrix


class Bucket():
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.container = list()

    def put(self, item):
        if self.__len__() >= self.capacity:
            self.container.pop(0)

        self.container.append(item)

    def mean(self):
        return sum(self.container) / len(self.container)

    def __len__(self):
        return self.container.__len__()


def on_verbose_epoch():

    plt.clf()
    plt.ylim((0., 1.5))
    sns.lineplot(range(train_losses.__len__()), train_losses, label='train loss')
    sns.lineplot(range(train_losses.__len__()), test_losses, label='test loss')
    sns.lineplot(range(train_losses.__len__()), train_accuracies, label='train accuracy')
    sns.lineplot(range(train_losses.__len__()), test_accuracies, label='test accuracy')
    plt.xlabel('# of epoch')
    plt.ylabel('Magnitude')
    plt.savefig(os.path.join(output_dir, 'lr_curve.png'))
    plt.show(block=False)
    plt.pause(0.001)



    cm_train = confusion_matrix(train_pred_df['label'], train_pred_df['pred'])
    cm_test = confusion_matrix(test_pred_df['label'], test_pred_df['pred'])
    train_pred_df.to_csv(os.path.join(output_dir, 'train_pred.csv'))
    test_pred_df.to_csv(os.path.join(output_dir, 'test_pred.csv'))
    np.savetxt(os.path.join(output_dir, 'cm_train.txt'), cm_train)
    np.savetxt(os.path.join(output_dir, 'cm_test.txt'), cm_test)
    pd.DataFrame({'train_loss': train_losses,
                  'test_loss': test_losses,
                  'train_acc': train_accuracies,
                  'test_acc': test_accuracies}).to_csv(os.path.join(output_dir, 'result.csv'))



    plot_confusion_matrix(y_true=train_pred_df['label'],
                          y_pred=train_pred_df['pred'],
                          classes=('no chatter', 'medium chatter', 'severe chatter'),
                          save_path=os.path.join(output_dir, 'train_confusion_matrix.png'),
                          title='Train Confusion Matrix',
                          cmap=plt.get_cmap('Blues')
                          )
    plot_confusion_matrix(y_true=test_pred_df['label'],
                          y_pred=test_pred_df['pred'],
                          classes=('no chatter', 'medium chatter', 'severe chatter'),
                          save_path=os.path.join(output_dir, 'test_confusion_matrix.png'),
                          title='Test Confusion Matrix',
                          cmap=plt.get_cmap('Blues')
                          )




def on_finish():
    on_verbose_epoch()
    plt.savefig(os.path.join(output_dir, '..', 'lr_curve.png'))
    torch.save(model.state_dict(), os.path.join(output_dir, '..', 'vibration_model.pth'))





if __name__ == "__main__":

    EXPERIMENT_NAME = 'cim_paper_verification'
    seed = 7
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_epochs = 150
    train_batch_size = 100
    test_batch_size = 20
    learning_rate = 1e-4
    train_ratio = 0.75

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(device)


    dataset = TimeSeriesDataset.from_readings(reading_path='../input/preprocessed_data/alu_v2',
                                              train_ratio=train_ratio)


    # dataset.feature_heatmap(dataset.FEATURES, dataset.train_dataset)
    # dataset.feature_importance(dataset.FEATURES, dataset.train_dataset, dataset.test_dataset)

    train_dataloader = DataLoader(dataset.train_dataset,
                                  batch_size=train_batch_size,
                                  sampler=ImbalancedDatasetSampler(dataset.train_dataset)
                                  )
    test_dataloader = DataLoader(dataset.test_dataset,
                                 batch_size=test_batch_size)

    model = NN(input_dim=dataset.dataset.features[0].shape[0],
               output_dim=3,
               dropout=0.6).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_accuracies_bucket = Bucket()
    test_accuracies_bucket = Bucket()
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(num_epochs):

        accuracies = []
        losses = []
        predictions = []
        proba0s, proba1s = [], []
        labels = []
        # print(f"lr:{optimizer.param_groups[0]['lr']}")
        # scheduler.step()
        for data in train_dataloader:
            img, label = data[0].to(device).float(), data[1].to(device).long()

            output = model(img)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracies.append(torch.mean((torch.argmax(output, dim=1) == label).float()).item())
            losses.append(loss.item())
            predictions += list((torch.argmax(output, dim=1).numpy()))
            labels += list(label.numpy())
            proba = torch.nn.functional.softmax(output.detach(), dim=1).numpy()
            proba0s += list(proba[:, 0])
            proba1s += list(proba[:, 1])

        train_losses.append(np.array(losses).mean())
        train_accuracies.append(np.array(accuracies).mean())
        train_accuracies_bucket.put(train_accuracies[-1])
        print(f'[Train]  epoch [{epoch + 1}/{num_epochs}], '
              f'loss:{train_losses[-1]:.4f}, '
              f'acc:{train_accuracies[-1]:.4f}, '
              f'predict_mean:{np.array(predictions).mean():.4f}, '
              f'label_mean:{np.array(labels).mean():.4f}, '
              f'acc_mean10:{train_accuracies_bucket.mean():.4f}')

        train_pred_df = pd.DataFrame({'label': labels,
                                      'pred': predictions,
                                      'proba0': proba0s,
                                      'proba1': proba1s})

        accuracies = []
        losses = []
        predictions = []
        proba0s, proba1s = [], []
        labels = []
        model.train(mode=False)
        for data in test_dataloader:
            img, label = data[0].to(device).float(), data[1].to(device).long()
            output = model(img)
            loss = criterion(output, label)
            optimizer.zero_grad()

            accuracies.append(torch.mean((torch.argmax(output, dim=1) == label).float()).item())
            losses.append(loss.item())
            predictions += list((torch.argmax(output, dim=1).numpy()))
            labels += list(label.numpy())
            proba = torch.nn.functional.softmax(output.detach(), dim=1).numpy()
            proba0s += list(proba[:, 0])
            proba1s += list(proba[:, 1])

        test_losses.append(np.array(losses).mean())
        test_accuracies.append(np.array(accuracies).mean())
        test_accuracies_bucket.put(test_accuracies[-1])
        print(f'[Test ]  epoch [{epoch + 1}/{num_epochs}], '
              f'loss:{test_losses[-1]:.4f}, '
              f'acc:{test_accuracies[-1]:.4f}, '
              f'predict_mean:{np.array(predictions).mean():.4f}, '
              f'label_mean:{np.array(labels).mean():.4f}, '
              f'acc_mean10:{test_accuracies_bucket.mean():.4f}')
        model.train(mode=True)
        test_pred_df = pd.DataFrame({'label': labels,
                                     'pred': predictions,
                                     'proba0': proba0s,
                                     'proba1': proba1s})

        if (epoch + 1) % 50 == 0:
            output_dir = f'../results/{EXPERIMENT_NAME}/{epoch}'
            os.makedirs(output_dir, exist_ok=True)
            on_verbose_epoch()
    on_finish()





