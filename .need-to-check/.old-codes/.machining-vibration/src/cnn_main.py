# -*- coding: utf-8 -*-
# @Time   : 2/19/2020 6:56 PM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : cnn_main.py


import random
import os
from pathlib import Path
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# pytorch imports
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# scikit-learn imports
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix

# local imports
from dataset import VibrationDataset
from model import CNN2D



def accuracy(y_true, y_pred):
    return torch.mean((torch.argmax(y_pred, dim=1) == y_true).float()).item()

def print_log(per_epoch, epoch, train=True):
    print(f"[{'Train' if train else 'Test'}]  epoch [{epoch + 1}/{num_epochs}], "
          f"loss:{np.array(per_epoch['losses']).mean():.4f}, "
          f"acc:{np.array(per_epoch['accuracies']).mean():.4f}, "
          f"predict_mean:{np.array(per_epoch['predictions']).mean():.4f}, "
          f"label_mean:{np.array(per_epoch['labels']).mean():.4f}, "
          # f'acc_mean10:{test_accuracies_bucket.mean():.4f}',
          )

if __name__ == "__main__":

    EXPERIMENT_NAME = f'bati-cnn-{time.time()}'
    seed = 7
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_epochs = 1000
    batch_size = 2
    learning_rate = 1e-4
    train_ratio = 0.75

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(device)

    dataset = VibrationDataset(path=Path('D:/machining/chatter_data/preprocessed/alu_v1'),
                               train_ratio=train_ratio, kind='acc', shuffle_mode=1)

    train_dataloader = DataLoader(dataset.train_dataset, batch_size=batch_size,
                                  # sampler=ImbalancedDatasetSampler(dataset.train_dataset)
                                  )
    test_dataloader = DataLoader(dataset.test_dataset, batch_size=1)

    model = CNN2D().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train_accuracies_bucket = Bucket()
    # test_accuracies_bucket = Bucket()
    log = {'train_losses': [],
           'test_losses': [],
           'train_accuracies': [],
           'test_accuracies': []}

    for epoch in range(num_epochs):

        per_epoch_train = {'accuracies':[],
                     'losses':[],
                     'predictions':[],
                     'labels':[],
                     }
        model.train(mode=True)
        for img,label in train_dataloader:
            img, label = img.to(device).float(), label.to(device).long()  # classification

            output = model(img)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            per_epoch_train['accuracies'].append(accuracy(y_true=label, y_pred=output))
            per_epoch_train['losses'].append(loss.item())
            per_epoch_train['predictions'].append(list((torch.argmax(output, dim=1).numpy())))
            per_epoch_train['labels'].append(list(label.numpy()))
            # proba = torch.nn.functional.softmax(output.detach(), dim=1).numpy()
            # proba0s += list(proba[:, 0])
            # proba1s += list(proba[:, 1])

        log['train_losses'].append(np.array(per_epoch_train['losses']).mean())
        log['train_accuracies'].append(np.array(per_epoch_train['accuracies']).mean())

        print_log(per_epoch=per_epoch_train, epoch=epoch, train=True)

        train_pred_df = pd.DataFrame({'label': per_epoch_train['labels'],
                                      'pred': per_epoch_train['predictions'],
                                      # 'proba0': proba0s,
                                      # 'proba1': proba1s,
                                      })


        per_epoch_test = {'accuracies':[],
                     'losses':[],
                     'predictions':[],
                     'labels':[],
                     }
        model.train(mode=False)
        for img,label in test_dataloader:
            img, label = img.to(device).float(), label.to(device).long()

            output = model(img)
            loss = criterion(output, label)
            optimizer.zero_grad()

            per_epoch_test['accuracies'].append(accuracy(y_true=label, y_pred=output))
            per_epoch_test['losses'].append(loss.item())
            per_epoch_test['predictions'].append(list((torch.argmax(output, dim=1).numpy())))
            per_epoch_test['labels'].append(list(label.numpy()))
            # proba = torch.nn.functional.softmax(output.detach(), dim=1).numpy()
            # proba0s += list(proba[:, 0])
            # proba1s += list(proba[:, 1])

        log['test_losses'].append(np.array(per_epoch_test['losses']).mean())
        log['test_accuracies'].append(np.array(per_epoch_test['accuracies']).mean())

        print_log(per_epoch=per_epoch_test, epoch=epoch, train=False)

        test_pred_df = pd.DataFrame({'label': per_epoch_test['labels'],
                                      'pred': per_epoch_test['predictions'],
                                      # 'proba0': proba0s,
                                      # 'proba1': proba1s,
                                      })

        if (epoch + 1) % 50 == 0:
            output_dir = f'../results/{EXPERIMENT_NAME}/{epoch}'
            os.makedirs(output_dir, exist_ok=True)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            # plt.clf()

            sns.lineplot(range(log['train_losses'].__len__()), log['train_losses'], label='training loss')
            sns.lineplot(range(log['train_losses'].__len__()), log['test_losses'], label='test loss')
            sns.lineplot(range(log['train_losses'].__len__()), log['train_accuracies'], label='training accuracy')
            sns.lineplot(range(log['train_losses'].__len__()), log['test_accuracies'], label='test accuracy')
            ax.set_xlabel("Epoch")
            # plt.show(block=False)
            # plt.pause(0.001)
            plt.ylim((0., 1.5))
            plt.savefig(os.path.join(output_dir, 'lr_curve.png'))


            # cm_train = confusion_matrix(train_pred_df['label'], train_pred_df['pred'])
            # cm_test = confusion_matrix(test_pred_df['label'], test_pred_df['pred'])
            # utils.plot_confusion_matrix(y_true=train_pred_df['label'],
            #                             y_pred=train_pred_df['pred'],
            #                             classes=['no chatter', 'medium chatter', 'high chatter'],
            #                             save_path=os.path.join(output_dir, 'cm_train.png'))
            # utils.plot_confusion_matrix(y_true=test_pred_df['label'],
            #                             y_pred=test_pred_df['pred'],
            #                             classes=['no chatter', 'medium chatter', 'high chatter'],
            #                             save_path=os.path.join(output_dir, 'cm_test.png'))

            train_pred_df.to_csv(os.path.join(output_dir, 'train_pred.csv'))
            test_pred_df.to_csv(os.path.join(output_dir, 'test_pred.csv'))
            # np.savetxt(os.path.join(output_dir, 'cm_train.txt'), cm_train)
            # np.savetxt(os.path.join(output_dir, 'cm_test.txt'), cm_test)
            pd.DataFrame({'train_loss': log['train_losses'],
                          'test_loss': log['test_losses'],
                          'train_acc': log['train_accuracies'],
                          'test_acc': log['test_accuracies'],
                          }).to_csv(os.path.join(output_dir, 'result.csv'))


    torch.save(model.state_dict(), os.path.join(output_dir, '..', 'vibration_model.pth'))
